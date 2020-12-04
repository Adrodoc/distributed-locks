#pragma once

#include <mpi.h>

#include "lock/Lock.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

class McsLockWithTtsStealing : public Lock {
  static constexpr int MAX_SHUFFLES = 1024;
  static constexpr bool FALSE = false;
  static constexpr bool TRUE = true;
  static constexpr int NULL_RANK = -1;
  static constexpr uint8_t ONE_u8 = 1;
  static constexpr uint8_t ZERO_u8 = 0;
  static constexpr uint16_t ONE_u16 = 1;
  static constexpr uint16_t ZERO_u16 = 0;

  struct memory_layout {
    alignas(64) bool locked;
    alignas(64) int next;

    alignas(64) uint16_t glock;
    alignas(64) int tail;
  };
  MpiWindow window;
  const MPI_Aint locked_disp = offsetof(memory_layout, locked) + window.disp;
  const MPI_Aint next_disp = offsetof(memory_layout, next) + window.disp;
  const MPI_Aint glock_disp = offsetof(memory_layout, glock) + window.disp;
  const MPI_Aint no_stealing_disp = glock_disp + 1 + window.disp;
  const MPI_Aint tail_disp = offsetof(memory_layout, tail) + window.disp;

  memory_layout &mem;

  const int master_rank;
  const int rank;
  const int node_id;

 public:
  static constexpr std::string_view NAME() { return "McsLockWithTtsStealing"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) { return sizeof(memory_layout); }

  McsLockWithTtsStealing(MPI_Comm comm = MPI_COMM_WORLD, int master_rank = 0)
      : McsLockWithTtsStealing(
            MpiWindow::allocate(get_win_size(get_rank(comm) == master_rank), comm), master_rank) {}

  McsLockWithTtsStealing(MpiWindow window, int master_rank = 0)
      : window{std::move(window)},
        mem{*(memory_layout *)window.mem},
        master_rank{master_rank},
        rank{get_rank(window.comm)},
        node_id{get_node_id(window.comm)} {
    if (rank == master_rank) {
      mem.glock = 0;
      mem.tail = NULL_RANK;
    }
    MPI_Win_lock_all(0, this->window.win);
    MPI_Barrier(this->window.comm);
  }

  McsLockWithTtsStealing(McsLockWithTtsStealing &&other) = default;

  ~McsLockWithTtsStealing() {
    if (window.win != MPI_WIN_NULL) MPI_Win_unlock_all(window.win);
  }

  MPI_Comm communicator() { return window.comm; }

#ifdef STATS
  uint64_t acquired_immediately = 0;
  uint64_t acquired_delayed = 0;

  std::map<std::string, double> stats() {
    double acquired_immediately = this->acquired_immediately;
    double acquired_delayed = this->acquired_delayed;
    this->acquired_immediately = 0;
    this->acquired_delayed = 0;
    return {
        {"acquired_immediately", acquired_immediately},
        {"acquired_delayed", acquired_delayed},
    };
  }
#endif

  void acquire() {
    // log() << "entering acquire()" << std::endl;

    // Try to steal/acquire the lock if there is no lock holder
    uint16_t glock;
    MPI_Fetch_and_op(NULL, &glock, MPI_UINT16_T, master_rank, glock_disp, MPI_NO_OP, window.win);
    MPI_Win_flush(master_rank, window.win);
    if (glock == 0) {
      MPI_Compare_and_swap(
          &ONE_u16, &ZERO_u16, &glock, MPI_UINT16_T, master_rank, glock_disp, window.win);
      MPI_Win_flush(master_rank, window.win);
      if (glock == 0) {
#ifdef STATS
        acquired_immediately++;
#endif
        // log() << "exiting acquire()" << std::endl;
        return;
      }
    }
#ifdef STATS
    acquired_delayed++;
#endif

    // Did not get the node, time to join the queue; initialize node states
    mem.locked = true;
    mem.next = NULL_RANK;
    MPI_Win_sync(window.win);

    // log() << "Atomically adding to the queue tail" << std::endl;
    int qprev;
    MPI_Fetch_and_op(
        &window.mem_rank, &qprev, MPI_INT, master_rank, tail_disp, MPI_REPLACE, window.win);
    MPI_Win_flush(master_rank, window.win);
    if (qprev != NULL_RANK) {  // There are waiters ahead
      spin_until_very_next_waiter(qprev);
    } else {
      // log() << "Disable stealing to maintain the FIFO property" << std::endl;
      // no_stealing is the second byte of glock
      MPI_Accumulate(
          &ONE_u8, 1, MPI_UINT8_T, master_rank, no_stealing_disp, 1, MPI_UINT8_T, MPI_REPLACE,
          window.win);
      MPI_Win_flush(master_rank, window.win);
    }

    // log() << "qnode is at the head of the queue; time to get the TAS lock" << std::endl;
    while (true) {
      // Wait until the lock holder exits the critical section
      uint8_t glock;
      do {
        MPI_Fetch_and_op(NULL, &glock, MPI_UINT8_T, master_rank, glock_disp, MPI_NO_OP, window.win);
        MPI_Win_flush(master_rank, window.win);
#ifdef OPEN_MPI
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
#endif
      } while (glock == 1);

      // Try to atomically get the lock
      MPI_Compare_and_swap(
          &ONE_u8, &ZERO_u8, &glock, MPI_UINT8_T, master_rank, glock_disp, window.win);
      MPI_Win_flush(master_rank, window.win);
      if (glock == 0) break;
    }

    // MCS unlock phase is moved here
    auto qnext = mem.next;
    if (qnext == NULL_RANK) {  // qnode is the last one / next pointer is being updated
      // log() << "Last one in the queue, reset the tail" << std::endl;
      int tail;
      MPI_Compare_and_swap(
          &NULL_RANK, &window.mem_rank, &tail, MPI_INT, master_rank, tail_disp, window.win);
      MPI_Win_flush(master_rank, window.win);
      if (tail == window.mem_rank) {
        // log() << "Try resetting, else someone joined" << std::endl;
        uint8_t no_stealing;
        MPI_Compare_and_swap(
            &ZERO_u8, &ONE_u8, &no_stealing, MPI_UINT8_T, master_rank, no_stealing_disp,
            window.win);
        MPI_Win_flush(master_rank, window.win);
        // log() << "exiting acquire()" << std::endl;
        return;
      }

      // log() << "waiting for successor" << std::endl;
      while ((qnext = mem.next) == NULL_RANK) {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        MPI_Win_sync(window.win);
      }
    }
    // log() << "Notify the very next waiter: " << qnext << std::endl;
    MPI_Put(&FALSE, 1, MPI_CXX_BOOL, qnext, locked_disp, 1, MPI_CXX_BOOL, window.win);
    MPI_Win_flush(qnext, window.win);
    // log() << "exiting acquire()" << std::endl;
  }

  void spin_until_very_next_waiter(int qprev) {
    // log() << "notifying qprev: " << qprev << std::endl;
    MPI_Put(&window.mem_rank, 1, MPI_INT, qprev, next_disp, 1, MPI_INT, window.win);
    MPI_Win_flush(qprev, window.win);

    // log() << "waiting for qprev" << std::endl;
    while (mem.locked) {
      int flag;  // Trigger MPI progress
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      MPI_Win_sync(window.win);
    }
  }

  void release() {
    // log() << "entering release()" << std::endl;
    // no_stealing is not overwritten
    MPI_Accumulate(
        &ZERO_u8, 1, MPI_UINT8_T, master_rank, glock_disp, 1, MPI_UINT8_T, MPI_REPLACE, window.win);
    MPI_Win_flush(master_rank, window.win);
    // log() << "exiting release()" << std::endl;
  }
};
