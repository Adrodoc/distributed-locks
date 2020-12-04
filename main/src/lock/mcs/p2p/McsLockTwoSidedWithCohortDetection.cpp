#pragma once

#include <mpi.h>

#include "lock/Lock.cpp"
#include "lock/cohort/CohortLockInlineCounter.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

class McsLockTwoSidedWithCohortDetection : public LockWithInlineCohortDetection {
  static constexpr int NULL_RANK = -1;

  struct memory_layout {
    alignas(64) int next;
    alignas(64) int tail;
  };
  MpiWindow window;
  const MPI_Aint next_disp = offsetof(memory_layout, next) + window.disp;
  const MPI_Aint tail_disp = offsetof(memory_layout, tail) + window.disp;

  memory_layout &mem;
  const int master_rank;
  const int rank;

 public:
  static constexpr std::string_view NAME() { return "McsLockTwoSidedWithCohortDetection"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) { return sizeof(memory_layout); }

  static McsLockTwoSidedWithCohortDetection per_node(
      MPI_Comm global_comm, MPI_Comm local_comm, int master_rank = 0) {
    return {MpiWindow::allocate_per_node(
        get_win_size(get_rank(global_comm) == master_rank), global_comm, local_comm)};
  }

  McsLockTwoSidedWithCohortDetection(MPI_Comm comm = MPI_COMM_WORLD, int master_rank = 0)
      : McsLockTwoSidedWithCohortDetection(
            MpiWindow::allocate(get_win_size(get_rank(comm) == master_rank), comm), master_rank) {}

  McsLockTwoSidedWithCohortDetection(MpiWindow window, int master_rank = 0)
      : window{std::move(window)},
        mem{*(memory_layout *)window.mem},
        master_rank{master_rank},
        rank{get_rank(window.comm)} {
    if (rank == master_rank) {
      mem.tail = NULL_RANK;
    }

    MPI_Win_lock_all(0, this->window.win);
    MPI_Barrier(this->window.comm);
  }

  McsLockTwoSidedWithCohortDetection(McsLockTwoSidedWithCohortDetection &&other) = default;

  ~McsLockTwoSidedWithCohortDetection() {
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

  bool alone() { return mem.next == NULL_RANK; }

  uint8_t acquire_cd() {
    // log() << "entering acquire()" << std::endl;
    mem.next = NULL_RANK;
    MPI_Win_sync(window.win);

    // log() << "finding predecessor" << std::endl;
    int predecessor;
    MPI_Fetch_and_op(
        &window.mem_rank, &predecessor, MPI_INT, master_rank, tail_disp, MPI_REPLACE, window.win);
    MPI_Win_flush(master_rank, window.win);
    if (predecessor != NULL_RANK) {
#ifdef STATS
      acquired_delayed++;
#endif
      // log() << "notifying predecessor: " << predecessor << std::endl;
      MPI_Put(&rank, 1, MPI_INT, predecessor, next_disp, 1, MPI_INT, window.win);
      MPI_Win_flush(predecessor, window.win);

      // log() << "waiting for predecessor" << std::endl;
      uint8_t status;
      MPI_Recv(&status, 1, MPI_UINT8_T, MPI_ANY_SOURCE, 0, window.comm, MPI_STATUS_IGNORE);
      return status;
    }
#ifdef STATS
    else
      acquired_immediately++;
#endif
    // log() << "exiting acquire()" << std::endl;
    return ACQUIRE_GLOBAL;
  }

  void release_cd(uint8_t status) {
    // log() << "entering release()" << std::endl;
    int successor = mem.next;
    if (successor == NULL_RANK) {
      // log() << "nulling tail" << std::endl;
      int old_value;
      MPI_Compare_and_swap(
          &NULL_RANK, &window.mem_rank, &old_value, MPI_INT, master_rank, tail_disp, window.win);
      MPI_Win_flush(master_rank, window.win);
      if (old_value == window.mem_rank) {
        // log() << "exiting release()" << std::endl;
        return;
      }
      // log() << "waiting for successor" << std::endl;
      while ((successor = mem.next) == NULL_RANK) {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        MPI_Win_sync(window.win);
      }
    }
    // log() << "notifying successor: " << successor << std::endl;
    MPI_Send(&status, 1, MPI_UINT8_T, successor, 0, window.comm);
    // log() << "exiting release()" << std::endl;
  }
};
