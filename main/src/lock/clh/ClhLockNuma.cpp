#pragma once

#include <mpi.h>

#include "lock/Lock.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_type_provider.cpp"
#include "mpi_utils/mpi_utils.cpp"

class ClhLockNuma : public Lock {
  static constexpr bool FALSE = false;
  static constexpr int PENDING = -1;
  static constexpr int GRANTED = -2;

  struct global_ptr {
    int rank;
    int disp;
  };
  typedef uint64_t global_ptr_integral;
  static_assert(sizeof(global_ptr) <= sizeof(global_ptr_integral));
  const MPI_Datatype MPI_GLOBAL_PTR = get_mpi_type<global_ptr_integral>();

  struct memory_layout {
    alignas(64) bool locked;
    alignas(64) int state;
    alignas(64) int state2;
    alignas(64) global_ptr tail;  // Request to be watched by next requester
  };
  MpiWindow window;
  const MPI_Aint locked_disp = offsetof(memory_layout, locked) + window.disp;
  const MPI_Aint state_disp = offsetof(memory_layout, state) + window.disp;
  const MPI_Aint state2_disp = offsetof(memory_layout, state2) + window.disp;
  const MPI_Aint tail_disp = offsetof(memory_layout, tail) + window.disp;

  memory_layout &mem;
  const int master_rank;

  global_ptr watch;  // Request that grants lock to me
  global_ptr myreq;  // Request that I grant when thru

  ClhLockNuma(MpiWindow window, int master_rank, int rank)
      : window{std::move(window)},
        mem{*(memory_layout *)window.mem},
        master_rank{master_rank},
        myreq{rank, (int)state_disp} {
    if (rank == master_rank) {
      memory_layout &mem = *(memory_layout *)window.mem;
      mem.state2 = GRANTED;
      mem.tail = {rank, (int)state2_disp};
    }

    MPI_Win_lock_all(0, this->window.win);
    MPI_Barrier(this->window.comm);
  }

 public:
  static constexpr std::string_view NAME() { return "ClhLockNuma"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) { return sizeof(memory_layout); }

  ClhLockNuma(const MPI_Comm comm = MPI_COMM_WORLD, const int master_rank = 0)
      : ClhLockNuma(
            MpiWindow::allocate(get_win_size(get_rank(comm) == master_rank), comm), master_rank) {}

  ClhLockNuma(MpiWindow window, int master_rank = 0)
      : ClhLockNuma(std::move(window), master_rank, get_rank(window.comm)) {}

  ClhLockNuma(ClhLockNuma &&other) = default;

  ~ClhLockNuma() {
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
    MPI_Put(&PENDING, 1, MPI_INT, myreq.rank, myreq.disp, 1, MPI_INT, window.win);
    MPI_Win_flush(myreq.rank, window.win);

    mem.locked = true;
    MPI_Win_sync(window.win);

    // log() << "finding predecessor" << std::endl;
    MPI_Fetch_and_op(
        &myreq, &watch, MPI_GLOBAL_PTR, master_rank, tail_disp, MPI_REPLACE, window.win);
    MPI_Win_flush(master_rank, window.win);

    // log() << "checking for predecessor" << std::endl;
    int state;
    MPI_Fetch_and_op(
        &window.mem_rank, &state, MPI_INT, watch.rank, watch.disp, MPI_REPLACE, window.win);
    MPI_Win_flush(watch.rank, window.win);
    if (state == PENDING) {
#ifdef STATS
      acquired_delayed++;
#endif
      // log() << "waiting for predecessor" << std::endl;
      while (mem.locked) {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        MPI_Win_sync(window.win);
      }
    }
#ifdef STATS
    else
      acquired_immediately++;
#endif
    // log() << "exiting acquire()" << std::endl;
  }

  void release() {
    // log() << "entering release()" << std::endl;
    int successor;
    // Use either FAO or CAS, both work fine.
    // MPI_Fetch_and_op(&GRANTED, &state, MPI_INT,
    //                  myreq.rank, myreq.disp, MPI_REPLACE, win);
    MPI_Compare_and_swap(
        &GRANTED, &PENDING, &successor, MPI_INT, myreq.rank, myreq.disp, window.win);
    MPI_Win_flush(myreq.rank, window.win);
    if (successor != PENDING) {
      // log() << "notifying successor: " << successor << std::endl;
      MPI_Put(&FALSE, 1, MPI_CXX_BOOL, successor, locked_disp, 1, MPI_CXX_BOOL, window.win);
      MPI_Win_flush(successor, window.win);
    }

    myreq = watch;
    // log() << "exiting release()" << std::endl;
  }
};
