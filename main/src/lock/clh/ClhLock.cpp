#pragma once

#include <mpi.h>

#include "lock/Lock.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_type_provider.cpp"
#include "mpi_utils/mpi_utils.cpp"

class ClhLock : public Lock {
  static constexpr bool FALSE = false;
  static constexpr bool TRUE = true;

  struct global_ptr {
    int rank;
    int disp;
  };
  typedef uint64_t global_ptr_integral;
  static_assert(sizeof(global_ptr) <= sizeof(global_ptr_integral));
  const MPI_Datatype MPI_GLOBAL_PTR = get_mpi_type<global_ptr_integral>();

  struct memory_layout {
    alignas(64) bool locked;
    alignas(64) bool locked2;
    alignas(64) global_ptr tail;  // Request to be watched by next requester
  };
  MpiWindow window;
  const MPI_Aint locked_disp = offsetof(memory_layout, locked) + window.disp;
  const MPI_Aint locked2_disp = offsetof(memory_layout, locked2) + window.disp;
  const MPI_Aint tail_disp = offsetof(memory_layout, tail) + window.disp;

  const int master_rank;

  global_ptr watch;  // Request that grants lock to me
  global_ptr myreq;  // Request that I grant when thru

  ClhLock(MpiWindow window, int master_rank, int rank)
      : window{std::move(window)}, master_rank{master_rank}, myreq{rank, (int)locked_disp} {
    if (rank == master_rank) {
      memory_layout &mem = *(memory_layout *)window.mem;
      mem.locked2 = false;
      mem.tail = {rank, (int)locked2_disp};
    }

    MPI_Win_lock_all(0, this->window.win);
    MPI_Barrier(this->window.comm);
  }

 public:
  static constexpr std::string_view NAME() { return "ClhLock"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) { return sizeof(memory_layout); }

  ClhLock(const MPI_Comm comm = MPI_COMM_WORLD, const int master_rank = 0)
      : ClhLock(
            MpiWindow::allocate(get_win_size(get_rank(comm) == master_rank), comm), master_rank) {}

  ClhLock(MpiWindow window, int master_rank = 0)
      : ClhLock(std::move(window), master_rank, get_rank(window.comm)) {}

  ClhLock(ClhLock &&other) = default;

  ~ClhLock() {
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
    MPI_Put(&TRUE, 1, MPI_CXX_BOOL, myreq.rank, myreq.disp, 1, MPI_CXX_BOOL, window.win);
    MPI_Win_flush(myreq.rank, window.win);

    // log() << "finding predecessor" << std::endl;
    MPI_Fetch_and_op(
        &myreq, &watch, MPI_GLOBAL_PTR, master_rank, tail_disp, MPI_REPLACE, window.win);
    MPI_Win_flush(master_rank, window.win);

    // log() << "waiting for predecessor" << std::endl;
#ifdef STATS
    bool first = true;
#endif
    bool locked;
    do {
      MPI_Fetch_and_op(NULL, &locked, MPI_CXX_BOOL, watch.rank, watch.disp, MPI_NO_OP, window.win);
      MPI_Win_flush(watch.rank, window.win);

#ifdef STATS
      if (!locked)
        if (first)
          acquired_immediately++;
        else
          acquired_delayed++;
      first = false;
#endif
    } while (locked);
    // log() << "exiting acquire()" << std::endl;
  }

  void release() {
    // log() << "entering release()" << std::endl;
    MPI_Accumulate(
        &FALSE, 1, MPI_CXX_BOOL, myreq.rank, myreq.disp, 1, MPI_CXX_BOOL, MPI_REPLACE, window.win);
    MPI_Win_flush(myreq.rank, window.win);

    myreq = watch;
    // log() << "exiting release()" << std::endl;
  }
};
