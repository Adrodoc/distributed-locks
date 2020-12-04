#pragma once

#include <mpi.h>

#include <atomic>

#include "lock/Lock.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

class TasLockAtomic : public Lock {
  struct memory_layout {
    alignas(64) std::atomic_flag locked;
  };

  MpiWindow window;
  memory_layout &mem;

  TasLockAtomic(MPI_Comm comm, int master_rank, int rank)
      : TasLockAtomic(
            MpiWindow::allocate_shared(get_win_size(rank == master_rank), comm, master_rank),
            master_rank, rank) {}

  TasLockAtomic(MpiWindow window, int master_rank, int rank)
      : window{std::move(window)}, mem{*(memory_layout *)window.mem} {
    if (rank == master_rank) {
      mem.locked.clear(std::memory_order_relaxed);
    }
    MPI_Barrier(this->window.comm);
  }

 public:
  static constexpr std::string_view NAME() { return "TasLockAtomic"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) {
    return on_master_rank ? sizeof(memory_layout) : 0;
  }

  TasLockAtomic(MPI_Comm comm = MPI_COMM_WORLD, int master_rank = 0)
      : TasLockAtomic(comm, master_rank, get_rank(comm)) {}

  TasLockAtomic(MpiWindow window, int master_rank = 0)
      : TasLockAtomic(std::move(window), master_rank, get_rank(window.comm)) {}

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

#ifdef STATS
    bool immediate = true;
#endif
    while (mem.locked.test_and_set(std::memory_order_acquire)) {
#ifdef STATS
      immediate = false;
#endif
      int flag;  // Trigger MPI progress
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    }
#ifdef STATS
    if (immediate)
      acquired_immediately++;
    else
      acquired_delayed++;
#endif
    // log() << "exiting acquire()" << std::endl;
  }

  void release() {
    // log() << "entering release()" << std::endl;
    mem.locked.clear(std::memory_order_release);
    // log() << "exiting release()" << std::endl;
  }
};
