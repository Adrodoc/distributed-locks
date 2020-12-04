#pragma once

#include <mpi.h>

#include <atomic>

#include "lock/Lock.cpp"
#include "lock/cohort/CohortLockInlineCounter.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

class TasLockAtomicWithCohortDetection : public LockWithInlineCohortDetection {
  struct memory_layout {
    alignas(64) std::atomic<uint8_t> status;
    alignas(64) std::atomic<bool> successor_exists;
  };

  MpiWindow window;
  memory_layout &mem;

  TasLockAtomicWithCohortDetection(MPI_Comm comm, int master_rank, int rank)
      : TasLockAtomicWithCohortDetection(
            MpiWindow::allocate_shared(get_win_size(rank == master_rank), comm, master_rank),
            master_rank, rank) {}

  TasLockAtomicWithCohortDetection(MpiWindow window, int master_rank, int rank)
      : window{std::move(window)}, mem{*(memory_layout *)window.mem} {
    if (rank == master_rank) {
      mem.status.store(ACQUIRE_GLOBAL, std::memory_order_relaxed);
      mem.successor_exists.store(false, std::memory_order_relaxed);
    }
    MPI_Barrier(this->window.comm);
  }

 public:
  static constexpr std::string_view NAME() { return "TasLockAtomicWithCohortDetection"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) {
    return on_master_rank ? sizeof(memory_layout) : 0;
  }

  TasLockAtomicWithCohortDetection(MPI_Comm comm = MPI_COMM_WORLD, int master_rank = 0)
      : TasLockAtomicWithCohortDetection(comm, master_rank, get_rank(comm)) {}

  TasLockAtomicWithCohortDetection(MpiWindow window, int master_rank = 0)
      : TasLockAtomicWithCohortDetection(std::move(window), master_rank, get_rank(window.comm)) {}

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

  bool alone() { return !mem.successor_exists.load(std::memory_order_relaxed); }

  uint8_t acquire_cd() {
    // log() << "entering acquire()" << std::endl;
#ifdef STATS
    bool immediate = true;
#endif

    mem.successor_exists.store(true, std::memory_order_relaxed);

    uint8_t status;
    while ((status = mem.status.exchange(WAIT, std::memory_order_acquire)) == WAIT) {
#ifdef STATS
      immediate = false;
#endif
      if (!mem.successor_exists.load(std::memory_order_relaxed)) {
        mem.successor_exists.store(true, std::memory_order_relaxed);
      }
      int flag;  // Trigger MPI progress
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    }

    mem.successor_exists.store(false, std::memory_order_relaxed);

#ifdef STATS
    if (immediate)
      acquired_immediately++;
    else
      acquired_delayed++;
#endif

    return status;
    // log() << "exiting acquire()" << std::endl;
  }

  void release_cd(uint8_t status) {
    // log() << "entering release()" << std::endl;
    mem.status.store(status, std::memory_order_release);
    // log() << "exiting release()" << std::endl;
  }
};
