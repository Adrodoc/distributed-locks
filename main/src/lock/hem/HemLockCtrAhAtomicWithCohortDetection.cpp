#pragma once

#include <mpi.h>

#include <atomic>

#include "lock/Lock.cpp"
#include "lock/cohort/CohortLockInlineCounter.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

class HemLockCtrAhAtomicWithCohortDetection : public LockWithInlineCohortDetection {
  static constexpr int NULL_RANK = -1;

  struct memory_layout {
    alignas(64) std::atomic<uint8_t> grant;
    alignas(64) std::atomic<int> tail;
  };

  MpiWindow window;
  memory_layout& mem;
  const int master_rank;
  const int rank;

  HemLockCtrAhAtomicWithCohortDetection(MPI_Comm comm, int master_rank, int rank)
      : HemLockCtrAhAtomicWithCohortDetection(
            MpiWindow::allocate_shared(get_win_size(rank == master_rank), comm), master_rank,
            rank) {}

  HemLockCtrAhAtomicWithCohortDetection(MpiWindow window, int master_rank, int rank)
      : window{std::move(window)},
        mem{*(memory_layout*)window.mem},
        master_rank{master_rank},
        rank{rank} {
    mem.grant.store(WAIT, std::memory_order_relaxed);
    if (rank == master_rank) {
      mem.tail.store(NULL_RANK, std::memory_order_relaxed);
    }
    MPI_Barrier(this->window.comm);
  }

 public:
  static constexpr std::string_view NAME() { return "HemLockCtrAhAtomicWithCohortDetection"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) { return sizeof(memory_layout); }

  HemLockCtrAhAtomicWithCohortDetection(MPI_Comm comm = MPI_COMM_WORLD, int master_rank = 0)
      : HemLockCtrAhAtomicWithCohortDetection(comm, master_rank, get_rank(comm)) {}

  HemLockCtrAhAtomicWithCohortDetection(MpiWindow window, int master_rank = 0)
      : HemLockCtrAhAtomicWithCohortDetection(
            std::move(window), master_rank, get_rank(window.comm)) {}

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

  memory_layout& get_mem(int rank) const {
    int rank_offset = rank - this->rank;
    return *(&mem + rank_offset);
  }

  bool alone() {
    auto& master_mem = get_mem(master_rank);
    return master_mem.tail.load(std::memory_order_relaxed) == rank;
  }

  uint8_t acquire_cd() {
    // log() << "entering acquire()" << std::endl;

    // log() << "finding predecessor" << std::endl;
    auto& master_mem = get_mem(master_rank);
    int predecessor = master_mem.tail.exchange(rank, std::memory_order_acquire);
    if (predecessor != NULL_RANK) {
#ifdef STATS
      acquired_delayed++;
#endif

      // log() << "waiting for predecessor " << predecessor << std::endl;
      auto& predecessor_mem = get_mem(predecessor);

      uint8_t status;
      while ((status = predecessor_mem.grant.exchange(WAIT, std::memory_order_acquire)) == WAIT) {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      }
      return status;
    }
#ifdef STATS
    else
      acquired_immediately++;
#endif
    return ACQUIRE_GLOBAL;
    // log() << "exiting acquire()" << std::endl;
  }

  void release_cd(uint8_t status) {
    // log() << "entering release()" << std::endl;

    // log() << "notifying successor" << std::endl;
    mem.grant.store(status, std::memory_order_release);

    // log() << "nulling tail" << std::endl;
    auto& master_mem = get_mem(master_rank);
    auto expected = rank;
    if (master_mem.tail.compare_exchange_strong(
            expected, NULL_RANK, std::memory_order_release, std::memory_order_relaxed)) {
      // log() << "resetting grant" << std::endl;
      mem.grant.store(WAIT, std::memory_order_relaxed);
    } else {
      // log() << "waiting for successor" << std::endl;
      while (mem.grant.fetch_add(0, std::memory_order_relaxed) != WAIT) {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      }
    }
    // log() << "exiting release()" << std::endl;
  }
};
