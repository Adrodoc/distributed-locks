#pragma once

#include <mpi.h>

#include <atomic>

#include "lock/Lock.cpp"
#include "lock/cohort/CohortLockInlineCounter.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

class McsLockAtomicWithCohortDetection : public LockWithInlineCohortDetection {
  static constexpr int NULL_RANK = -1;

  struct memory_layout {
    alignas(64) std::atomic<uint8_t> status;
    alignas(64) std::atomic<int> next;
    alignas(64) std::atomic<int> tail;
  };

  MpiWindow window;
  memory_layout& mem;
  const int master_rank;
  const int rank;

  McsLockAtomicWithCohortDetection(MPI_Comm comm, int master_rank, int rank)
      : McsLockAtomicWithCohortDetection(
            MpiWindow::allocate_shared(get_win_size(rank == master_rank), comm), master_rank,
            rank) {}

  McsLockAtomicWithCohortDetection(MpiWindow window, int master_rank, int rank)
      : window{std::move(window)},
        mem{*(memory_layout*)window.mem},
        master_rank{master_rank},
        rank{rank} {
    if (rank == master_rank) {
      mem.tail.store(NULL_RANK, std::memory_order_relaxed);
    }
    MPI_Barrier(this->window.comm);
  }

 public:
  static constexpr std::string_view NAME() { return "McsLockAtomicWithCohortDetection"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) { return sizeof(memory_layout); }

  McsLockAtomicWithCohortDetection(MPI_Comm comm = MPI_COMM_WORLD, int master_rank = 0)
      : McsLockAtomicWithCohortDetection(comm, master_rank, get_rank(comm)) {}

  McsLockAtomicWithCohortDetection(MpiWindow window, int master_rank = 0)
      : McsLockAtomicWithCohortDetection(std::move(window), master_rank, get_rank(window.comm)) {}

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

  bool alone() { return mem.next.load(std::memory_order_relaxed) == NULL_RANK; }

  uint8_t acquire_cd() {
    // log() << "entering acquire()" << std::endl;
    mem.next.store(NULL_RANK, std::memory_order_relaxed);

    // log() << "finding predecessor" << std::endl;
    auto& master_mem = get_mem(master_rank);
    int predecessor = master_mem.tail.exchange(rank, std::memory_order_acq_rel);
    if (predecessor != NULL_RANK) {
#ifdef STATS
      acquired_delayed++;
#endif
      mem.status.store(WAIT, std::memory_order_relaxed);

      // log() << "notifying predecessor: " << predecessor << std::endl;
      auto& predecessor_mem = get_mem(predecessor);
      predecessor_mem.next.store(rank, std::memory_order_release);

      // log() << "waiting for predecessor" << std::endl;
      uint8_t status;
      while ((status = mem.status.load(std::memory_order_acquire)) == WAIT) {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      }
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
    int successor = mem.next.load(std::memory_order_acquire);
    if (successor == NULL_RANK) {
      // log() << "nulling tail" << std::endl;
      auto& master_mem = get_mem(master_rank);
      auto expected = rank;
      if (master_mem.tail.compare_exchange_strong(
              expected, NULL_RANK, std::memory_order_release, std::memory_order_relaxed)) {
        // log() << "exiting release()" << std::endl;
        return;
      }
      // log() << "waiting for successor" << std::endl;
      while ((successor = mem.next.load(std::memory_order_acquire)) == NULL_RANK) {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      }
    }
    // log() << "notifying successor: " << successor << std::endl;
    auto& successor_mem = get_mem(successor);
    successor_mem.status.store(status, std::memory_order_release);
    // log() << "exiting release()" << std::endl;
  }
};
