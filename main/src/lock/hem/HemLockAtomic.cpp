#pragma once

#include <mpi.h>

#include <atomic>

#include "lock/Lock.cpp"
#include "lock/cohort/CohortLockInlineCounter.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

class HemLockAtomic : public LockWithInlineCohortDetection {
  static constexpr int NULL_RANK = -1;
  static constexpr uintptr_t NULL_GRANT = reinterpret_cast<uintptr_t>(nullptr);

  struct memory_layout {
    alignas(64) std::atomic<uintptr_t> grant;
    alignas(64) std::atomic<int> tail;
    alignas(64) uint8_t status;
  };

  MpiWindow window;
  memory_layout& mem;
  const int master_rank;
  const int rank;
  uintptr_t lock_id;

  HemLockAtomic(MPI_Comm comm, int master_rank, int rank)
      : HemLockAtomic(
            MpiWindow::allocate_shared(get_win_size(rank == master_rank), comm), master_rank,
            rank) {}

  HemLockAtomic(MpiWindow window, int master_rank, int rank)
      : window{std::move(window)},
        mem{*(memory_layout*)window.mem},
        master_rank{master_rank},
        rank{rank} {
    mem.grant.store(NULL_GRANT, std::memory_order_relaxed);
    if (rank == master_rank) {
      mem.tail.store(NULL_RANK, std::memory_order_relaxed);
      mem.status = ACQUIRE_GLOBAL;
    }
    lock_id = reinterpret_cast<uintptr_t>(this);
    MPI_Bcast(&lock_id, 1, MPI_UINTPTR_T, master_rank, window.comm);
    // log() << "lock_id=" << lock_id << std::endl;
    MPI_Barrier(this->window.comm);
  }

 public:
  static constexpr std::string_view NAME() { return "HemLockAtomic"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) { return sizeof(memory_layout); }

  HemLockAtomic(MPI_Comm comm = MPI_COMM_WORLD, int master_rank = 0)
      : HemLockAtomic(comm, master_rank, get_rank(comm)) {}

  HemLockAtomic(MpiWindow window, int master_rank = 0)
      : HemLockAtomic(std::move(window), master_rank, get_rank(window.comm)) {}

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
    acquire();
    auto& master_mem = get_mem(master_rank);
    return master_mem.status;
  }

  void acquire() {
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
      while (predecessor_mem.grant.load(std::memory_order_acquire) != lock_id) {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      }

      // log() << "notifying for predecessor " << predecessor << std::endl;
      predecessor_mem.grant.store(NULL_GRANT, std::memory_order_relaxed);
    }
#ifdef STATS
    else
      acquired_immediately++;
#endif
    // log() << "exiting acquire()" << std::endl;
  }

  void release_cd(uint8_t status) {
    auto& master_mem = get_mem(master_rank);
    master_mem.status = status;
    release();
  }

  void release() {
    // log() << "entering release()" << std::endl;

    // log() << "nulling tail" << std::endl;
    auto& master_mem = get_mem(master_rank);
    auto expected = rank;
    if (!master_mem.tail.compare_exchange_strong(
            expected, NULL_RANK, std::memory_order_release, std::memory_order_relaxed)) {
      // log() << "notifying successor" << std::endl;
      mem.grant.store(lock_id, std::memory_order_release);

      // log() << "waiting for successor" << std::endl;
      while (mem.grant.load(std::memory_order_relaxed) != NULL_GRANT) {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      }
    }
    // log() << "exiting release()" << std::endl;
  }
};
