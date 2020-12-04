#pragma once

#include <mpi.h>

#include <atomic>

#include "lock/Lock.cpp"
#include "lock/cohort/CohortLockInlineCounter.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

class McsLockTwoSidedAtomic : public Lock {
  static constexpr int NULL_RANK = -1;

  struct memory_layout {
    alignas(64) std::atomic<int> next;
    alignas(64) std::atomic<int> tail;
  };

  MpiWindow window;
  memory_layout& mem;
  const int master_rank;
  const int rank;

  McsLockTwoSidedAtomic(MPI_Comm comm, int master_rank, int rank)
      : McsLockTwoSidedAtomic(
            MpiWindow::allocate_shared(get_win_size(rank == master_rank), comm), master_rank,
            rank) {}

  McsLockTwoSidedAtomic(MpiWindow window, int master_rank, int rank)
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
  static constexpr std::string_view NAME() { return "McsLockTwoSidedAtomic"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) { return sizeof(memory_layout); }

  McsLockTwoSidedAtomic(MPI_Comm comm = MPI_COMM_WORLD, int master_rank = 0)
      : McsLockTwoSidedAtomic(comm, master_rank, get_rank(comm)) {}

  McsLockTwoSidedAtomic(MpiWindow window, int master_rank = 0)
      : McsLockTwoSidedAtomic(std::move(window), master_rank, get_rank(window.comm)) {}

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

  memory_layout& get_mem(int rank) {
    int rank_offset = rank - this->rank;
    return *(&mem + rank_offset);
  }

  void acquire() {
    // log() << "entering acquire()" << std::endl;
    mem.next.store(NULL_RANK, std::memory_order_relaxed);

    // log() << "finding predecessor" << std::endl;
    auto& master_mem = get_mem(master_rank);
    int predecessor = master_mem.tail.exchange(rank, std::memory_order_acq_rel);
    if (predecessor != NULL_RANK) {
#ifdef STATS
      acquired_delayed++;
#endif
      // log() << "notifying predecessor: " << predecessor << std::endl;
      auto& predecessor_mem = get_mem(predecessor);
      predecessor_mem.next.store(rank, std::memory_order_relaxed);

      // log() << "waiting for predecessor" << std::endl;
      MPI_Recv(NULL, 0, MPI_UINT8_T, predecessor, 0, window.comm, MPI_STATUS_IGNORE);
    }
#ifdef STATS
    else
      acquired_immediately++;
#endif
    // log() << "exiting acquire()" << std::endl;
  }

  void release() {
    // log() << "entering release()" << std::endl;
    int successor = mem.next.load(std::memory_order_relaxed);
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
      while ((successor = mem.next.load(std::memory_order_relaxed)) == NULL_RANK) {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      }
    }
    // log() << "notifying successor: " << successor << std::endl;
    MPI_Send(NULL, 0, MPI_UINT8_T, successor, 0, window.comm);
    // log() << "exiting release()" << std::endl;
  }
};
