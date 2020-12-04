#pragma once

#include <mpi.h>

#include <atomic>

#include "lock/Lock.cpp"
#include "lock/cohort/CohortLockInlineCounter.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

class TktLockAtomicWithCohortDetection : public LockWithInlineCohortDetection {
  struct ticket_and_cnt {
    int ticket;
    uint8_t local_pass_cnt;
  };
  struct memory_layout {
    alignas(64) std::atomic<int> next_ticket;
    alignas(64) std::atomic<ticket_and_cnt> now_serving;
  };

  MpiWindow window;
  memory_layout &mem;

  TktLockAtomicWithCohortDetection(MPI_Comm comm, int master_rank, int rank)
      : TktLockAtomicWithCohortDetection(
            MpiWindow::allocate_shared(get_win_size(rank == master_rank), comm, master_rank),
            master_rank, rank) {}

  TktLockAtomicWithCohortDetection(MpiWindow window, int master_rank, int rank)
      : window{std::move(window)}, mem{*(memory_layout *)window.mem} {
    if (rank == master_rank) {
      mem.next_ticket.store(0, std::memory_order_relaxed);
      mem.now_serving.store({0, ACQUIRE_GLOBAL}, std::memory_order_relaxed);
    }
    MPI_Barrier(this->window.comm);
  }

 public:
  static constexpr std::string_view NAME() { return "TktLockAtomicWithCohortDetection"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) {
    return on_master_rank ? sizeof(memory_layout) : 0;
  }

  TktLockAtomicWithCohortDetection(MPI_Comm comm = MPI_COMM_WORLD, int master_rank = 0)
      : TktLockAtomicWithCohortDetection(comm, master_rank, get_rank(comm)) {}

  TktLockAtomicWithCohortDetection(MpiWindow window, int master_rank = 0)
      : TktLockAtomicWithCohortDetection(std::move(window), master_rank, get_rank(window.comm)) {}

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

  bool alone() {
    return mem.now_serving.load(std::memory_order_relaxed).ticket ==
           mem.next_ticket.load(std::memory_order_relaxed) - 1;
  }

  uint8_t acquire_cd() {
    // log() << "entering acquire()" << std::endl;
    int my_ticket = mem.next_ticket.fetch_add(1, std::memory_order_relaxed);

#ifdef STATS
    bool immediate = true;
#endif
    // log() << "waiting for ticket: " << my_ticket << std::endl;
    ticket_and_cnt now_serving;
    while (my_ticket != (now_serving = mem.now_serving.load(std::memory_order_acquire)).ticket) {
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
    return now_serving.local_pass_cnt;
    // log() << "exiting acquire()" << std::endl;
  }

  void release_cd(uint8_t local_pass_cnt) {
    // log() << "entering release()" << std::endl;
    ticket_and_cnt now_serving = mem.now_serving.load(std::memory_order_relaxed);
    now_serving.local_pass_cnt = local_pass_cnt;
    now_serving.ticket += 1;
    mem.now_serving.store(now_serving, std::memory_order_release);
    // log() << "exiting release()" << std::endl;
  }
};
