#pragma once

#include <mpi.h>

#include <atomic>

#include "lock/Lock.cpp"
#include "lock/cohort/CohortLockInlineCounter.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

class ClhLockAtomicWithCohortDetection : public LockWithInlineCohortDetection {
  struct global_ptr {
    int rank;
    int disp;

    constexpr bool operator==(const global_ptr& other) const noexcept {
      return rank == other.rank && disp == other.disp;
    };
  };

  struct memory_layout {
    alignas(64) std::atomic<uint8_t> status;
    alignas(64) std::atomic<uint8_t> status2;
    alignas(64) std::atomic<global_ptr> tail;
  };

  MpiWindow window;
  memory_layout& mem;
  const int master_rank;
  const int rank;

  global_ptr watch;  // Request that grants lock to me
  global_ptr myreq;  // Request that I grant when thru

  ClhLockAtomicWithCohortDetection(MPI_Comm comm, int master_rank, int rank)
      : ClhLockAtomicWithCohortDetection(
            MpiWindow::allocate_shared(get_win_size(rank == master_rank), comm), master_rank,
            rank) {}

  ClhLockAtomicWithCohortDetection(MpiWindow window, int master_rank, int rank)
      : window{std::move(window)},
        mem{*(memory_layout*)window.mem},
        master_rank{master_rank},
        rank{rank},
        myreq{rank, offsetof(memory_layout, status)} {
    if (rank == master_rank) {
      mem.status2.store(ACQUIRE_GLOBAL, std::memory_order_relaxed);
      mem.tail.store({rank, offsetof(memory_layout, status2)}, std::memory_order_relaxed);
    }
    MPI_Barrier(this->window.comm);
  }

 public:
  static constexpr std::string_view NAME() { return "ClhLockAtomicWithCohortDetection"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) { return sizeof(memory_layout); }

  ClhLockAtomicWithCohortDetection(MPI_Comm comm = MPI_COMM_WORLD, int master_rank = 0)
      : ClhLockAtomicWithCohortDetection(comm, master_rank, get_rank(comm)) {}

  ClhLockAtomicWithCohortDetection(MpiWindow window, int master_rank = 0)
      : ClhLockAtomicWithCohortDetection(std::move(window), master_rank, get_rank(window.comm)) {}

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

  void* get_mem(global_ptr ptr) const {
    auto& mem = get_mem(ptr.rank);
    return (((uint8_t*)&mem) + ptr.disp);
  }

  bool alone() {
    auto& master_mem = get_mem(master_rank);
    return master_mem.tail.load(std::memory_order_relaxed) == myreq;
  }

  uint8_t acquire_cd() {
    // log() << "entering acquire()" << std::endl;
    auto& myreq_mem = *(std::atomic<uint8_t>*)get_mem(myreq);
    myreq_mem.store(WAIT, std::memory_order_relaxed);

    // log() << "finding predecessor" << std::endl;
    auto& master_mem = get_mem(master_rank);
    watch = master_mem.tail.exchange(myreq, std::memory_order_acq_rel);

    // log() << "waiting for predecessor" << std::endl;
    uint8_t status;
#ifdef STATS
    bool first = true;
#endif
    auto& watch_mem = *(std::atomic<uint8_t>*)get_mem(watch);
    while ((status = watch_mem.load(std::memory_order_acquire)) == WAIT) {
      int flag;  // Trigger MPI progress
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
#ifdef STATS
      first = false;
#endif
    }
#ifdef STATS
    if (first)
      acquired_immediately++;
    else
      acquired_delayed++;
#endif
    return status;
    // log() << "exiting acquire()" << std::endl;
  }

  void release_cd(uint8_t status) {
    // log() << "entering release()" << std::endl;
    auto& myreq_mem = *(std::atomic<uint8_t>*)get_mem(myreq);
    myreq_mem.store(status, std::memory_order_release);

    myreq = watch;
    // log() << "exiting release()" << std::endl;
  }
};
