#pragma once

#include <mpi.h>

#include "lock/Lock.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

template <typename G, typename L>
class CohortLockDirectCounter : public Lock {
  static constexpr uint8_t MAX_LOCAL_PASSES = 50;

  struct memory_layout {
    uint8_t local_pass_cnt;
  };
  MpiWindow window;
  memory_layout &mem;

  G global_lock;
  L local_lock;

  CohortLockDirectCounter(
      G global_lock, L local_lock, MPI_Comm global_comm, MPI_Comm local_comm, int local_master_rank)
      : global_lock{std::move(global_lock)},
        local_lock{std::move(local_lock)},
        window{MpiWindow::allocate_per_node(
            sizeof(memory_layout), global_comm, local_comm, local_master_rank)},
        mem{*(memory_layout *)window.mem} {
    MPI_Win_free(&window.win);  // We only need the shared memory

    int rank = get_rank(global_comm);
    if (rank == window.mem_rank) mem.local_pass_cnt = 0;

    MPI_Barrier(global_comm);
  }

 public:
  static std::string NAME() {
    return "CohortLockDirectCounter_" + std::string{G::NAME()} + "_" + std::string{L::NAME()};
  }

  CohortLockDirectCounter(MPI_Comm global_comm = MPI_COMM_WORLD)
      : CohortLockDirectCounter(global_comm, split_comm_shared(global_comm)) {}

  CohortLockDirectCounter(
      MPI_Comm global_comm, MPI_Comm local_comm, int global_master_rank = 0,
      int local_master_rank = 0)
      : CohortLockDirectCounter(
            std::move(G{global_comm, global_master_rank}),
            std::move(L{local_comm, local_master_rank}), local_master_rank) {}

  CohortLockDirectCounter(G global_lock, L local_lock, int local_master_rank = 0)
      : CohortLockDirectCounter(
            std::move(global_lock), std::move(local_lock), global_lock.communicator(),
            local_lock.communicator(), local_master_rank) {}

  MPI_Comm communicator() { return global_lock.communicator(); }

#ifdef STATS
  uint64_t acquired_immediately = 0;
  uint64_t acquired_delayed = 0;
  uint64_t local_release_cnt = 0;
  uint64_t global_release_cnt = 0;

  std::map<std::string, double> stats() {
    double acquired_immediately = this->acquired_immediately;
    double acquired_delayed = this->acquired_delayed;
    double local_release_cnt = this->local_release_cnt;
    double global_release_cnt = this->global_release_cnt;
    this->acquired_immediately = 0;
    this->acquired_delayed = 0;
    this->local_release_cnt = 0;
    this->global_release_cnt = 0;
    std::map<std::string, double> stats = {
        {"acquired_immediately", acquired_immediately},
        {"acquired_delayed", acquired_delayed},
        {"local_release_cnt", local_release_cnt},
        {"global_release_cnt", global_release_cnt},
    };
    std::map<std::string, double> global_stats = global_lock.stats();
    for (const auto &[key, value] : global_stats) {
      stats.insert({"global." + key, value});
    }
    std::map<std::string, double> local_stats = local_lock.stats();
    for (const auto &[key, value] : local_stats) {
      stats.insert({"local." + key, value});
    }
    return stats;
  }
#endif

  void acquire() {
    // log() << "entering cohort acquire()" << std::endl;
#ifdef STATS
    bool immediate = true;
    uint64_t local_acquired_immediately = local_lock.acquired_immediately;
#endif
    bool local_release = local_lock.acquire_cd();
#ifdef STATS
    if (local_acquired_immediately == local_lock.acquired_immediately) immediate = false;
#endif

    // log() << "local release: " << local_release << std::endl;
    if (!local_release) {
#ifdef STATS
      uint64_t global_acquired_immediately = global_lock.acquired_immediately;
#endif
      global_lock.acquire();
#ifdef STATS
      if (global_acquired_immediately == global_lock.acquired_immediately) immediate = false;
#endif
    }

#ifdef STATS
    if (immediate)
      acquired_immediately++;
    else
      acquired_delayed++;
#endif
    // log() << "exiting cohort acquire()" << std::endl;
  }

  void release() {
    // log() << "entering cohort release()" << std::endl;
    bool alone = local_lock.alone();
    // log() << "alone: " << alone << std::endl;
    if (!alone && may_pass_local()) {
      local_lock.release_cd(true);
#ifdef STATS
      local_release_cnt++;
#endif
    } else {
      mem.local_pass_cnt = 0;
      global_lock.release();
      local_lock.release_cd(false);
#ifdef STATS
      global_release_cnt++;
#endif
    }
    // log() << "exiting cohort release()" << std::endl;
  }

  bool may_pass_local() {
    uint8_t local_pass_cnt = mem.local_pass_cnt++;
    return local_pass_cnt < MAX_LOCAL_PASSES;
  }
};
