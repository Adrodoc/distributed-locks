#pragma once

#include <mpi.h>

#include "lock/Lock.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

template <typename G, typename L>
class CohortLock : public Lock {
  static constexpr uint8_t MAX_LOCAL_PASSES = 50;
  static constexpr uint8_t ZERO = 0;
  static constexpr uint8_t ONE = 1;

  struct memory_layout {
    uint8_t local_pass_cnt;
  };
  MpiWindow window;
  const MPI_Aint local_pass_cnt_disp = offsetof(memory_layout, local_pass_cnt) + window.disp;

  const int master_rank;

  G global_lock;
  L local_lock;

  CohortLock(G global_lock, L local_lock, int master_rank, MPI_Comm comm)
      : CohortLock(
            std::move(global_lock), std::move(local_lock), master_rank, comm, get_rank(comm)) {}

  CohortLock(G global_lock, L local_lock, int master_rank, MPI_Comm comm, int rank)
      : global_lock{std::move(global_lock)},
        local_lock{std::move(local_lock)},
        master_rank{master_rank},
        window{MpiWindow::allocate(
            (MPI_Aint)((rank == master_rank ? 1 : 0) * sizeof(memory_layout)), comm)} {
    memory_layout &mem = *(memory_layout *)window.mem;

    if (rank == master_rank) {
      mem.local_pass_cnt = 0;
    }

    MPI_Win_lock_all(0, window.win);
    MPI_Barrier(comm);
  }

 public:
  static std::string NAME() {
    return "CohortLock_" + std::string{G::NAME()} + "_" + std::string{L::NAME()};
  }

  CohortLock(MPI_Comm global_comm = MPI_COMM_WORLD)
      : CohortLock(global_comm, split_comm_shared(global_comm)) {}

  CohortLock(
      MPI_Comm global_comm, MPI_Comm local_comm, int global_master_rank = 0,
      int local_master_rank = 0)
      : CohortLock(
            std::move(G{global_comm, global_master_rank}),
            std::move(L{local_comm, local_master_rank}), global_master_rank) {}

  CohortLock(G global_lock, L local_lock, int master_rank = 0)
      : CohortLock(
            std::move(global_lock), std::move(local_lock), master_rank,
            global_lock.communicator()) {}

  CohortLock(CohortLock &&other) = default;

  ~CohortLock() {
    if (window.win != MPI_WIN_NULL) MPI_Win_unlock_all(window.win);
  }

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
      MPI_Put(&ZERO, 1, MPI_UINT8_T, master_rank, local_pass_cnt_disp, 1, MPI_UINT8_T, window.win);
      MPI_Win_flush(master_rank, window.win);
      global_lock.release();
      local_lock.release_cd(false);
#ifdef STATS
      global_release_cnt++;
#endif
    }
    // log() << "exiting cohort release()" << std::endl;
  }

  bool may_pass_local() {
    uint8_t local_pass_cnt;
    MPI_Fetch_and_op(
        &ONE, &local_pass_cnt, MPI_UINT8_T, master_rank, local_pass_cnt_disp, MPI_SUM, window.win);
    MPI_Win_flush(master_rank, window.win);
    return local_pass_cnt < MAX_LOCAL_PASSES;
  }
};
