#pragma once

#include <mpi.h>

#include "lock/Lock.cpp"
#include "log.cpp"
#include "mpi_utils/mpi_utils.cpp"

class LockWithInlineCohortDetection : public Lock {
 public:
  static constexpr uint8_t ACQUIRE_GLOBAL = 0;
  static constexpr uint8_t WAIT = -1;

  virtual bool alone() = 0;

  virtual uint8_t acquire_cd() = 0;

  virtual void acquire() { acquire_cd(); }

  virtual void release() { release_cd(ACQUIRE_GLOBAL); }

  virtual void release_cd(uint8_t status) = 0;
};

template <typename G, typename L>
class CohortLockInlineCounter : public Lock {
  static constexpr uint8_t MAX_LOCAL_PASSES = 50;

  G global_lock;
  L local_lock;

  uint8_t status;

 public:
  static std::string NAME() {
    return "CohortLockInlineCounter_" + std::string{G::NAME()} + "_" + std::string{L::NAME()};
  }

  CohortLockInlineCounter(MPI_Comm global_comm = MPI_COMM_WORLD)
      : CohortLockInlineCounter(global_comm, split_comm_shared(global_comm)) {}

  CohortLockInlineCounter(
      MPI_Comm global_comm, MPI_Comm local_comm, int global_master_rank = 0,
      int local_master_rank = 0)
      : CohortLockInlineCounter(
            std::move(G{global_comm, global_master_rank}),
            std::move(L{local_comm, local_master_rank})) {}

  CohortLockInlineCounter(G global_lock, L local_lock)
      : global_lock{std::move(global_lock)}, local_lock{std::move(local_lock)} {}

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
    status = local_lock.acquire_cd();
#ifdef STATS
    if (local_acquired_immediately == local_lock.acquired_immediately) immediate = false;
#endif

    // log() << "status: " << +status << std::endl;
    if (status == LockWithInlineCohortDetection::ACQUIRE_GLOBAL) {
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
      local_lock.release_cd(status + 1);
#ifdef STATS
      local_release_cnt++;
#endif
    } else {
      // log() << "global release after " << status << " local passes" << std::endl;
      global_lock.release();
      local_lock.release_cd(LockWithInlineCohortDetection::ACQUIRE_GLOBAL);
#ifdef STATS
      global_release_cnt++;
#endif
    }
    // log() << "exiting cohort release()" << std::endl;
  }

  bool may_pass_local() { return status < MAX_LOCAL_PASSES; }
};
