#pragma once

#include <mpi.h>

#include "lock/Lock.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

// Non inner class due to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=88165
struct RhLockConfig {
  int fair_factor{1};
  int local_backoff_min{1};
  int local_backoff_max{1};
  int remote_backoff_min{0};
  int remote_backoff_max{0};
};

class RhLock : public Lock {
  static constexpr int FREE = std::numeric_limits<int>::min();
  static constexpr int L_FREE = FREE + 1;

  struct memory_layout {
    int status;
  };
  MpiWindow window;
  const MPI_Aint status_disp = offsetof(memory_layout, status) + window.disp;

  memory_layout &mem;
  const int rank;
  bool be_fair = true;

  const RhLockConfig config;

  RhLock(
      MPI_Comm global_comm, MPI_Comm local_comm, RhLockConfig config, int global_master_rank,
      int local_master_rank, int global_rank, int local_rank)
      : window{MpiWindow::allocate_per_node(
            get_win_size(local_rank == local_master_rank), global_comm, local_comm)},
        mem{*(memory_layout *)window.mem},
        rank{global_rank},
        config{config} {
    if (global_rank == 0) {
      mem.status = FREE;
    } else if (local_rank == local_master_rank) {
      mem.status = encode_remote_rank(0);
    }

    MPI_Win_lock_all(0, this->window.win);
    MPI_Barrier(this->window.comm);
  }

 public:
  static constexpr std::string_view NAME() { return "RhLock"; }

  static constexpr MPI_Aint get_win_size(bool on_local_master_rank) {
    return on_local_master_rank ? sizeof(memory_layout) : 0;
  }

  RhLock(MPI_Comm global_comm = MPI_COMM_WORLD, RhLockConfig config = {})
      : RhLock(global_comm, split_comm_shared(global_comm), config) {}

  RhLock(
      MPI_Comm global_comm, MPI_Comm local_comm, RhLockConfig config = {},
      int global_master_rank = 0, int local_master_rank = 0)
      : RhLock(
            global_comm, local_comm, config, global_master_rank, local_master_rank,
            get_rank(global_comm), get_rank(local_comm)) {}

  RhLock(RhLock &&other) = default;

  ~RhLock() {
    if (window.win != MPI_WIN_NULL) MPI_Win_unlock_all(window.win);
  }

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

  int encode_remote_rank(int remote_rank) { return -remote_rank - 1; }

  int decode_remote_rank(int remote_rank) { return -remote_rank - 1; }

  void acquire() {
    // log() << "entering acquire()" << std::endl;
    int tmp;
    MPI_Fetch_and_op(&rank, &tmp, MPI_INT, window.mem_rank, status_disp, MPI_REPLACE, window.win);
    MPI_Win_flush(window.mem_rank, window.win);
    // log() << "tmp=" << tmp << std::endl;

    if (tmp == L_FREE || tmp == FREE) {
#ifdef STATS
      acquired_immediately++;
#endif
      // log() << "exiting acquire() immediately" << std::endl;
      return;
    }
    if (tmp < 0) {
      rh_acquire_remote_lock(
          decode_remote_rank(tmp)
#ifdef STATS
              ,
          true
#endif
      );
      // log() << "exiting acquire() after acquiring remote lock" << std::endl;
      return;
    }
#ifdef STATS
    acquired_delayed++;
#endif
    rh_acquire_slowpath();
    // log() << "exiting acquire after slowpath()" << std::endl;
  }

  void rh_acquire_slowpath() {
    // log() << "entering rh_acquire_slowpath()" << std::endl;
    int tmp;
    int b = config.local_backoff_min, i;

    if ((random() % config.fair_factor) == 0)
      be_fair = true;
    else
      be_fair = false;
    // log() << "be_fair=" << be_fair << std::endl;

    while (1) {
      for (i = b; i; i--) {  // delay
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      }
      b = std::min(b * 2, config.local_backoff_max);

      // log() << "mem.status=" << mem.status << std::endl;
      if (mem.status >= 0) continue;

      // log() << "Setting " << window.mem_rank << " to " << rank << std::endl;
      MPI_Fetch_and_op(&rank, &tmp, MPI_INT, window.mem_rank, status_disp, MPI_REPLACE, window.win);
      MPI_Win_flush(window.mem_rank, window.win);

      if (tmp == L_FREE || tmp == FREE) break;
      if (tmp < 0) {
        rh_acquire_remote_lock(decode_remote_rank(tmp));
        break;
      }
    }
    // log() << "exiting rh_acquire_slowpath()" << std::endl;
  }

  void rh_acquire_remote_lock(
      int remote_rank
#ifdef STATS
      ,
      bool stats = false
#endif
  ) {
    // log() << "entering rh_acquire_remote_lock(" << remote_rank << ")" << std::endl;
    int b = config.remote_backoff_min, i;

    int status = encode_remote_rank(window.mem_rank);

#ifdef STATS
    bool immediate = true;
#endif
    while (1) {
      int result;
      MPI_Compare_and_swap(&status, &FREE, &result, MPI_INT, remote_rank, status_disp, window.win);
      MPI_Win_flush(remote_rank, window.win);
      if (result == FREE) break;
      if (result < 0 && result != L_FREE) {
        remote_rank = decode_remote_rank(result);
        continue;
      }
#ifdef STATS
      immediate = false;
#endif
      for (i = b; i; i--) {  // delay
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      }
      b = std::min(b * 2, config.remote_backoff_max);
    }
#ifdef STATS
    if (stats)
      if (immediate)
        acquired_immediately++;
      else
        acquired_delayed++;
#endif
    // log() << "CASed " << remote_rank << " to " << status << std::endl;
    // log() << "exiting rh_acquire_remote_lock()" << std::endl;
  }

  void release() {
    // log() << "entering release()" << std::endl;
    // log() << "be_fair=" << be_fair << std::endl;
    if (be_fair) {
      // log() << "Setting " << window.mem_rank << " to FREE" << std::endl;
      MPI_Accumulate(
          &FREE, 1, MPI_INT, window.mem_rank, status_disp, 1, MPI_INT, MPI_REPLACE, window.win);
      MPI_Win_flush(window.mem_rank, window.win);
    } else {
      // log() << "Try setting " << window.mem_rank << " to FREE" << std::endl;
      int result;
      MPI_Compare_and_swap(
          &FREE, &rank, &result, MPI_INT, window.mem_rank, status_disp, window.win);
      MPI_Win_flush(window.mem_rank, window.win);
      if (result != rank) {
        // log() << "Setting " << window.mem_rank << " to L_FREE" << std::endl;
        MPI_Accumulate(
            &L_FREE, 1, MPI_INT, window.mem_rank, status_disp, 1, MPI_INT, MPI_REPLACE, window.win);
        MPI_Win_flush(window.mem_rank, window.win);
      }
    }
    // log() << "exiting release()" << std::endl;
  }
};
