#pragma once

#include <mpi.h>

#include <atomic>

#include "lock/Lock.cpp"
#include "log.cpp"
#include "mpi_utils/mpi_utils.cpp"

class RhLockAtomicRef : public Lock {
  static constexpr int BACKOFF_BASE = 625;
  static constexpr int BACKOFF_FACTOR = 2;
  static constexpr int BACKOFF_CAP = 2500;
  static constexpr int REMOTE_BACKOFF_BASE = 2500;
  static constexpr int REMOTE_BACKOFF_CAP = 10000;
  static constexpr int FAIR_FACTOR = 1;

  static constexpr int FREE = std::numeric_limits<int>::min();
  static constexpr int L_FREE = FREE + 1;

  struct memory_layout {
    int status;
  };
  static constexpr MPI_Aint status_disp = offsetof(memory_layout, status);

  // Memory of local master in global window
  memory_layout *mem;

  // Global communicator
  const MPI_Comm global_comm;
  // Global rank of global master
  const int global_master_rank;
  // Global rank
  const int global_rank;
  // Global window
  MPI_Win global_win;

  const MPI_Comm local_comm;
  // Global rank of local master
  int local_master_global_rank;
  // Local rank
  const int local_rank;
  // Local window
  MPI_Win local_win;

  bool be_fair = true;

 public:
  static constexpr std::string_view NAME() { return "RhLockAtomicRef"; }

  RhLockAtomicRef(
      const MPI_Comm comm = MPI_COMM_WORLD, const int global_master_rank = 0,
      const int local_master_rank = 0)
      : global_comm{comm},
        global_master_rank{global_master_rank},
        global_rank{get_rank(global_comm)},
        local_comm{split_comm_shared(comm)},
        local_rank{get_rank(local_comm)} {
    // log () << "entering RhLockAtomicRef" << std::endl;
    MPI_Aint size = (local_rank == local_master_rank ? 1 : 0) * sizeof(memory_layout);

    // Allocate local window
    MPI_Info local_info;
    MPI_Info_create(&local_info);
    MPI_Info_set(local_info, "accumulate_ordering", "none");
    MPI_Info_set(local_info, "same_disp_unit", "true");
    MPI_Info_set(local_info, "alloc_shared_noncontig", "true");
    MPI_Win_allocate_shared(size, 1, local_info, local_comm, &mem, &local_win);

    // Create global window
    MPI_Info global_info;
    MPI_Info_create(&global_info);
    MPI_Info_set(global_info, "accumulate_ordering", "none");
    MPI_Info_set(global_info, "same_disp_unit", "true");
    MPI_Win_create(mem, size, 1, global_info, global_comm, &global_win);

    if (local_rank == local_master_rank) {
      if (global_rank == 0)
        mem->status = FREE;
      else
        mem->status = encode_remote_rank(0);
    } else {
      // Update mem to point to memory of local master rank
      MPI_Aint local_size;
      int local_disp_unit;
      MPI_Win_shared_query(local_win, local_master_rank, &local_size, &local_disp_unit, &mem);
    }

    local_master_global_rank = global_rank;
    MPI_Bcast(&local_master_global_rank, 1, MPI_INT, local_master_rank, local_comm);

    MPI_Win_lock_all(0, global_win);
    MPI_Barrier(comm);
    // log () << "exiting RhLockAtomicRef" << std::endl;
  }

  RhLockAtomicRef(RhLockAtomicRef &&other) noexcept
      : mem{other.mem},
        global_comm{other.global_comm},
        global_master_rank{other.global_master_rank},
        global_rank{other.global_rank},
        global_win{other.global_win},
        local_comm{other.local_comm},
        local_master_global_rank{other.local_master_global_rank},
        local_rank{other.local_rank},
        local_win{other.local_win},
        be_fair{other.be_fair} {
    other.global_win = MPI_WIN_NULL;
    other.local_win = MPI_WIN_NULL;
  }

  ~RhLockAtomicRef() {
    // log () << "entering ~RhLockAtomicRef" << std::endl;
    if (global_win != MPI_WIN_NULL) {
      MPI_Win_unlock_all(global_win);
      MPI_Win_free(&global_win);
    }
    if (local_win != MPI_WIN_NULL) MPI_Win_free(&local_win);
    // log () << "exiting ~RhLockAtomicRef" << std::endl;
  }

  MPI_Comm communicator() { return global_comm; }

  int encode_remote_rank(int remote_rank) { return -remote_rank - 1; }

  int decode_remote_rank(int remote_rank) { return -remote_rank - 1; }

  void acquire() {
    // log () << "entering acquire()" << std::endl;
    std::atomic_ref<int> status{mem->status};
    int tmp = status.exchange(global_rank, std::memory_order_acquire);
    // log () << "tmp=" << tmp << std::endl;

    if (tmp == L_FREE || tmp == FREE) return;
    if (tmp < 0) {
      rh_acquire_remote_lock(decode_remote_rank(tmp));
      return;
    }
    rh_acquire_slowpath();
    // log () << "exiting acquire()" << std::endl;
  }

  void rh_acquire_slowpath() {
    // log () << "entering rh_acquire_slowpath()" << std::endl;
    int tmp;
    int b = BACKOFF_BASE, i;

    if ((random() % FAIR_FACTOR) == 0)
      be_fair = true;
    else
      be_fair = false;

    std::atomic_ref<int> status{mem->status};
    while (1) {
      for (i = b; i; i--)
        ;  // delay
      b = std::min(b * BACKOFF_FACTOR, BACKOFF_CAP);

      int flag;  // Trigger MPI progress
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);

      // log () << "mem->status=" << mem->status << std::endl;
      if (status.load(std::memory_order_relaxed) >= 0) continue;

      tmp = status.exchange(global_rank, std::memory_order_acquire);

      if (tmp == L_FREE || tmp == FREE) break;
      if (tmp < 0) {
        rh_acquire_remote_lock(decode_remote_rank(tmp));
        break;
      }
    }
    // log () << "exiting rh_acquire_slowpath()" << std::endl;
  }

  void rh_acquire_remote_lock(int remote_rank) {
    // log () << "entering rh_acquire_remote_lock()" << std::endl;
    int b = REMOTE_BACKOFF_BASE, i;

    int status = encode_remote_rank(local_master_global_rank);

    while (1) {
      int result;
      MPI_Compare_and_swap(&status, &FREE, &result, MPI_INT, remote_rank, status_disp, global_win);
      MPI_Win_flush(remote_rank, global_win);
      if (result == FREE) break;
      if (result < 0) {
        remote_rank = decode_remote_rank(result);
        continue;
      }
      for (i = b; i; i--)
        ;  // delay
      b = std::min(b * BACKOFF_FACTOR, REMOTE_BACKOFF_CAP);
    }
    // log () << "exiting rh_acquire_remote_lock()" << std::endl;
  }

  void release() {
    // log () << "entering release()" << std::endl;
    // log () << "be_fair=" << be_fair << std::endl;
    std::atomic_ref<int> status{mem->status};
    if (be_fair) {
      status.store(FREE, std::memory_order_release);
    } else {
      int expected = global_rank;
      if (!status.compare_exchange_strong(
              expected, FREE, std::memory_order_release, std::memory_order_relaxed)) {
        status.store(L_FREE, std::memory_order_release);
      }
    }
    // log () << "exiting release()" << std::endl;
  }
};
