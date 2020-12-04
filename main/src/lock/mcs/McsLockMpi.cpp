#pragma once

#include <mpi.h>

#include "lock/Lock.cpp"
#include "log.cpp"
#include "mpi_utils/mpi_utils.cpp"

class McsLockMpi : public Lock {
  static constexpr bool FALSE = false;
  static constexpr bool TRUE = true;
  static constexpr int NULL_RANK = -1;

 public:
  struct memory_layout {
    alignas(64) bool locked;
    alignas(64) int next;
    alignas(64) int tail;
  };
  static constexpr MPI_Aint locked_disp = offsetof(memory_layout, locked);
  static constexpr MPI_Aint next_disp = offsetof(memory_layout, next);
  static constexpr MPI_Aint tail_disp = offsetof(memory_layout, tail);

 private:
  const MPI_Comm comm;
  const int master_rank;
  const int rank;
  MPI_Win win;

  static MPI_Win allocate_win(const MPI_Comm comm, const int master_rank) {
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "accumulate_ordering", "none");
    MPI_Info_set(info, "same_disp_unit", "true");
    MPI_Info_set(info, "same_size", "true");
    memory_layout *mem;
    MPI_Win win;
    MPI_Win_allocate(sizeof(memory_layout), 1, info, comm, &mem, &win);
    return win;
  }

 public:
  static constexpr std::string_view NAME() { return "McsLockMpi"; }

  McsLockMpi(const MPI_Comm comm = MPI_COMM_WORLD, const int master_rank = 0)
      : McsLockMpi(comm, master_rank, get_rank(comm), allocate_win(comm, master_rank)) {}

  McsLockMpi(const MPI_Comm comm, const int master_rank, const int mem_rank, const MPI_Win win)
      : McsLockMpi(comm, master_rank, mem_rank, (memory_layout *)get_win_base(win), win) {}

  McsLockMpi(
      const MPI_Comm comm, const int master_rank, const int mem_rank, memory_layout *const mem,
      const MPI_Win win)
      : comm{comm}, master_rank{master_rank}, rank{get_rank(comm)}, win{win} {
    if (rank == master_rank) {
      mem->tail = NULL_RANK;
    }

    MPI_Win_lock_all(0, win);
    MPI_Barrier(comm);
  }

  McsLockMpi(McsLockMpi &&other) noexcept
      :
#ifdef STATS
        acquired_immediately{other.acquired_immediately},
        acquired_delayed{other.acquired_delayed},
#endif
        comm{other.comm},
        master_rank{other.master_rank},
        rank{other.rank},
        win{other.win} {
    other.win = MPI_WIN_NULL;
  }

  ~McsLockMpi() {
    if (win != MPI_WIN_NULL) {
      MPI_Win_unlock_all(win);
      MPI_Win_free(&win);
    }
  }

  MPI_Comm communicator() { return comm; }

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

  void acquire() {
    // log() << "entering acquire()" << std::endl;
    MPI_Put(&TRUE, 1, MPI_CXX_BOOL, rank, locked_disp, 1, MPI_CXX_BOOL, win);
    MPI_Put(&NULL_RANK, 1, MPI_INT, rank, next_disp, 1, MPI_INT, win);
    MPI_Win_flush(rank, win);

    // log() << "finding predecessor" << std::endl;
    int predecessor;
    MPI_Fetch_and_op(&rank, &predecessor, MPI_INT, master_rank, tail_disp, MPI_REPLACE, win);
    MPI_Win_flush(master_rank, win);
    if (predecessor != NULL_RANK) {
#ifdef STATS
      acquired_delayed++;
#endif
      // log() << "notifying predecessor: " << predecessor << std::endl;
      MPI_Accumulate(&rank, 1, MPI_INT, predecessor, next_disp, 1, MPI_INT, MPI_REPLACE, win);
      MPI_Win_flush(predecessor, win);

      // log() << "waiting for predecessor" << std::endl;
      bool locked;
      do {
#ifdef OPEN_MPI
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
#endif
        MPI_Fetch_and_op(NULL, &locked, MPI_CXX_BOOL, rank, locked_disp, MPI_NO_OP, win);
        MPI_Win_flush(rank, win);
      } while (locked);
    }
#ifdef STATS
    else
      acquired_immediately++;
#endif
    // log() << "exiting acquire()" << std::endl;
  }

  void release() {
    // log() << "entering release()" << std::endl;
    int successor;
    MPI_Fetch_and_op(NULL, &successor, MPI_INT, rank, next_disp, MPI_NO_OP, win);
    MPI_Win_flush(rank, win);
    if (successor == NULL_RANK) {
      // log() << "nulling tail" << std::endl;
      int old_value;
      MPI_Compare_and_swap(&NULL_RANK, &rank, &old_value, MPI_INT, master_rank, tail_disp, win);
      MPI_Win_flush(master_rank, win);
      if (old_value == rank) {
        // log() << "exiting release()" << std::endl;
        return;
      }
      // log() << "waiting for successor" << std::endl;
      do {
#ifdef OPEN_MPI
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
#endif
        MPI_Fetch_and_op(NULL, &successor, MPI_INT, rank, next_disp, MPI_NO_OP, win);
        MPI_Win_flush(rank, win);
      } while (successor == NULL_RANK);
    }
    // log() << "notifying successor: " << successor << std::endl;
    MPI_Accumulate(
        &FALSE, 1, MPI_CXX_BOOL, successor, locked_disp, 1, MPI_CXX_BOOL, MPI_REPLACE, win);
    MPI_Win_flush(successor, win);
    // log() << "exiting release()" << std::endl;
  }
};
