#pragma once

#include <mpi.h>

#include "lock/Lock.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

class TtsLock : public Lock {
  static constexpr bool FALSE = false;
  static constexpr bool TRUE = true;

  struct memory_layout {
    alignas(64) bool locked;
  };
  MpiWindow window;
  const MPI_Aint locked_disp = offsetof(memory_layout, locked) + window.disp;

  const int master_rank;

  TtsLock(MPI_Comm comm, int master_rank, int rank)
      : TtsLock(MpiWindow::allocate(get_win_size(rank == master_rank), comm), master_rank, rank) {}

  TtsLock(MpiWindow window, int master_rank, int rank)
      : window{std::move(window)}, master_rank{master_rank} {
    if (rank == master_rank) {
      memory_layout &mem = *(memory_layout *)window.mem;
      mem.locked = false;
    }

    MPI_Win_lock_all(0, this->window.win);
    MPI_Barrier(this->window.comm);
  }

 public:
  static constexpr std::string_view NAME() { return "TtsLock"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) {
    return on_master_rank ? sizeof(memory_layout) : 0;
  }

  TtsLock(MPI_Comm comm = MPI_COMM_WORLD, int master_rank = 0)
      : TtsLock(comm, master_rank, get_rank(comm)) {}

  TtsLock(MpiWindow window, int master_rank = 0)
      : TtsLock(std::move(window), master_rank, get_rank(window.comm)) {}

  TtsLock(TtsLock &&other) = default;

  ~TtsLock() {
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

  void acquire() {
    // log() << "entering acquire()" << std::endl;

#ifdef STATS
    uint64_t loops = 0;
#endif
    while (true) {
#ifdef STATS
      loops++;
#endif
      bool locked;
      MPI_Fetch_and_op(
          &TRUE, &locked, MPI_CXX_BOOL, master_rank, locked_disp, MPI_REPLACE, window.win);
      MPI_Win_flush(master_rank, window.win);
      if (!locked) break;
      do {
#ifdef OPEN_MPI
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
#endif
        MPI_Fetch_and_op(
            NULL, &locked, MPI_CXX_BOOL, master_rank, locked_disp, MPI_NO_OP, window.win);
        MPI_Win_flush(master_rank, window.win);
      } while (locked);
    }
#ifdef STATS
    if (loops == 1)
      acquired_immediately++;
    else
      acquired_delayed++;
#endif
    // log() << "exiting acquire()" << std::endl;
  }

  void release() {
    // log() << "entering release()" << std::endl;
    MPI_Accumulate(
        &FALSE, 1, MPI_CXX_BOOL, master_rank, locked_disp, 1, MPI_CXX_BOOL, MPI_REPLACE,
        window.win);
    MPI_Win_flush(master_rank, window.win);
    // log() << "exiting release()" << std::endl;
  }
};
