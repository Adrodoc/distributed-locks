#pragma once

#include <mpi.h>

#include "lock/Lock.cpp"
#include "log.cpp"
#include "mpi_utils/MpiWindow.cpp"
#include "mpi_utils/mpi_utils.cpp"

class TktLock : public Lock {
  static constexpr int ONE = 1;

  struct memory_layout {
    alignas(64) int next_ticket;
    alignas(64) int now_serving;
  };
  MpiWindow window;
  const MPI_Aint next_ticket_disp = offsetof(memory_layout, next_ticket) + window.disp;
  const MPI_Aint now_serving_disp = offsetof(memory_layout, now_serving) + window.disp;

  const int master_rank;

  TktLock(MPI_Comm comm, int master_rank, int rank)
      : TktLock(MpiWindow::allocate(get_win_size(rank == master_rank), comm), master_rank, rank) {}

  TktLock(MpiWindow window, int master_rank, int rank)
      : window{std::move(window)}, master_rank{master_rank} {
    if (rank == master_rank) {
      memory_layout &mem = *(memory_layout *)window.mem;
      mem.next_ticket = 0;
      mem.now_serving = 0;
    }

    MPI_Win_lock_all(0, this->window.win);
    MPI_Barrier(this->window.comm);
  }

 public:
  static constexpr std::string_view NAME() { return "TktLock"; }

  static constexpr MPI_Aint get_win_size(bool on_master_rank) {
    return on_master_rank ? sizeof(memory_layout) : 0;
  }

  TktLock(MPI_Comm comm = MPI_COMM_WORLD, int master_rank = 0)
      : TktLock(comm, master_rank, get_rank(comm)) {}

  TktLock(MpiWindow window, int master_rank = 0)
      : TktLock(std::move(window), master_rank, get_rank(window.comm)) {}

  TktLock(TktLock &&other) = default;

  ~TktLock() {
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
    int my_ticket;
    MPI_Fetch_and_op(&ONE, &my_ticket, MPI_INT, master_rank, next_ticket_disp, MPI_SUM, window.win);

#ifdef STATS
    uint64_t loops = 0;
#endif
    // log() << "waiting for ticket: " << my_ticket << std::endl;
    int now_serving;
    do {
#ifdef STATS
      loops++;
#endif
      MPI_Fetch_and_op(
          NULL, &now_serving, MPI_INT, master_rank, now_serving_disp, MPI_NO_OP, window.win);
      MPI_Win_flush(master_rank, window.win);
#ifdef OPEN_MPI
      int flag;  // Trigger MPI progress
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
#endif
    } while (my_ticket != now_serving);
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
        &ONE, 1, MPI_INT, master_rank, now_serving_disp, 1, MPI_INT, MPI_SUM, window.win);
    MPI_Win_flush(master_rank, window.win);
    // log() << "exiting release()" << std::endl;
  }
};
