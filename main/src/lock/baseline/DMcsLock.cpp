#include <mpi.h>

#include "lock/Lock.cpp"

enum { nextRank = 0, blocked = 1, lockTail = 2 };

class DMcsLock : public Lock {
  int64_t *lmem;
  int64_t rank;
  MPI_Win win;

 public:
  static constexpr std::string_view NAME() { return "DMcsLock"; }

  /* Initialization of window for a MCS-queue */
  DMcsLock() {
    MPI_Aint winsize;
    int temp_rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &temp_rank);
    rank = (int64_t)temp_rank;

    winsize = 3 * sizeof(int64_t);
    MPI_Win_allocate(winsize, sizeof(int64_t), MPI_INFO_NULL, comm, &lmem, &win);
    lmem[nextRank] = -1;
    lmem[blocked] = 0;
    lmem[lockTail] = -1;
    MPI_Win_fence(0, win);
    MPI_Win_lock_all(0, win);
  }

  DMcsLock(DMcsLock &&other) noexcept
      :
#ifdef STATS
        acquired_immediately{other.acquired_immediately},
        acquired_delayed{other.acquired_delayed},
#endif
        lmem{other.lmem},
        rank{other.rank},
        win{other.win} {
    other.win = MPI_WIN_NULL;
  }

  /* Free window for MCS-queue */
  ~DMcsLock() {
    if (win != MPI_WIN_NULL) {
      MPI_Win_unlock_all(win);
      MPI_Win_free(&win);
    }
  }

  MPI_Comm communicator() { return MPI_COMM_WORLD; }

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
    int64_t predecessor;

    lmem[blocked] = 1;
    lmem[nextRank] = -1;

    /* Add yourself to end of queue */
    MPI_Fetch_and_op(&rank, &predecessor, MPI_INT64_T, 0, lockTail, MPI_REPLACE, win);
    MPI_Win_flush(0, win);

#ifdef PRINT
    printf(
        "%" PRId64 " predecessor: %" PRId64 ", nextRank: %" PRId64 ", blocked: %" PRId64
        ", tail: %" PRId64 "\n",
        rank, predecessor, lmem[nextRank], lmem[blocked], lmem[lockTail]);
#endif

    if (predecessor != -1) {
#ifdef STATS
      acquired_delayed++;
#endif
      /* We didn't get lock. Add us as next at predecessor. */
      MPI_Accumulate(
          &rank, 1, MPI_INT64_T, (int)predecessor, nextRank, 1, MPI_INT64_T, MPI_REPLACE, win);

      /* Now spin on our local value "blocked" until we are given the lock. */
      do {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        MPI_Win_sync(win);
      } while (lmem[blocked] == 1);
    }
#ifdef STATS
    else
      acquired_immediately++;
#endif
  }

  void release() {
    int64_t nullrank = -1, zero = 0, curtail;

#ifdef PRINT
    printf("%" PRId64 " release successor: %" PRId64 "\n", rank, lmem[nextRank]);
#endif

    if (lmem[nextRank] == -1) {
      /* See if we're waiting for the next to notify us */
      MPI_Compare_and_swap(&nullrank, &rank, &curtail, MPI_INT64_T, 0, lockTail, win);
      MPI_Win_flush(0, win);

#ifdef PRINT
      printf(
          "%" PRId64 " after flush, curtail: %" PRId64 ", tail: %" PRId64 "\n", rank, curtail,
          lmem[lockTail]);
#endif

      if (curtail == rank) {
        /* We are only process in the list */
        return;
      }
      /* Otherwise, someone else has added themselves to the list. */
      do {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        MPI_Win_sync(win);
      } while (lmem[nextRank] == -1);
    }

    /* Now we can notify them. */
    MPI_Accumulate(
        &zero, 1, MPI_INT64_T, (int)lmem[nextRank], blocked, 1, MPI_INT64_T, MPI_REPLACE, win);
  }
};
