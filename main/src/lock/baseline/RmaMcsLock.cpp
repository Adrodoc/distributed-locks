#pragma once

#include <mpi.h>

#include "lock/Lock.cpp"

#define HMCS_THRESHOLD 50

class RmaMcsLock : public Lock {
  int64_t *local_mem, *global_mem;
  int64_t local_rank, global_rank, local_tail_rank;
  MPI_Win local_win, global_win;

  enum {
    COHORT_START = 0,
    CHANGE_MODE = INT64_MAX - 1,
    ACQUIRE_PARENT = INT64_MAX - 2,
    WAIT = INT64_MAX,
    ACQUIRE_GLOBAL = INT64_MAX - 1
  };
  enum { NEXT = 0, STATUS = 1, TAIL = 2, PARENT = 3, THRESHOLD = 4, WRITER_THRESHOLD = 5 };

 public:
  static constexpr std::string_view NAME() { return "RmaMcsLock"; }

  /* Initialization of window for a MCS-queue */
  RmaMcsLock() {
    MPI_Aint winsize;
    int temp_rank, local_size;
    MPI_Comm local_comm;

    MPI_Comm global_comm = MPI_COMM_WORLD;
    MPI_Comm_split_type(global_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);

    MPI_Comm_rank(global_comm, &temp_rank);
    global_rank = (int64_t)temp_rank;
    MPI_Comm_rank(local_comm, &temp_rank);
    local_rank = (int64_t)temp_rank;

    MPI_Comm_size(local_comm, &local_size);
    local_tail_rank = (global_rank / local_size) * local_size;

#ifdef PRINT
    printf(
        "%" PRId64 " HMCS init, local_rank: %" PRId64 ", local_tail_rank: %" PRId64 "\n",
        global_rank, local_rank, local_tail_rank);
#endif

    winsize = 3 * sizeof(int64_t);
    MPI_Win_allocate(winsize, sizeof(int64_t), MPI_INFO_NULL, local_comm, &local_mem, &local_win);
    MPI_Win_allocate(
        winsize, sizeof(int64_t), MPI_INFO_NULL, global_comm, &global_mem, &global_win);
    local_mem[NEXT] = -1;
    local_mem[STATUS] = 0;
    local_mem[TAIL] = -1;
    global_mem[NEXT] = -1;
    global_mem[STATUS] = 0;
    global_mem[TAIL] = -1;
    MPI_Win_fence(0, local_win);
    MPI_Win_fence(0, global_win);
    MPI_Win_lock_all(0, local_win);
    MPI_Win_lock_all(0, global_win);
#ifdef PRINT
    printf("%" PRId64 " HMCS initialized\n", global_rank);
#endif
  }

  RmaMcsLock(RmaMcsLock &&other) noexcept
      :
#ifdef STATS
        acquired_immediately{other.acquired_immediately},
        acquired_delayed{other.acquired_delayed},
#endif
        local_mem{other.local_mem},
        global_mem{other.global_mem},
        local_rank{other.local_rank},
        global_rank{other.global_rank},
        local_tail_rank{other.local_tail_rank},
        local_win{other.local_win},
        global_win{other.global_win} {
    other.local_win = MPI_WIN_NULL;
    other.global_win = MPI_WIN_NULL;
  }

  /* Free window for MCS-queue */
  ~RmaMcsLock() {
#ifdef PRINT
    printf("%" PRId64 " HMCS Finalization\n", global_rank);
#endif
    if (local_win != MPI_WIN_NULL) MPI_Win_unlock_all(local_win);
    if (global_win != MPI_WIN_NULL) MPI_Win_unlock_all(global_win);
    if (local_win != MPI_WIN_NULL) MPI_Win_free(&local_win);
    if (global_win != MPI_WIN_NULL) MPI_Win_free(&global_win);
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
#ifdef STATS
    bool immediate = true;
#endif
    int64_t predecessor, status;
    int64_t null = -1, wait = WAIT;

    local_mem[STATUS] = wait;
    local_mem[NEXT] = null;

    /* Add yourself to end of local queue */
    MPI_Fetch_and_op(&local_rank, &predecessor, MPI_INT64_T, 0, TAIL, MPI_REPLACE, local_win);
    MPI_Win_flush(0, local_win);

#ifdef PRINT
    printf("%" PRId64 " LOCAL predecessor: %" PRId64 "\n", global_rank, predecessor);
#endif

    if (predecessor != -1) {
#ifdef STATS
      immediate = false;
#endif
      /* We didn't get lock. Add us as next at predecessor. */
      MPI_Accumulate(
          &local_rank, 1, MPI_INT64_T, predecessor, NEXT, 1, MPI_INT64_T, MPI_REPLACE, local_win);
      MPI_Win_flush(predecessor, local_win);

#ifdef PRINT
      printf("%" PRId64 " LOCAL sync predecessor: %" PRId64 "\n", global_rank, predecessor);
#endif

      /* Now spin on our local value "status" until we are given the lock. */
      do {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        MPI_Win_sync(local_win);
      } while ((status = local_mem[STATUS]) == WAIT);
      if (status != ACQUIRE_GLOBAL) {
#ifdef PRINT
        printf("%" PRId64 " LOCAL lock acquired\n", global_rank);
#endif
#ifdef STATS
        acquired_delayed++;
#endif
        return;
      }
    }
    MPI_Accumulate(
        &wait, 1, MPI_INT64_T, local_tail_rank, STATUS, 1, MPI_INT64_T, MPI_REPLACE, global_win);
    MPI_Accumulate(
        &null, 1, MPI_INT64_T, local_tail_rank, NEXT, 1, MPI_INT64_T, MPI_REPLACE, global_win);
    MPI_Win_flush(local_tail_rank, global_win);
    /* Add yourself to end of global queue */
    MPI_Fetch_and_op(&local_tail_rank, &predecessor, MPI_INT64_T, 0, TAIL, MPI_REPLACE, global_win);
    MPI_Win_flush(0, global_win);

#ifdef PRINT
    printf("%" PRId64 " GLOBAL predecessor: %" PRId64 "\n", global_rank, predecessor);
#endif

    if (predecessor != -1) {
#ifdef STATS
      immediate = false;
#endif
      /* We didn't get lock. Add us as next at predecessor. */
      MPI_Accumulate(
          &local_tail_rank, 1, MPI_INT64_T, predecessor, NEXT, 1, MPI_INT64_T, MPI_REPLACE,
          global_win);

#ifdef PRINT
      printf("%" PRId64 " GLOBAL sync, predecessor: %" PRId64 "\n", global_rank, predecessor);
#endif

      /* Now spin on our local value "status" until we are given the lock. */
      do {
        MPI_Get_accumulate(
            0, 0, MPI_INT64_T, &status, 1, MPI_INT64_T, local_tail_rank, STATUS, 1, MPI_INT64_T,
            MPI_NO_OP, global_win);
        MPI_Win_flush(local_tail_rank, global_win);
      } while (status == WAIT);
    }
    local_mem[STATUS] = COHORT_START;
#ifdef PRINT
    printf("%" PRId64 " GLOBAL lock acquired\n", global_rank);
#endif
#ifdef STATS
    if (immediate)
      acquired_immediately++;
    else
      acquired_delayed++;
#endif
  }

  void release() {
    int64_t nullrank = -1, zero = 0, curtail = -1;
    int64_t local_successor, global_successor, current_status, acquire_global = ACQUIRE_GLOBAL;

    MPI_Win_sync(local_win);
#ifdef PRINT
    printf(
        "%" PRId64 " LOCAL release successor: %" PRId64 ", status: %" PRId64 "\n", global_rank,
        local_mem[NEXT], local_mem[STATUS]);
#endif
    local_successor = local_mem[NEXT];

    if (local_successor != -1 && local_mem[STATUS] < HMCS_THRESHOLD) {
      /* Notify the successor */
      current_status = local_mem[STATUS] + 1;
      MPI_Accumulate(
          &current_status, 1, MPI_INT64_T, local_successor, STATUS, 1, MPI_INT64_T, MPI_REPLACE,
          local_win);
      MPI_Win_flush(local_successor, local_win);
#ifdef PRINT
      printf(
          "%" PRId64 " LOCAL released to successor: %" PRId64 ", status: %" PRIx64 "\n",
          global_rank, local_successor, current_status);
#endif
      return;
    }
    MPI_Get_accumulate(
        0, 0, MPI_INT64_T, &global_successor, 1, MPI_INT64_T, local_tail_rank, NEXT, 1, MPI_INT64_T,
        MPI_NO_OP, global_win);
    MPI_Win_flush(local_tail_rank, global_win);
#ifdef PRINT
    printf("%" PRId64 " GLOBAL release successor: %" PRId64 "\n", global_rank, global_successor);
#endif
    if (global_successor == -1) {
      /* See if we're waiting for the next to notify us */
      MPI_Compare_and_swap(&nullrank, &local_tail_rank, &curtail, MPI_INT64_T, 0, TAIL, global_win);
      MPI_Win_flush(0, global_win);
#ifdef PRINT
      printf("%" PRId64 " GLOBAL CAS curtail: %" PRId64 "\n", global_rank, curtail);
#endif
      if (curtail != local_tail_rank) {
#ifdef PRINT
        printf("%" PRId64 " GLOBAL sync release\n", global_rank);
#endif
        do {
          MPI_Get_accumulate(
              0, 0, MPI_INT64_T, &global_successor, 1, MPI_INT64_T, local_tail_rank, NEXT, 1,
              MPI_INT64_T, MPI_NO_OP, global_win);
          MPI_Win_flush(local_tail_rank, global_win);
        } while (global_successor == -1);
      }
    }
    if (global_successor != -1 && curtail != local_tail_rank) {
      MPI_Accumulate(
          &zero, 1, MPI_INT64_T, global_successor, STATUS, 1, MPI_INT64_T, MPI_REPLACE, global_win);
      MPI_Win_flush(global_successor, global_win);
#ifdef PRINT
      printf(
          "%" PRId64 " GLOBAL released to successor: %" PRId64 "\n", global_rank, global_successor);
#endif
    }
#ifdef PRINT
    if (global_successor == -1) {
      printf("%" PRId64 " GLOBAL released, no successor\n", global_rank);
    }
#endif

    if (local_successor == -1) {
      /* See if we're waiting for the next to notify us */
      MPI_Compare_and_swap(&nullrank, &local_rank, &curtail, MPI_INT64_T, 0, TAIL, local_win);
      MPI_Win_flush(0, local_win);
#ifdef PRINT
      printf("%" PRId64 " LOCAL CAS curtail: %" PRId64 "\n", global_rank, curtail);
#endif
      if (curtail == local_rank) {
#ifdef PRINT
        printf("%" PRId64 " LOCAL released, no successor \n", global_rank);
#endif
        return;
      }
/* Someone else has added themselves to the list. */
#ifdef PRINT
      printf("%" PRId64 " LOCAL sync release\n", global_rank);
#endif
      do {
        int flag;  // Trigger MPI progress
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        MPI_Win_sync(local_win);
      } while ((local_successor = local_mem[NEXT]) == -1);
    }

#ifdef PRINT
    printf(
        "%" PRId64 " LOCAL notify successor acquire global: %" PRId64 "\n", global_rank,
        local_successor);
#endif
    MPI_Accumulate(
        &acquire_global, 1, MPI_INT64_T, local_successor, STATUS, 1, MPI_INT64_T, MPI_REPLACE,
        local_win);
    MPI_Win_flush(local_successor, local_win);
  }
};
