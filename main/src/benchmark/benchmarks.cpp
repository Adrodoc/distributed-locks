#pragma once

#include <algorithm>
#include <cstring>
#include <random>

#include "mpi_lock_bench.cpp"
#include "mpi_utils/mpi_utils.cpp"

using std::chrono::nanoseconds;
using namespace mpi_lock_bench;

void spin(nanoseconds duration) {
  using clock = std::chrono::high_resolution_clock;
  auto spin_until = clock::now() + duration;
  while (clock::now() < spin_until)
    ;
}

MPI_Win mpi_allocate(MPI_Aint size, int rank = -1, MPI_Comm comm = MPI_COMM_WORLD) {
  if (rank != -1) {
    int my_rank;
    MPI_Comm_rank(comm, &my_rank);
    if (my_rank != rank) size = 0;
  }

  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "accumulate_ordering", "none");
  MPI_Info_set(info, "same_disp_unit", "true");
  void *mem;
  MPI_Win win;
  MPI_Win_allocate(size, 1, info, comm, &mem, &win);

  if (size != 0) memset(mem, 0, size);
  MPI_Barrier(comm);
  return win;
}

/*
 * The empty-critical-section benchmark (ECSB) derives the throughput of acquiring an empty lock
 * with no workload in the CS.
 */
void ECSB(Lock &lock) {
  lock.acquire();
  lock.release();
}

/*
 * The single-operation benchmark (SOB) measures the throughput of acquiring a lock with only one
 * single operation (one memory access) in the CS; it represents irregular parallel work-loads such
 * as graph processing with vertices protected by fine locks.
 */
class SOB : public Benchmark {
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<> dist{1000, 4000};
  MPI_Win win;
  // A rank that is most likely not on the same node.
  const int remote_rank = (get_rank() + get_size() / 2) % get_size();

 public:
  void before() {
    win = mpi_allocate(sizeof(int));
    MPI_Win_lock_all(0, win);
  }

  void run(Lock &lock, Context &context) {
    lock.acquire();
    int value = dist(gen);
    MPI_Put(&value, 1, MPI_INT, remote_rank, 0, 1, MPI_INT, win);
    MPI_Win_flush(0, win);
    lock.release();
  }

  void after() {
    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
  }
};

/*
 * The workload-critical-section benchmark (WCSB) covers variable workloads in the CS: each process
 * increments a shared counter and then spins for a random time (1-4μs) to simulate local
 * computation.
 */
class WCSB : public Benchmark {
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<> dist{1000, 4000};
  MPI_Win win;

 public:
  void before() {
    win = mpi_allocate(sizeof(int), 0);
    MPI_Win_lock_all(0, win);
  }

  void run(Lock &lock, Context &context) {
    lock.acquire();
    int value;
    MPI_Get(&value, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
    MPI_Win_flush(0, win);
    value += 1;
    MPI_Put(&value, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
    MPI_Win_flush(0, win);

    spin(nanoseconds{dist(gen)});
    lock.release();
  }

  void after() {
    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
  }
};

/*
 * The wait-before-acquire benchmark (WBAB) varies lock contention: before each acquire, processes
 * wait for a random time (0-5μs per process).
 */
class WBAB : public Benchmark {
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<> dist;
  const int avg_wait_ns;
  const bool mpi_progress;

  WBAB(int avg_wait_ns, bool mpi_progress, int size)
      : WBAB(avg_wait_ns, mpi_progress, (avg_wait_ns * size) / 3, size) {}

  WBAB(int avg_wait_ns, bool mpi_progress, int delta_wait_ns, int size)
      : avg_wait_ns{avg_wait_ns},
        dist{avg_wait_ns * size - delta_wait_ns, avg_wait_ns * size + delta_wait_ns},
        mpi_progress{mpi_progress} {}

 public:
  WBAB(int avg_wait_ns, bool mpi_progress) : WBAB(avg_wait_ns, mpi_progress, get_size()) {}

  std::map<std::string, std::string> reporter_context() {
    return {
        {"avg_wait_ns", std::to_string(avg_wait_ns)},
        {"mpi_progress", std::to_string(mpi_progress)},
    };
  }

  void run(Lock &lock, Context &context) {
    if (mpi_progress)
      spin_with_mpi_progress(nanoseconds{dist(gen)});
    else
      spin(nanoseconds{dist(gen)});
    lock.acquire();
    lock.release();
  }
};

template <class L, typename... _Args>
void run_wbab(bool mpi_progress, _Args &&...__args) {
  WBAB benchmark{0, mpi_progress};
  run_benchmark<L>(benchmark, "WBAB", std::forward<_Args>(__args)...);

  for (int i = 1; i <= 256; i *= 2) {
    int avg_wait_ns = i * 250;
    WBAB benchmark{avg_wait_ns, mpi_progress};
    run_benchmark<L>(benchmark, "WBAB", std::forward<_Args>(__args)...);
  }
}

/*
 * The wait-for-new-owner benchmark (WFNOB) prevents the same thread from acquiring the lock over
 * and over again: after release, processes wait until another process has entered the critical
 * section.
 */
class WFNOB : public Benchmark {
  const int master_rank{0};
  const int rank{get_rank()};
  MPI_Win win;

 public:
  void before() {
    win = mpi_allocate(sizeof(int), master_rank);
    MPI_Win_lock_all(0, win);
  }

  void run(Lock &lock, Context &context) {
    lock.acquire();
    MPI_Accumulate(&rank, 1, MPI_INT, master_rank, 0, 1, MPI_INT, MPI_REPLACE, win);
    MPI_Win_flush(master_rank, win);
    lock.release();

    int last_owner;
    do {
      MPI_Fetch_and_op(NULL, &last_owner, MPI_INT, master_rank, 0, MPI_NO_OP, win);
      MPI_Win_flush(master_rank, win);
    } while (last_owner == rank && !context.time_quota_expired());
  }

  void after() {
    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
  }
};

/*
 * The changing-critical-work benchmark (CCWB) measures performance under different contention
 * levels: The amount of work in the critical section is varied, changing the ratio of critical and
 * noncritical work.
 */
class CCWB : public Benchmark {
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<> work_dist;
  const int rank{get_rank()};
  const int size{get_size()};
  MPI_Win win;
  const int critical_work;
  const int work;

 public:
  CCWB(int critical_work, int work)
      : critical_work{critical_work}, work{work}, work_dist{work, work * 2} {}

  std::map<std::string, std::string> reporter_context() {
    return {
        {"critical_work", std::to_string(critical_work)},
        {"work", std::to_string(work)},
    };
  }

  void before() {
    win = mpi_allocate(sizeof(int) * (work * 2));
    MPI_Win_lock_all(0, win);
  }

  void run(Lock &lock, Context &context) {
    int actual_work = work_dist(gen);
    int noncritical_work = actual_work - critical_work;
    for (int i = 0; i < noncritical_work; i++) {
      MPI_Aint disp = sizeof(int) * (critical_work + i);
      increment_non_atomically(disp);
    }

    lock.acquire();
    for (int i = 0; i < critical_work; i++) {
      MPI_Aint disp = sizeof(int) * i;
      increment_non_atomically(disp);
    }
    lock.release();
  }

  void increment_non_atomically(MPI_Aint disp) {
    // A rank that is most likely not on the same node.
    int remote_rank = (rank + size / 2) % size;
    int value;
    MPI_Get(&value, 1, MPI_INT, remote_rank, disp, 1, MPI_INT, win);
    MPI_Win_flush(remote_rank, win);
    value += 1;
    MPI_Put(&value, 1, MPI_INT, remote_rank, disp, 1, MPI_INT, win);
    MPI_Win_flush(remote_rank, win);
  }

  void after() {
    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
  }
};

template <class L, typename... _Args>
void run_ccwb(_Args &&...__args) {
  int size = get_size();
  int iterations = 6;
  // int equilibrium = iterations / 2;
  // int avg_work_factor = 1.5;
  // int work = size * equilibrium / avg_work_factor;
  int work = size * iterations / 3;
  for (int i = 0; i < iterations; i++) {
    int critical_work = i;
    CCWB benchmark{critical_work, work};
    run_benchmark<L>(benchmark, "CCWB", std::forward<_Args>(__args)...);
  }
}

/*
 * The wait benchmark (WB) is a sanity check: processes wait for a random time (1-4μs) without
 * involving a lock.
 */
class WB : public Benchmark {
  std::random_device rd;
  std::mt19937 gen{rd()};
  std::uniform_int_distribution<> dist{1000, 4000};

 public:
  void run(Lock &lock) { spin(nanoseconds{dist(gen)}); }
};

template <class L, typename... _Args>
void run_mpi_lock_benchmarks(_Args &&...__args) {
  benchmark_uncontested_performance<L>(std::forward<_Args>(__args)...);
  RUN_MPI_LOCK_BENCHMARK(ECSB, L, std::forward<_Args>(__args)...);
  // run_wbab<L>(false, std::forward<_Args>(__args)...);
  run_wbab<L>(true, std::forward<_Args>(__args)...);
  run_ccwb<L>(std::forward<_Args>(__args)...);
  RUN_MPI_LOCK_BENCHMARK_CLASS(SOB, L, std::forward<_Args>(__args)...);
  RUN_MPI_LOCK_BENCHMARK_CLASS(WCSB, L, std::forward<_Args>(__args)...);
  RUN_MPI_LOCK_BENCHMARK_CLASS(WFNOB, L, std::forward<_Args>(__args)...);
}
