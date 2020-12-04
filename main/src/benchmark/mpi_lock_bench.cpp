#pragma once

#include <mpi.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <map>

#include "BenchmarkResult.cpp"
#include "Config.cpp"
#include "Reporter.cpp"
#include "lock/Lock.cpp"
#include "log.cpp"
#include "statistics.cpp"

namespace mpi_lock_bench {
using std::chrono::nanoseconds;
using clock = std::chrono::high_resolution_clock;

class Context {
  const clock::time_point time_quota_end;

 public:
  Context(clock::time_point time_quota_end) : time_quota_end{time_quota_end} {}

  bool time_quota_expired() { return clock::now() >= time_quota_end; }
};

class Benchmark {
 public:
  virtual std::map<std::string, std::string> reporter_context() { return {}; }
  virtual void before() {}
  virtual void run(Lock &lock, Context &context) = 0;
  virtual void after() {}
};

class FunctionBenchmark : public Benchmark {
  const std::function<void(Lock &)> &function;

 public:
  FunctionBenchmark(const std::function<void(Lock &)> &function) : function{function} {}

  void run(Lock &lock, Context &context) { function(lock); }
};

static Config static_config;

void initialize(int argc, char **argv) { static_config = parse_arguments(argc, argv); }

std::string to_file_suffix(std::map<std::string, std::string> context) {
  std::stringstream stream;
  for (const auto &[key, value] : context) {
    stream << '-' << key << '=' << value;
  }
  return stream.str();
}

BenchmarkResult run_single_repetition(
    Benchmark &benchmark, Lock &lock, const Config &config = static_config) {
  nanoseconds min_duration = config.min_duration;
  double warm_up_ratio = config.warm_up_ratio;

  auto warm_up_duration = std::chrono::duration_cast<nanoseconds>(min_duration * warm_up_ratio);
  auto benchmark_duration =
      std::chrono::duration_cast<nanoseconds>(min_duration * (1 - warm_up_ratio));

  MPI_Comm comm = lock.communicator();

  benchmark.before();

  // warm up
  MPI_Barrier(comm);
  auto warm_up_start = clock::now();
  auto warm_up_end = warm_up_start + warm_up_duration;
  Context warm_up_context{warm_up_end};
  while (!warm_up_context.time_quota_expired()) {
    benchmark.run(lock, warm_up_context);
  }
  warm_up_end = clock::now();
  warm_up_duration = warm_up_end - warm_up_start;  // Adjust duration for actual end

#ifdef STATS
  lock.stats();  // Clear stats
#endif

  // benchmark
  MPI_Barrier(comm);
  auto benchmark_start = clock::now();
  auto benchmark_end = benchmark_start + benchmark_duration;
  Context benchmark_context{benchmark_end};
  uint64_t benchmark_iterations = 0;
  for (; !benchmark_context.time_quota_expired(); benchmark_iterations++) {
    benchmark.run(lock, benchmark_context);
  }
  benchmark_end = clock::now();
  benchmark_duration = benchmark_end - benchmark_start;  // Adjust duration for actual end

  benchmark.after();

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::map<std::string, double> stats_sum;
#ifdef STATS
  auto stats = lock.stats();
  for (const auto &[key, value] : stats) {
    double sum;
    MPI_Reduce(&value, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (rank == 0) stats_sum.insert({key, sum});
  }
#endif

  int64_t nanos = benchmark_duration.count();
  int64_t max_nanos;
  MPI_Reduce(&nanos, &max_nanos, 1, MPI_INT64_T, MPI_MAX, 0, comm);

  uint64_t total_iterations;
  MPI_Reduce(&benchmark_iterations, &total_iterations, 1, MPI_UINT64_T, MPI_SUM, 0, comm);

  int mpi_processes;
  MPI_Comm_size(comm, &mpi_processes);

  double iterations = benchmark_iterations;
  std::vector<double> iterations_per_process(mpi_processes);
  MPI_Gather(&iterations, 1, MPI_DOUBLE, iterations_per_process.data(), 1, MPI_DOUBLE, 0, comm);

  if (rank == 0) {
    nanoseconds max_duration{max_nanos};
    return BenchmarkResult{
        .iterations = total_iterations,
        .iterations_per_process_min = statistics::min(iterations_per_process),
        .iterations_per_process_max = statistics::max(iterations_per_process),
        .iterations_per_process_median = statistics::median(iterations_per_process),
        .iterations_per_process_mean = statistics::mean(iterations_per_process),
        .iterations_per_process_sd = statistics::standard_deviation(iterations_per_process),
        .iterations_per_process_cv = statistics::coefficient_of_variation(iterations_per_process),
        .duration = max_duration,
        .mpi_processes = mpi_processes,
        .stats = stats_sum,
    };
  } else
    return BenchmarkResult{};
}

template <class L>
L create() {
  return L();
}

#define RUN_MPI_LOCK_BENCHMARK_CLASS(B, L, ...) \
  mpi_lock_bench::run_benchmark_class<B, L>(B(), #B, ##__VA_ARGS__);

template <class B, class L>
void run_benchmark_class(
    B benchmark, std::string_view benchmark_name = B::NAME(),
    const std::function<L()> &lock_supplier = create<L>, std::string_view lock_name = L::NAME(),
    const Config &config = static_config) {
  run_benchmark(benchmark, benchmark_name, lock_supplier, lock_name, config);
}

#define RUN_MPI_LOCK_BENCHMARK(b, L, ...) mpi_lock_bench::run_benchmark<L>(b, #b, ##__VA_ARGS__);

template <class L>
void run_benchmark(
    const std::function<void(Lock &)> &benchmark, std::string_view benchmark_name,
    const std::function<L()> &lock_supplier = create<L>, std::string_view lock_name = L::NAME(),
    const Config &config = static_config) {
  FunctionBenchmark b{benchmark};
  run_benchmark(b, benchmark_name, lock_supplier, lock_name, config);
}

template <class L>
void run_benchmark(
    Benchmark &benchmark, std::string_view benchmark_name,
    const std::function<L()> &lock_supplier = create<L>, std::string_view lock_name = L::NAME(),
    const Config &config = static_config) {
  if (!config.should_run_benchmark(benchmark_name, lock_name)) return;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto context = benchmark.reporter_context();
  context.insert({"processes", std::to_string(size)});
  auto benchmark_suffix = to_file_suffix(context);
  std::string output_file = (config.output_dir + "/")
                                .append(benchmark_name)
                                .append("-")
                                .append(lock_name)
                                .append(benchmark_suffix + config.file_suffix + ".json");
  Reporter reporter{std::move(output_file)};
  if (rank == 0) {
    context.insert({
        {"benchmark", std::string{benchmark_name}},
        {"lock", std::string{lock_name}},
    });
    reporter.report_context(context);
  }

  for (uint64_t repetition = 0; repetition < config.repetitions; repetition++) {
    auto lock = lock_supplier();
    auto result = run_single_repetition(benchmark, lock, config);
    if (rank == 0) reporter.report_benchmark_run(benchmark_name, lock_name, result);
  }
}

template <class L>
nanoseconds acquire_and_release_all_locks(std::vector<L> &locks) {
  auto start = clock::now();
  for (auto &&lock : locks) {
    lock.acquire();
    lock.release();
  }
  auto end = clock::now();
  return end - start;
}

static bool print_uncontested_ranks{true};

template <class L>
void benchmark_uncontested_performance(
    const std::function<L()> &lock_supplier = create<L>, std::string_view lock_name = L::NAME(),
    const Config &config = static_config) {
  constexpr std::string_view benchmark_name = "UPB";
  if (!config.should_run_benchmark(benchmark_name, lock_name)) return;

  const MPI_Comm comm = MPI_COMM_WORLD;
  int rank;
  MPI_Comm_rank(comm, &rank);

  MPI_Comm local_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(comm, &local_rank);

  int min_rank_on_node;
  MPI_Allreduce(&rank, &min_rank_on_node, 1, MPI_INT, MPI_MIN, local_comm);
  bool is_master_node = min_rank_on_node == 0;

  MPI_Comm_free(&local_comm);

  // The minimum rank that is on the same node as, but is not itself rank 0.
  int rank_on_master;
  const int rank_only_same_node =
      is_master_node && rank != 0 ? rank : std::numeric_limits<int>::max();
  MPI_Allreduce(&rank_only_same_node, &rank_on_master, 1, MPI_INT, MPI_MIN, comm);

  // The minimum rank that is on a different node than rank 0.
  int min_rank_on_slave;
  const int rank_only_slave_nodes = !is_master_node ? rank : std::numeric_limits<int>::max();
  MPI_Allreduce(&rank_only_slave_nodes, &min_rank_on_slave, 1, MPI_INT, MPI_MIN, comm);

  // The maximum rank that is on the same node as min_rank_on_slave
  int max_rank_on_slave;
  const int rank_only_slave_node =
      min_rank_on_node == min_rank_on_slave ? rank : std::numeric_limits<int>::min();
  MPI_Allreduce(&rank_only_slave_node, &max_rank_on_slave, 1, MPI_INT, MPI_MAX, comm);

  if (print_uncontested_ranks && rank == 0) {
    print_uncontested_ranks = false;
    std::cout << "rank_on_master=" << rank_on_master << "\tmin_rank_on_slave=" << min_rank_on_slave
              << "\tmax_rank_on_slave=" << max_rank_on_slave << std::endl;
  }

  std::map<std::string, std::string> context = {{"lock_count", std::to_string(config.upb_locks)}};
  auto benchmark_suffix = to_file_suffix(context);
  std::string output_file = (config.output_dir + "/")
                                .append(benchmark_name)
                                .append("-")
                                .append(lock_name)
                                .append(benchmark_suffix + config.file_suffix + ".json");
  Reporter reporter{std::move(output_file)};
  if (rank == 0) {
    context.insert({
        {"benchmark", std::string{benchmark_name}},
        {"lock", std::string{lock_name}},
    });
    reporter.report_context(context);
  }

  std::vector<L> locks;
  for (size_t i = 0; i < config.upb_locks; i++) {
    locks.push_back(std::move(lock_supplier()));
  }

  for (uint64_t repetition = 0; repetition < config.repetitions; repetition++) {
    // if (rank == 0)
    // log() << "UPB: warm up" << std::endl;
    acquire_and_release_all_locks(locks);
    MPI_Barrier(comm);

    // make max_rank_on_slave the previous owner
    if (rank == max_rank_on_slave) {
      // log() << "UPB: make " << max_rank_on_slave << " the previous owner" << std::endl;
      acquire_and_release_all_locks(locks);
    }
    MPI_Barrier(comm);

    // scenario 3a: previous owner is on a different node and I am the master process.
    nanoseconds different_node_master_rank;
    if (rank == 0) {
      // log() << "UPB: scenario 3a" << std::endl;
      different_node_master_rank = acquire_and_release_all_locks(locks);
    }
    MPI_Barrier(comm);

    // scenario 1a: previous owner is same process and I am the master process.
    nanoseconds same_process_master_rank;
    if (rank == 0) {
      // log() << "UPB: scenario 1a" << std::endl;
      same_process_master_rank = acquire_and_release_all_locks(locks);
    }
    MPI_Barrier(comm);

    // scenario 2b: previous owner is on the same node and I am on the master node.
    nanoseconds same_node_master_node;
    if (rank == rank_on_master) {
      // log() << "UPB: scenario 2b" << std::endl;
      same_node_master_node = acquire_and_release_all_locks(locks);
    }
    MPI_Barrier(comm);

    // scenario 1b: previous owner is same process and I am on the master node.
    nanoseconds same_process_master_node;
    if (rank == rank_on_master) {
      // log() << "UPB: scenario 1b" << std::endl;
      same_process_master_node = acquire_and_release_all_locks(locks);
    }
    MPI_Barrier(comm);

    // scenario 2a: previous owner is on the same node and I am the master process.
    nanoseconds same_node_master_rank;
    if (rank == 0) {
      // log() << "UPB: scenario 2a" << std::endl;
      same_node_master_rank = acquire_and_release_all_locks(locks);
    }
    MPI_Barrier(comm);

    // scenario 3c: previous owner is on a different node and I am on a slave node.
    nanoseconds different_node_slave_node;
    if (rank == min_rank_on_slave) {
      // log() << "UPB: scenario 3c" << std::endl;
      different_node_slave_node = acquire_and_release_all_locks(locks);
    }
    MPI_Barrier(comm);

    // scenario 1c: previous owner is same process and I am on a slave node.
    nanoseconds same_process_slave_node;
    if (rank == min_rank_on_slave) {
      // log() << "UPB: scenario 1c" << std::endl;
      same_process_slave_node = acquire_and_release_all_locks(locks);
    }
    MPI_Barrier(comm);

    // scenario 2c: previous owner is on the same node and I am on a slave node.
    nanoseconds same_node_slave_node;
    if (rank == max_rank_on_slave) {
      // log() << "UPB: scenario 2c" << std::endl;
      same_node_slave_node = acquire_and_release_all_locks(locks);
    }
    MPI_Barrier(comm);

    // scenario 3b: previous owner is on a different node and I am on the master node.
    nanoseconds different_node_master_node;
    if (rank == rank_on_master) {
      // log() << "UPB: scenario 3b" << std::endl;
      different_node_master_node = acquire_and_release_all_locks(locks);
    }
    MPI_Barrier(comm);

    if (rank == rank_on_master) {
      int64_t same_process_master_node_ns = same_process_master_node.count();
      MPI_Send(&same_process_master_node_ns, 1, MPI_INT64_T, 0, 0, comm);
      int64_t same_node_master_node_ns = same_node_master_node.count();
      MPI_Send(&same_node_master_node_ns, 1, MPI_INT64_T, 0, 0, comm);
      int64_t different_node_master_node_ns = different_node_master_node.count();
      MPI_Send(&different_node_master_node_ns, 1, MPI_INT64_T, 0, 0, comm);
    }
    if (rank == min_rank_on_slave) {
      int64_t same_process_slave_node_ns = same_process_slave_node.count();
      MPI_Send(&same_process_slave_node_ns, 1, MPI_INT64_T, 0, 0, comm);
      int64_t different_node_slave_node_ns = different_node_slave_node.count();
      MPI_Send(&different_node_slave_node_ns, 1, MPI_INT64_T, 0, 0, comm);
    }
    if (rank == max_rank_on_slave) {
      int64_t same_node_slave_node_ns = same_node_slave_node.count();
      MPI_Send(&same_node_slave_node_ns, 1, MPI_INT64_T, 0, 0, comm);
    }
    if (rank == 0) {
      int64_t same_process_master_node_ns = 0;
      int64_t same_node_master_node_ns = 0;
      int64_t different_node_master_node_ns = 0;
      if (rank_on_master != std::numeric_limits<int>::max()) {
        MPI_Recv(
            &same_process_master_node_ns, 1, MPI_INT64_T, rank_on_master, 0, comm,
            MPI_STATUS_IGNORE);
        MPI_Recv(
            &same_node_master_node_ns, 1, MPI_INT64_T, rank_on_master, 0, comm, MPI_STATUS_IGNORE);
        MPI_Recv(
            &different_node_master_node_ns, 1, MPI_INT64_T, rank_on_master, 0, comm,
            MPI_STATUS_IGNORE);
      }

      int64_t same_process_slave_node_ns = 0;
      int64_t different_node_slave_node_ns = 0;
      if (min_rank_on_slave != std::numeric_limits<int>::max()) {
        MPI_Recv(
            &same_process_slave_node_ns, 1, MPI_INT64_T, min_rank_on_slave, 0, comm,
            MPI_STATUS_IGNORE);
        MPI_Recv(
            &different_node_slave_node_ns, 1, MPI_INT64_T, min_rank_on_slave, 0, comm,
            MPI_STATUS_IGNORE);
      }

      int64_t same_node_slave_node_ns = 0;
      if (max_rank_on_slave != std::numeric_limits<int>::min()) {
        MPI_Recv(
            &same_node_slave_node_ns, 1, MPI_INT64_T, max_rank_on_slave, 0, comm,
            MPI_STATUS_IGNORE);
      }

      reporter.report_uncontested(
          benchmark_name, lock_name, config.upb_locks, same_process_master_rank.count(),
          same_node_master_rank.count(), different_node_master_rank.count(),
          same_process_master_node_ns, same_node_master_node_ns, different_node_master_node_ns,
          same_process_slave_node_ns, same_node_slave_node_ns, different_node_slave_node_ns);
    }
  }
}
}  // namespace mpi_lock_bench