#pragma once

#include <chrono>

namespace mpi_lock_bench {
struct BenchmarkResult {
  uint64_t iterations;
  double iterations_per_process_min;
  double iterations_per_process_max;
  double iterations_per_process_median;
  double iterations_per_process_mean;
  double iterations_per_process_sd;
  double iterations_per_process_cv;
  std::chrono::nanoseconds duration;
  int mpi_processes;
  std::map<std::string, double> stats;
};
}  // namespace mpi_lock_bench