#pragma once

#include <regex>

namespace mpi_lock_bench {
struct Config {
  std::string output_dir;
  std::chrono::nanoseconds min_duration = std::chrono::seconds{1};
  double warm_up_ratio = 0.1;
  uint64_t repetitions = 1;
  std::string file_suffix = "";
  // 2048 is the upper limit for communicators and therefor windows in MPICH.
  // Some locks use 2 windows
  size_t upb_locks = 1000;
  std::regex include_benchmarks{".+"};
  std::regex exclude_benchmarks;
  std::regex include_locks{".+"};
  std::regex exclude_locks;

  bool should_run_benchmark(std::string_view benchmark_name, std::string_view lock_name) const {
    return std::regex_match(benchmark_name.begin(), benchmark_name.end(), include_benchmarks) &&
           !std::regex_match(benchmark_name.begin(), benchmark_name.end(), exclude_benchmarks) &&
           std::regex_match(lock_name.begin(), lock_name.end(), include_locks) &&
           !std::regex_match(lock_name.begin(), lock_name.end(), exclude_locks);
  }
};

void parse_argument(std::string_view argument, Config &config) {
  const std::string_view PREFIX = "--";
  if (argument.rfind(PREFIX, 0) == 0)  // starts with
  {
    std::string_view suffix = argument.substr(PREFIX.length());
    auto eq = suffix.find('=');
    if (eq != std::string::npos) {
      std::string_view key = suffix.substr(0, eq);
      std::string value{suffix.substr(eq + 1)};
      if (key == "out") config.output_dir = value;
      if (key == "secs") config.min_duration = std::chrono::seconds{std::stoi(value)};
      if (key == "warm_up_ratio") config.warm_up_ratio = std::stod(value);
      if (key == "repetitions") config.repetitions = std::stoul(value);
      if (key == "file_suffix") config.file_suffix = value;
      if (key == "upb_locks") config.upb_locks = std::stoul(value);
      if (key == "include_benchmarks") config.include_benchmarks = value;
      if (key == "exclude_benchmarks") config.exclude_benchmarks = value;
      if (key == "include_locks") config.include_locks = value;
      if (key == "exclude_locks") config.exclude_locks = value;
    }
  }
}

Config parse_arguments(int argc, char **argv) {
  Config config;
  for (int i = 1; i < argc; i++) parse_argument(std::string_view(argv[i]), config);
  return config;
}
}  // namespace mpi_lock_bench