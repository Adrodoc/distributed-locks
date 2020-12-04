#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

#include "BenchmarkResult.cpp"

namespace mpi_lock_bench {
using std::chrono::nanoseconds;

class Reporter {
  static constexpr std::string_view INDENT = "  ";

  static std::string json_escape(char c) {
    switch (c) {
      case '\b': return "\\b";
      case '\f': return "\\f";
      case '\n': return "\\n";
      case '\r': return "\\r";
      case '\t': return "\\t";
      case '\\': return "\\\\";
      case '"': return "\\\"";
      default: return std::string(1, c);
    }
  }

  static std::string json_escape(std::string_view string) {
    std::string result;
    result.reserve(string.size());
    for (char c : string) result += json_escape(c);
    return result;
  }

  static std::string json_key_value(std::string_view key, std::string_view value) {
    return '"' + json_escape(key) + "\": \"" + json_escape(value) + '"';
  }

  static std::string json_key_value(std::string_view key, const double &value) {
    return '"' + json_escape(key) + "\": " + std::to_string(value);
  }

  static std::string json_key_value(std::string_view key, const int32_t &value) {
    return '"' + json_escape(key) + "\": " + std::to_string(value);
  }

  static std::string json_key_value(std::string_view key, const uint32_t &value) {
    return '"' + json_escape(key) + "\": " + std::to_string(value);
  }

  static std::string json_key_value(std::string_view key, const int64_t &value) {
    return '"' + json_escape(key) + "\": " + std::to_string(value);
  }

  static std::string json_key_value(std::string_view key, const uint64_t &value) {
    return '"' + json_escape(key) + "\": " + std::to_string(value);
  }

  static std::string date_string() {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%FT%T");
    return oss.str();
  }

  const std::string output_file;
  std::ofstream out;
  std::string indent;
  bool is_open = false;
  bool reporting_benchmarks = false;

  void open() {
    out.open(output_file);
    if (!out.is_open()) {
      std::cerr << "Invalid output file: " << output_file << std::endl;
      std::exit(1);
    }
    out << "{\n";
    indent += INDENT;
    is_open = true;
  }

  void close() {
    if (reporting_benchmarks) stop_reporting_runs();
    indent.resize(indent.size() - INDENT.size());
    out << "}\n";
    out.flush();
    is_open = false;
  }

  void start_reporting_runs() {
    out << indent << "\"runs\": [\n";
    indent += INDENT;
    reporting_benchmarks = true;
  }

  void stop_reporting_runs() {
    indent.resize(indent.size() - INDENT.size());
    out << '\n' << indent << "]\n";
    reporting_benchmarks = false;
  }

 public:
  Reporter(std::string output_file) : output_file{output_file} {}

  ~Reporter() {
    if (is_open) close();
  }

  void report_context(const std::map<std::string, std::string> &context = {}) {
    if (!is_open) open();
    out << indent << "\"context\": {\n";
    indent += INDENT;

    for (const auto &[key, value] : context) {
      out << indent << json_key_value(key, value) << ",\n";
    }
    out << indent << json_key_value("date", date_string()) << '\n';

    indent.resize(indent.size() - INDENT.size());
    out << indent << "},\n";
    out.flush();
  }

  void report_benchmark_run(
      std::string_view benchmark_name, std::string_view lock_name, BenchmarkResult result) {
    if (!is_open) open();
    if (!reporting_benchmarks)
      start_reporting_runs();
    else
      out << ",\n";

    double iterations_per_sec =
        result.iterations / ((std::chrono::duration<double>)result.duration).count();
    std::cout << benchmark_name << '/' << lock_name << '\t'
              << result.duration.count() / result.iterations << " ns\t" << result.iterations
              << "\titerations_per_sec=" << iterations_per_sec << std::endl;

    out << indent << "{\n";
    indent += INDENT;
    out << indent << json_key_value("duration_ns", result.duration.count()) << ",\n"
        << indent << json_key_value("iterations", result.iterations) << ",\n"
        << indent << json_key_value("iterations_per_process_min", result.iterations_per_process_min)
        << ",\n"
        << indent << json_key_value("iterations_per_process_max", result.iterations_per_process_max)
        << ",\n"
        << indent
        << json_key_value("iterations_per_process_median", result.iterations_per_process_median)
        << ",\n"
        << indent
        << json_key_value("iterations_per_process_mean", result.iterations_per_process_mean)
        << ",\n"
        << indent << json_key_value("iterations_per_process_sd", result.iterations_per_process_sd)
        << ",\n"
        << indent << json_key_value("iterations_per_process_cv", result.iterations_per_process_cv)
        << ",\n";

    out << indent << "\"stats\": {";
    indent += INDENT;
    bool first = true;
    for (const auto &[key, value] : result.stats) {
      if (!first) out << ',';
      first = false;
      out << '\n' << indent << json_key_value(key, value);
    }
    out << '\n';
    indent.resize(indent.size() - INDENT.size());
    out << indent << "}\n";

    indent.resize(indent.size() - INDENT.size());
    out << indent << "}";
    out.flush();
  }

  void report_uncontested(
      std::string_view benchmark_name, std::string_view lock_name, size_t lock_count,
      int64_t same_process_master_rank_ns, int64_t same_node_master_rank_ns,
      int64_t different_node_master_rank_ns, int64_t same_process_master_node_ns,
      int64_t same_node_master_node_ns, int64_t different_node_master_node_ns,
      int64_t same_process_slave_node_ns, int64_t same_node_slave_node_ns,
      int64_t different_node_slave_node_ns) {
    if (!is_open) open();
    if (!reporting_benchmarks)
      start_reporting_runs();
    else
      out << ",\n";

    std::cout << benchmark_name << '/' << lock_name
              << "\t1a=" << same_process_master_rank_ns / lock_count
              << "\t2a=" << same_node_master_rank_ns / lock_count
              << "\t3a=" << different_node_master_rank_ns / lock_count
              << "\t1b=" << same_process_master_node_ns / lock_count
              << "\t2b=" << same_node_master_node_ns / lock_count
              << "\t3b=" << different_node_master_node_ns / lock_count
              << "\t1c=" << same_process_slave_node_ns / lock_count
              << "\t2c=" << same_node_slave_node_ns / lock_count
              << "\t3c=" << different_node_slave_node_ns / lock_count << std::endl;

    out << indent << "{\n";
    indent += INDENT;
    out << indent
        << json_key_value("same_process_master_rank_ns", same_process_master_rank_ns / lock_count)
        << ",\n"
        << indent
        << json_key_value("same_node_master_rank_ns", same_node_master_rank_ns / lock_count)
        << ",\n"
        << indent
        << json_key_value(
               "different_node_master_rank_ns", different_node_master_rank_ns / lock_count)
        << ",\n"
        << indent
        << json_key_value("same_process_master_node_ns", same_process_master_node_ns / lock_count)
        << ",\n"
        << indent
        << json_key_value("same_node_master_node_ns", same_node_master_node_ns / lock_count)
        << ",\n"
        << indent
        << json_key_value(
               "different_node_master_node_ns", different_node_master_node_ns / lock_count)
        << ",\n"
        << indent
        << json_key_value("same_process_slave_node_ns", same_process_slave_node_ns / lock_count)
        << ",\n"
        << indent << json_key_value("same_node_slave_node_ns", same_node_slave_node_ns / lock_count)
        << ",\n"
        << indent
        << json_key_value("different_node_slave_node_ns", different_node_slave_node_ns / lock_count)
        << '\n';
    indent.resize(indent.size() - INDENT.size());
    out << indent << "}";
    out.flush();
  }
};

}  // namespace mpi_lock_bench
