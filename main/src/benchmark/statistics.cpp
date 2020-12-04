#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace statistics {
double min(const std::vector<double> &sample) {
  return *std::min_element(sample.begin(), sample.end());
}

double max(const std::vector<double> &sample) {
  return *std::max_element(sample.begin(), sample.end());
}

double median(const std::vector<double> &sample) {
  if (sample.empty()) return 0.0;
  std::vector<double> copy(sample);

  const auto center = copy.begin() + sample.size() / 2;
  std::nth_element(copy.begin(), center, copy.end());

  // Odd number of samples?
  // Yes: center is the median
  // No: we are looking for the average between center and the value before
  if (sample.size() % 2 == 1) return *center;

  const auto center2 = center - 1;
  std::nth_element(copy.begin(), center2, copy.end());
  return (*center + *center2) / 2.0;
}

// Computes the sum of the given sample
double sum(const std::vector<double> &sample) {
  return std::accumulate(sample.begin(), sample.end(), 0.0);
}

// Computes the arithmetic mean of the given sample
double mean(const std::vector<double> &sample) {
  if (sample.empty()) return 0.0;
  return sum(sample) / sample.size();
}

// Computes the sum of the squares of the given sample
double sum_of_squares(const std::vector<double> &sample) {
  return std::inner_product(sample.begin(), sample.end(), sample.begin(), 0.0);
}

// Computes the standard deviation of the given sample
double standard_deviation(const std::vector<double> &sample) {
  // Sample standard deviation is undefined for n = 1
  const auto size = sample.size();
  if (size <= 1) return 0.0;

  const double mean = statistics::mean(sample);
  const double population_variance = sum_of_squares(sample) / size - mean * mean;
  const double bessels_correction = size / (size - 1.0);
  const double sample_variance = population_variance * bessels_correction;
  return std::sqrt(sample_variance);
}

double coefficient_of_variation(const std::vector<double> &sample) {
  return standard_deviation(sample) / mean(sample);
}
}  // namespace statistics
