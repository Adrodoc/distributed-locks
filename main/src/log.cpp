#pragma once

#include <iomanip>
#include <iostream>

#include "mpi_utils/mpi_utils.cpp"

std::string formatted_time() {
  using namespace std::chrono;

  // get current time
  auto now = system_clock::now();
  std::ostringstream oss;

  // convert to std::time_t in order to convert to std::tm (broken time)
  auto timer = system_clock::to_time_t(now);
  // convert to broken time
  std::tm *bt = std::localtime(&timer);
  oss << std::put_time(bt, "%T");  // HH:MM:SS

  auto time_since_epoch = now.time_since_epoch();

  // get number of milliseconds for the current second (remainder after division into seconds)
  auto millis = duration_cast<milliseconds>(time_since_epoch) % 1000;
  oss << '.' << std::setfill('0') << std::setw(3) << millis.count();

  // get number of microseconds for the current millisecond (remainder after division into
  // milliseconds)
  auto micros = duration_cast<microseconds>(time_since_epoch) % 1000;
  oss << '.' << std::setfill('0') << std::setw(3) << micros.count();

  return oss.str();
}

std::ostream &log() {
  return std::clog << formatted_time() << " (" << get_rank() << "/" << get_size() << "): ";
}
