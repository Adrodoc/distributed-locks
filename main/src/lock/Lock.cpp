#pragma once

#include <mpi.h>

class Lock {
 public:
  virtual MPI_Comm communicator() = 0;
#ifdef STATS
  virtual std::map<std::string, double> stats() = 0;
#endif
  virtual void acquire() = 0;
  virtual void release() = 0;
};
