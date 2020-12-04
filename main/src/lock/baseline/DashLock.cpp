#include <libdash.h>

#include "lock/Lock.cpp"
#include "log.cpp"

class DashLock : public Lock {
  dash::Mutex mx;

 public:
  static constexpr std::string_view NAME() { return "DashLock"; }

  DashLock() {
    // log() << "entering DashLock()" << std::endl;
    // log() << "exiting DashLock()" << std::endl;
  }

  DashLock(DashLock &&other) noexcept : mx{std::move(other.mx)} {}

  ~DashLock() {
    // log() << "entering ~DashLock()" << std::endl;
    // log() << "exiting ~DashLock()" << std::endl;
  }

  MPI_Comm communicator() { return MPI_COMM_WORLD; }

#ifdef STATS
  std::map<std::string, double> stats() { return {}; }
#endif

  void acquire() {
    // log() << "entering acquire()" << std::endl;
    mx.lock();
    // log() << "exiting acquire()" << std::endl;
  }

  void release() {
    // log() << "entering release()" << std::endl;
    mx.unlock();
    // log() << "exiting release()" << std::endl;
  }
};
