#pragma once

#include <mpi.h>
#include "Lock.cpp"
#include "log.cpp"
#include "McsLockWithCohortDetection.cpp"
#include "mpi_utils/mpi_utils.cpp"
#include "TktLock.cpp"

class CTktMcsLockOptimizedCounter : public Lock
{
private:
    enum status
    {
        ACQUIRE_GLOBAL = 0,
        WAIT = INT_MAX,
    };
    static constexpr int MAX_LOCAL_PASSES = 63;

    const MPI_Comm comm;
    const int master_rank;
    const int rank;

    TktLock global_lock;
    McsLockWithCohortDetection local_lock;

    int status;

public:
    CTktMcsLockOptimizedCounter(const CTktMcsLockOptimizedCounter &) = delete;
    CTktMcsLockOptimizedCounter(const MPI_Comm comm = MPI_COMM_WORLD, const int master_rank = 0)
        : comm{comm},
          master_rank{master_rank},
          rank{get_rank(comm)},
          global_lock{comm, master_rank},
          local_lock{split_comm_shared(comm)}
    {
        // log() << "entering CTktMcsLockOptimizedCounter" << std::endl;
        // log() << "exiting CTktMcsLockOptimizedCounter" << std::endl;
    }

    ~CTktMcsLockOptimizedCounter()
    {
        // log() << "entering ~CTktMcsLockOptimizedCounter" << std::endl;
        // log() << "exiting ~CTktMcsLockOptimizedCounter" << std::endl;
    }

    void acquire()
    {
        // log() << "entering cohort acquire()" << std::endl;
        status = local_lock.acquire_cd();
        // log() << "local release: " << local_release << std::endl;

        if (status == ACQUIRE_GLOBAL)
            global_lock.acquire();

        // log() << "exiting cohort acquire()" << std::endl;
    }

    void release()
    {
        // log() << "entering cohort release()" << std::endl;
        bool alone = local_lock.alone();
        // log() << "alone: " << alone << std::endl;
        bool may_pass_local_ = may_pass_local();
        // log() << "may_pass_local: " << may_pass_local_ << std::endl;
        if (!alone && may_pass_local_)
        {
            local_lock.release_cd(status + 1);
        }
        else
        {
            global_lock.release();
            local_lock.release_cd(ACQUIRE_GLOBAL);
        }
        // log() << "exiting cohort release()" << std::endl;
    }

    bool may_pass_local()
    {
        return status < MAX_LOCAL_PASSES;
    }
};
