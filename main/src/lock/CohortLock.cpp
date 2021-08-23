#pragma once

#include <mpi.h>
#include "Lock.cpp"
#include "log.cpp"
#include "McsLockWithCohortDetection.cpp"
#include "mpi_utils/mpi_utils.cpp"
#include "TktLock.cpp"

template <typename G, typename L>
class CohortLock : public Lock
{
private:
    static constexpr uint8_t MAX_LOCAL_PASSES = 63;

    struct memory_layout
    {
        uint8_t local_pass_cnt;
    };
    static constexpr MPI_Aint local_pass_cnt_disp = offsetof(memory_layout, local_pass_cnt);

    const MPI_Comm comm;
    const int master_rank;
    const int rank;
    memory_layout *mem;
    MPI_Win win;

    G global_lock;
    L local_lock;

public:
    CohortLock(const CohortLock &) = delete;
    CohortLock(const MPI_Comm comm = MPI_COMM_WORLD, const int master_rank = 0)
        : comm{comm},
          master_rank{master_rank},
          rank{get_rank(comm)},
          global_lock{comm, master_rank},
          local_lock{split_comm_shared(comm)}
    {
        // log() << "entering CohortLock" << std::endl;
        MPI_Aint size = (rank == master_rank ? 1 : 0) * sizeof(memory_layout);
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "accumulate_ordering", "none");
        MPI_Info_set(info, "same_disp_unit", "true");
        MPI_Win_allocate(size, 1, info, comm, &mem, &win);

        if (rank == master_rank)
            mem->local_pass_cnt = 0;

        MPI_Win_lock_all(0, win);
        // log() << "exiting CohortLock" << std::endl;
    }

    ~CohortLock()
    {
        // log() << "entering ~CohortLock" << std::endl;
        MPI_Win_unlock_all(win);
        MPI_Win_free(&win);
        // log() << "exiting ~CohortLock" << std::endl;
    }

    void acquire()
    {
        // log() << "entering cohort acquire()" << std::endl;
        bool local_release = local_lock.acquire_cd();
        // log() << "local release: " << local_release << std::endl;

        if (local_release)
        {
            // log() << "exiting cohort acquire()" << std::endl;
            return;
        }

        global_lock.acquire();
        // log() << "exiting cohort acquire()" << std::endl;
    }

    void release()
    {
        // log() << "entering cohort release()" << std::endl;
        bool alone = local_lock.alone();
        // log() << "alone: " << alone << std::endl;
        if (!alone && may_pass_local())
        {
            local_lock.release_cd(true);
        }
        else
        {
            const uint8_t zero = 0;
            MPI_Put(&zero, 1, MPI_UINT8_T, master_rank, local_pass_cnt_disp, 1, MPI_UINT8_T, win);
            MPI_Win_flush(master_rank, win);
            global_lock.release();
            local_lock.release_cd(false);
        }
        // log() << "exiting cohort release()" << std::endl;
    }

    bool may_pass_local()
    {
        const uint8_t zero = 0;
        const uint8_t one = 1;
        uint8_t local_pass_cnt;
        MPI_Fetch_and_op(&one, &local_pass_cnt, MPI_UINT8_T, master_rank, local_pass_cnt_disp, MPI_SUM, win);
        MPI_Win_flush(master_rank, win);
        return local_pass_cnt < MAX_LOCAL_PASSES;
    }
};
