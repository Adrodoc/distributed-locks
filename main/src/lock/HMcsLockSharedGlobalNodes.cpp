#pragma once

#include <mpi.h>
#include "Lock.cpp"
#include "log.cpp"
#include "McsLockWithCohortDetection.cpp"
#include "mpi_utils/mpi_utils.cpp"
#include "TktLock.cpp"

class HMcsLockSharedGlobalNodes : public Lock
{
private:
    enum status
    {
        ACQUIRE_GLOBAL = 0,
        WAIT = INT_MAX,
    };
    static constexpr int MAX_LOCAL_PASSES = 63;

    struct memory_layout
    {
        alignas(64) bool locked;
        alignas(64) int next;
        alignas(64) int tail;
    };
    static constexpr MPI_Aint locked_disp = offsetof(memory_layout, locked);
    static constexpr MPI_Aint next_disp = offsetof(memory_layout, next);
    static constexpr MPI_Aint tail_disp = offsetof(memory_layout, tail);

    // Memory of local master in global window
    memory_layout *mem;

    // Global communicator
    const MPI_Comm global_comm;
    // Global rank of global master
    const int global_master_rank;
    // Global rank
    const int global_rank;
    // Global window
    MPI_Win global_win;

    const MPI_Comm local_comm;
    // Global rank of local master
    int local_master_global_rank;
    // Local rank
    const int local_rank;
    // Local window
    MPI_Win local_win;

    McsLockWithCohortDetection local_lock;

    int status;

public:
    HMcsLockSharedGlobalNodes(const HMcsLockSharedGlobalNodes &) = delete;
    HMcsLockSharedGlobalNodes(const MPI_Comm comm = MPI_COMM_WORLD, const int global_master_rank = 0, const int local_master_rank = 0)
        : global_comm{comm},
          global_master_rank{global_master_rank},
          global_rank{get_rank(global_comm)},
          local_comm{split_comm_shared(comm)},
          local_rank{get_rank(local_comm)},
          local_lock{local_comm, local_master_rank}
    {
        // log() << "entering HMcsLockSharedGlobalNodes" << std::endl;
        MPI_Aint size = (local_rank == local_master_rank ? 1 : 0) * sizeof(memory_layout);

        // Allocate local window
        MPI_Info local_info;
        MPI_Info_create(&local_info);
        MPI_Info_set(local_info, "accumulate_ordering", "none");
        MPI_Info_set(local_info, "same_disp_unit", "true");
        MPI_Info_set(local_info, "alloc_shared_noncontig", "true");
        MPI_Win_allocate_shared(size, 1, local_info, local_comm, &mem, &local_win);

        // Create global window
        MPI_Info global_info;
        MPI_Info_create(&global_info);
        MPI_Info_set(global_info, "accumulate_ordering", "none");
        MPI_Info_set(global_info, "same_disp_unit", "true");
        MPI_Win_create(mem, size, 1, global_info, global_comm, &global_win);

        if (local_rank != local_master_rank)
        { // Update mem to point to memory of local master rank
            MPI_Aint local_size;
            int local_disp_unit;
            MPI_Win_shared_query(local_win, local_master_rank, &local_size, &local_disp_unit, &mem);
        }

        local_master_global_rank = global_rank;
        MPI_Bcast(&local_master_global_rank, 1, MPI_INT, local_master_rank, local_comm);

        if (global_rank == global_master_rank)
            mem->tail = -1;

        MPI_Win_lock_all(0, global_win);
        MPI_Barrier(comm);
        // log() << "exiting HMcsLockSharedGlobalNodes" << std::endl;
    }

    ~HMcsLockSharedGlobalNodes()
    {
        // log() << "entering ~HMcsLockSharedGlobalNodes" << std::endl;
        MPI_Win_unlock_all(global_win);
        MPI_Win_free(&global_win);
        MPI_Win_free(&local_win);
        // log() << "exiting ~HMcsLockSharedGlobalNodes" << std::endl;
    }

    void acquire()
    {
        // log() << "entering cohort acquire()" << std::endl;
        status = local_lock.acquire_cd();
        // log() << "local release: " << local_release << std::endl;

        if (status == ACQUIRE_GLOBAL)
            acquire_global();

        // log() << "exiting cohort acquire()" << std::endl;
    }

    void acquire_global()
    {
        // log() << "entering acquire_global()" << std::endl;
        mem->locked = true;
        mem->next = -1;
        MPI_Win_sync(global_win);

        // log() << "finding predecessor" << std::endl;
        int predecessor;
        MPI_Fetch_and_op(&local_master_global_rank, &predecessor, MPI_INT,
                         global_master_rank, tail_disp, MPI_REPLACE, global_win);
        MPI_Win_flush(global_master_rank, global_win);
        if (predecessor != -1)
        {
            // log() << "notifying predecessor: " << predecessor << std::endl;
            MPI_Put(&local_master_global_rank, 1, MPI_INT,
                    predecessor, next_disp, 1, MPI_INT,
                    global_win);
            MPI_Win_flush(predecessor, global_win);

            // log() << "waiting for predecessor" << std::endl;
            while (mem->locked)
            {
                // Trigger MPI progress
                int flag;
                MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, global_comm, &flag, MPI_STATUS_IGNORE);
                MPI_Win_sync(global_win);
            }
        }
        // log() << "exiting acquire_global()" << std::endl;
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
            // log() << "global release after " << status << " local passes" << std::endl;
            release_global();
            local_lock.release_cd(ACQUIRE_GLOBAL);
        }
        // log() << "exiting cohort release()" << std::endl;
    }

    void release_global()
    {
        // log() << "entering release()" << std::endl;
        int successor = mem->next;
        if (successor == -1)
        {
            // log() << "nulling tail" << std::endl;
            int null_rank = -1;
            int old_value;
            MPI_Compare_and_swap(&null_rank, &local_master_global_rank, &old_value, MPI_INT,
                                 global_master_rank, tail_disp, global_win);
            MPI_Win_flush(global_master_rank, global_win);
            if (old_value == local_master_global_rank)
            {
                // log() << "exiting release()" << std::endl;
                return;
            }
            // log() << "waiting for successor" << std::endl;
            while ((successor = mem->next) == -1)
            {
                // Trigger MPI progress
                int flag;
                MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, global_comm, &flag, MPI_STATUS_IGNORE);
                MPI_Win_sync(global_win);
            }
        }
        // log() << "notifying successor: " << successor << std::endl;
        bool false_ = false;
        MPI_Put(&false_, 1, MPI_CXX_BOOL,
                successor, locked_disp, 1, MPI_CXX_BOOL,
                global_win);
        MPI_Win_flush(successor, global_win);
        // log() << "exiting release()" << std::endl;
    }

    bool may_pass_local()
    {
        return status < MAX_LOCAL_PASSES;
    }
};
