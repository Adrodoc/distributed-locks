#pragma once

#include <limits.h>
#include <mpi.h>
#include "Lock.cpp"
#include "log.cpp"
#include "mpi_utils/mpi_utils.cpp"

class McsLockTwoSidedWithCohortDetection : public Lock
{
private:
    enum status
    {
        ACQUIRE_GLOBAL = 0,
        WAIT = INT_MAX,
    };
    struct memory_layout
    {
        alignas(64) int next;
        alignas(64) int tail;
    };
    static constexpr MPI_Aint next_disp = offsetof(memory_layout, next);
    static constexpr MPI_Aint tail_disp = offsetof(memory_layout, tail);

    const MPI_Comm comm;
    const int master_rank;
    const int rank;
    memory_layout *mem;
    MPI_Win win;

public:
    McsLockTwoSidedWithCohortDetection(const McsLockTwoSidedWithCohortDetection &) = delete;
    McsLockTwoSidedWithCohortDetection(const MPI_Comm comm = MPI_COMM_WORLD, const int master_rank = 0)
        : comm{comm},
          master_rank{master_rank},
          rank{get_rank(comm)}
    {
        // log() << "entering McsLockTwoSidedWithCohortDetection" << std::endl;
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "accumulate_ordering", "none");
        MPI_Info_set(info, "same_disp_unit", "true");
        MPI_Info_set(info, "same_size", "true");
        MPI_Win_allocate(sizeof(memory_layout), 1, info, comm, &mem, &win);

        if (rank == master_rank)
            mem->tail = -1;

        MPI_Win_lock_all(0, win);
        MPI_Barrier(comm);
        // log() << "exiting McsLockTwoSidedWithCohortDetection" << std::endl;
    }

    ~McsLockTwoSidedWithCohortDetection()
    {
        // log() << "entering ~McsLockTwoSidedWithCohortDetection" << std::endl;
        MPI_Win_unlock_all(win);
        MPI_Win_free(&win);
        // log() << "exiting ~McsLockTwoSidedWithCohortDetection" << std::endl;
    }

    bool alone()
    {
        return mem->next == -1;
    }

    void acquire() { acquire_cd(); }

    int acquire_cd()
    {
        // log() << "entering acquire()" << std::endl;
        mem->next = -1;
        MPI_Win_sync(win);

        // log() << "finding predecessor" << std::endl;
        int predecessor;
        MPI_Fetch_and_op(&rank, &predecessor, MPI_INT,
                         master_rank, tail_disp, MPI_REPLACE, win);
        MPI_Win_flush(master_rank, win);
        if (predecessor != -1)
        {
            // log() << "notifying predecessor: " << predecessor << std::endl;
            MPI_Put(&rank, 1, MPI_INT,
                    predecessor, next_disp, 1, MPI_INT,
                    win);
            MPI_Win_flush(predecessor, win);

            // log() << "waiting for predecessor" << std::endl;
            int status;
            MPI_Recv(&status, 1, MPI_INT, predecessor, 0, comm, MPI_STATUS_IGNORE);
            return status;
        }
        // log() << "exiting acquire()" << std::endl;
        return ACQUIRE_GLOBAL;
    }

    void release() { release_cd(false); }

    void release_cd(int status)
    {
        // log() << "entering release()" << std::endl;
        int successor = mem->next;
        if (successor == -1)
        {
            // log() << "nulling tail" << std::endl;
            int null_rank = -1;
            int old_value;
            MPI_Compare_and_swap(&null_rank, &rank, &old_value, MPI_INT,
                                 master_rank, tail_disp, win);
            MPI_Win_flush(master_rank, win);
            if (old_value == rank)
            {
                // log() << "exiting release()" << std::endl;
                return;
            }
            // log() << "waiting for successor" << std::endl;
            while ((successor = mem->next) == -1)
            {
                // Trigger MPI progress
                int flag;
                MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, MPI_STATUS_IGNORE);
                MPI_Win_sync(win);
            }
        }
        // log() << "notifying successor: " << successor << std::endl;
        MPI_Send(&status, 1, MPI_INT, successor, 0, comm);
        // log() << "exiting release()" << std::endl;
    }
};
