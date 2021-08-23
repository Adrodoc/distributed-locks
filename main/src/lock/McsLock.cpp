#pragma once

#include <mpi.h>
#include "Lock.cpp"
#include "log.cpp"
#include "mpi_utils/mpi_utils.cpp"

class McsLock : public Lock
{
private:
    struct memory_layout
    {
        alignas(64) bool locked;
        alignas(64) int next;
        alignas(64) int tail;
    };
    static constexpr MPI_Aint locked_disp = offsetof(memory_layout, locked);
    static constexpr MPI_Aint next_disp = offsetof(memory_layout, next);
    static constexpr MPI_Aint tail_disp = offsetof(memory_layout, tail);

    const MPI_Comm comm;
    const int master_rank;
    const int rank;
    memory_layout *mem;
    MPI_Win win;

public:
    McsLock(const McsLock &) = delete;
    McsLock(const MPI_Comm comm = MPI_COMM_WORLD, const int master_rank = 0)
        : comm{comm},
          master_rank{master_rank},
          rank{get_rank(comm)}
    {
        // log() << "entering McsLock" << std::endl;
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
        // log() << "exiting McsLock" << std::endl;
    }

    ~McsLock()
    {
        // log() << "entering ~McsLock" << std::endl;
        MPI_Win_unlock_all(win);
        MPI_Win_free(&win);
        // log() << "exiting ~McsLock" << std::endl;
    }

    void acquire()
    {
        // log() << "entering acquire()" << std::endl;
        mem->locked = true;
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
            while (mem->locked)
            {
                // Trigger MPI progress
                int flag;
                MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, MPI_STATUS_IGNORE);
                MPI_Win_sync(win);
            }
        }
        // log() << "exiting acquire()" << std::endl;
    }

    void release()
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
        bool false_ = false;
        MPI_Put(&false_, 1, MPI_CXX_BOOL,
                successor, locked_disp, 1, MPI_CXX_BOOL,
                win);
        MPI_Win_flush(successor, win);
        // log() << "exiting release()" << std::endl;
    }
};
