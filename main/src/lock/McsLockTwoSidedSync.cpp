#pragma once

#include <mpi.h>
#include "Lock.cpp"
#include "log.cpp"
#include "mpi_utils/mpi_utils.cpp"

class McsLockTwoSidedSync : public Lock
{
private:
    struct memory_layout
    {
        alignas(64) int tail;
    };
    static constexpr MPI_Aint tail_disp = offsetof(memory_layout, tail);

    const MPI_Comm comm;
    const int master_rank;
    const int rank;
    memory_layout *mem;
    MPI_Win win;

public:
    McsLockTwoSidedSync(const McsLockTwoSidedSync &) = delete;
    McsLockTwoSidedSync(const MPI_Comm comm = MPI_COMM_WORLD, const int master_rank = 0)
        : comm{comm},
          master_rank{master_rank},
          rank{get_rank(comm)}
    {
        // log() << "entering McsLockTwoSidedSync" << std::endl;
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
        // log() << "exiting McsLockTwoSidedSync" << std::endl;
    }

    ~McsLockTwoSidedSync()
    {
        // log() << "entering ~McsLockTwoSidedSync" << std::endl;
        MPI_Win_unlock_all(win);
        // log() << "exiting ~McsLockTwoSidedSync" << std::endl;
    }

    void acquire()
    {
        // log() << "entering acquire()" << std::endl;

        // log() << "finding predecessor" << std::endl;
        int predecessor;
        MPI_Fetch_and_op(&rank, &predecessor, MPI_INT,
                         master_rank, tail_disp, MPI_REPLACE, win);
        MPI_Win_flush(master_rank, win);
        if (predecessor != -1)
        {
            // log() << "waiting for predecessor" << std::endl;
            MPI_Ssend(NULL, 0, MPI_UINT8_T, predecessor, 0, comm);
        }
        // log() << "exiting acquire()" << std::endl;
    }

    void release()
    {
        // log() << "entering release()" << std::endl;

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
        MPI_Recv(NULL, 0, MPI_UINT8_T, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE);

        // log() << "exiting release()" << std::endl;
    }
};
