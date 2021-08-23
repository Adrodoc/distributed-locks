#pragma once

#include <mpi.h>
#include "Lock.cpp"
#include "log.cpp"
#include "mpi_utils/mpi_utils.cpp"

class TktLock : public Lock
{
private:
    struct memory_layout
    {
        alignas(64) int next_ticket;
        alignas(64) int now_serving;
    };
    static constexpr MPI_Aint next_ticket_disp = offsetof(memory_layout, next_ticket);
    static constexpr MPI_Aint now_serving_disp = offsetof(memory_layout, now_serving);

    const MPI_Comm comm;
    const int master_rank;
    const int rank;
    memory_layout *mem;
    MPI_Win win;

public:
    TktLock(const TktLock &) = delete;
    TktLock(const MPI_Comm comm = MPI_COMM_WORLD, const int master_rank = 0)
        : comm{comm},
          master_rank{master_rank},
          rank{get_rank(comm)}
    {
        // log() << "entering TktLock" << std::endl;
        MPI_Aint size = (rank == master_rank ? 1 : 0) * sizeof(memory_layout);
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "accumulate_ordering", "none");
        MPI_Info_set(info, "same_disp_unit", "true");
        MPI_Win_allocate(size, 1, info, comm, &mem, &win);

        if (rank == master_rank)
        {
            mem->next_ticket = 0;
            mem->now_serving = 0;
        }

        MPI_Win_lock_all(0, win);
        MPI_Barrier(comm);
        // log() << "exiting TktLock" << std::endl;
    }

    ~TktLock()
    {
        // log() << "entering ~TktLock" << std::endl;
        MPI_Win_unlock_all(win);
        MPI_Win_free(&win);
        // log() << "exiting ~TktLock" << std::endl;
    }

    void acquire()
    {
        // log() << "entering acquire()" << std::endl;
        const int one = 1;
        int my_ticket;
        MPI_Fetch_and_op(&one, &my_ticket, MPI_INT, master_rank, next_ticket_disp, MPI_SUM, win);

        // log() << "waiting for ticket: " << my_ticket << std::endl;
        int now_serving;
        do
        {
            MPI_Fetch_and_op(NULL, &now_serving, MPI_INT, master_rank, now_serving_disp, MPI_NO_OP, win);
            MPI_Win_flush(master_rank, win);
        } while (my_ticket != now_serving);
        // log() << "exiting acquire()" << std::endl;
    }

    void release()
    {
        // log() << "entering release()" << std::endl;
        const int one = 1;
        int my_ticket;
        MPI_Fetch_and_op(&one, &my_ticket, MPI_INT, master_rank, now_serving_disp, MPI_SUM, win);
        MPI_Win_flush(master_rank, win);
        // log() << "released ticket: " << my_ticket << std::endl;
        // log() << "exiting release()" << std::endl;
    }
};
