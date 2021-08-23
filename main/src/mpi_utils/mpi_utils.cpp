#pragma once

#include <mpi.h>

int get_rank(const MPI_Comm comm = MPI_COMM_WORLD)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

int split_comm_shared(const MPI_Comm comm = MPI_COMM_WORLD)
{
    MPI_Comm local_comm;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    return local_comm;
}

int get_node_id(const MPI_Comm comm = MPI_COMM_WORLD)
{
    int global_rank; // Rank outside local node
    MPI_Comm_rank(comm, &global_rank);

    // Communicator for inside local node
    MPI_Comm local_comm = split_comm_shared(comm);

    int local_rank; // Rank inside local node
    MPI_Comm_rank(local_comm, &local_rank);

    // Since the ranks in comm are unique we can use any one of those ranks as the id of our node.
    // Here we use the global rank of the process with local rank 0 as the node_id.
    int node_id = global_rank;
    MPI_Bcast(&node_id, 1, MPI_INT, 0, local_comm);

    MPI_Comm_free(&local_comm);
    return node_id;
}
