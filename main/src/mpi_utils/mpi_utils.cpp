#pragma once

#include <mpi.h>

#include <chrono>

#if UINTPTR_MAX == UCHAR_MAX
#define MPI_UINTPTR_T MPI_UNSIGNED_CHAR
#elif UINTPTR_MAX == USHRT_MAX
#define MPI_UINTPTR_T MPI_UNSIGNED_SHORT
#elif UINTPTR_MAX == UINT_MAX
#define MPI_UINTPTR_T MPI_UNSIGNED
#elif UINTPTR_MAX == ULONG_MAX
#define MPI_UINTPTR_T MPI_UNSIGNED_LONG
#elif UINTPTR_MAX == ULLONG_MAX
#define MPI_UINTPTR_T MPI_UNSIGNED_LONG_LONG
#else
#error "Unsupported value of UINTPTR_MAX"
#endif

int get_rank(const MPI_Comm comm = MPI_COMM_WORLD) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

int get_size(const MPI_Comm comm = MPI_COMM_WORLD) {
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}

void *get_win_base(const MPI_Win win, std::ptrdiff_t disp = 0) {
  uint8_t *mem;
  int flag;
  MPI_Win_get_attr(win, MPI_WIN_BASE, &mem, &flag);
  return mem + disp;
}

MPI_Comm split_comm_shared(const MPI_Comm comm = MPI_COMM_WORLD) {
  MPI_Comm local_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  return local_comm;
}

MPI_Win create_win(void *mem, MPI_Aint size, MPI_Comm comm) {
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "accumulate_ordering", "none");
  MPI_Info_set(info, "same_disp_unit", "true");
  MPI_Win win;
  MPI_Win_create(mem, size, 1, info, comm, &win);
  return win;
}

MPI_Win allocate_shared_win(MPI_Comm comm, MPI_Aint size) {
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Info_set(info, "accumulate_ordering", "none");
  MPI_Info_set(info, "same_disp_unit", "true");
  MPI_Info_set(info, "alloc_shared_noncontig", "true");
  void *mem;
  MPI_Win win;
  MPI_Win_allocate_shared(size, 1, info, comm, &mem, &win);
  return win;
}

int get_node_id(const MPI_Comm comm = MPI_COMM_WORLD) {
  int global_rank;  // Rank outside local node
  MPI_Comm_rank(comm, &global_rank);

  // Communicator for inside local node
  MPI_Comm local_comm = split_comm_shared(comm);

  int local_rank;  // Rank inside local node
  MPI_Comm_rank(local_comm, &local_rank);

  // Since the ranks in comm are unique we can use any one of those ranks as the id of our node.
  // Here we use the global rank of the process with local rank 0 as the node_id.
  int node_id = global_rank;
  MPI_Bcast(&node_id, 1, MPI_INT, 0, local_comm);

  MPI_Comm_free(&local_comm);
  return node_id;
}

void spin_with_mpi_progress(std::chrono::nanoseconds duration) {
  using clock = std::chrono::high_resolution_clock;
  auto spin_until = clock::now() + duration;
  int flag;
  while (clock::now() < spin_until)
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
}
