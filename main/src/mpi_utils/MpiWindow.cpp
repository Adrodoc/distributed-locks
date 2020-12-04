#pragma once

#include <mpi.h>

#include <utility>

#include "mpi_utils.cpp"

class MpiWindow {
  const bool owned_win;
  MPI_Win shared_win;

  MpiWindow(
      MPI_Comm comm, MPI_Win win, void *mem, int mem_rank, MPI_Win shared_win = MPI_WIN_NULL,
      MPI_Aint disp = 0, bool owned_win = true)
      : comm{comm},
        win{win},
        mem{mem},
        mem_rank{mem_rank},
        shared_win{shared_win},
        disp{disp},
        owned_win{owned_win} {}

 public:
  const MPI_Comm comm;  // The communicator used by win
  MPI_Win win;  // The window
  void *const mem;  // The memory behind win
  const int mem_rank;  // The rank that owns the memory
  const MPI_Aint disp;  // The offset of mem in the whole window memory

  MpiWindow(MpiWindow &&other)
      : comm{std::move(other.comm)},
        win{std::move(other.win)},
        mem{std::move(other.mem)},
        mem_rank{std::move(other.mem_rank)},
        shared_win{std::move(other.shared_win)},
        disp{std::move(other.disp)},
        owned_win{std::move(other.owned_win)} {
    other.win = MPI_WIN_NULL;
    other.shared_win = MPI_WIN_NULL;
  }

  ~MpiWindow() {
    if (owned_win && win != MPI_WIN_NULL) MPI_Win_free(&win);
    if (shared_win != MPI_WIN_NULL) MPI_Win_free(&shared_win);
  }

  // Allocates an ordinary window
  static MpiWindow allocate(MPI_Aint size, MPI_Comm comm) {
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "accumulate_ordering", "none");
    MPI_Info_set(info, "same_disp_unit", "true");
    MPI_Win win;
    void *mem;
    MPI_Win_allocate(size, 1, info, comm, &mem, &win);
    return {comm, win, mem, get_rank(comm)};
  }

  // Allocates a shared window
  static MpiWindow allocate_shared(MPI_Aint size, MPI_Comm comm, int mem_rank = -1) {
    // Only allocate memory on local_mem_rank
    if (mem_rank != -1 && get_rank(comm) != mem_rank) {
      size = 0;
    }

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "accumulate_ordering", "none");
    MPI_Info_set(info, "same_disp_unit", "true");
    MPI_Win win;
    void *mem;
    MPI_Win_allocate_shared(size, 1, info, comm, &mem, &win);

    if (mem_rank != -1) {
      MPI_Aint mem_rank_size;  // Ignore
      int mem_rank_disp_unit;  // Ignore
      // All ranks use the memory of mem_rank
      MPI_Win_shared_query(win, mem_rank, &mem_rank_size, &mem_rank_disp_unit, &mem);
    }

    return {comm, win, mem, get_rank(comm)};
  }

  // Creates a view on the supplied window
  static MpiWindow view(const MpiWindow &window) {
    return view(window.comm, window.win, window.disp);
  }

  // Creates a view on the supplied window
  static MpiWindow view(MPI_Comm comm, MPI_Win win, MPI_Aint disp = 0) {
    return {comm, win, get_win_base(win, disp), get_rank(comm), MPI_WIN_NULL, disp, false};
  }

  // Allocates a window where all processes on the same node point to the same memory
  static MpiWindow allocate_per_node(
      MPI_Aint size, MPI_Comm global_comm, MPI_Comm local_comm, int local_mem_rank = 0) {
    // Only allocate memory on local_mem_rank
    if (get_rank(local_comm) != local_mem_rank) size = 0;

    MPI_Win shared_win = allocate_shared_win(local_comm, size);

    // Convert local_mem_rank to rank in global_comm
    int mem_rank = get_rank(global_comm);
    MPI_Bcast(&mem_rank, 1, MPI_INT, local_mem_rank, local_comm);

    MPI_Aint local_size;  // Ignore
    int local_disp_unit;  // Ignore
    void *mem;  // All ranks use the memory of local_mem_rank
    MPI_Win_shared_query(shared_win, local_mem_rank, &local_size, &local_disp_unit, &mem);

    return {global_comm, create_win(mem, size, global_comm), mem, mem_rank, shared_win};
  }
};
