#pragma once

#include "mpi_type_provider.cpp"

class Window {
  static MPI_Win allocate_win(const MPI_Aint size, const MPI_Comm comm) {
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "accumulate_ordering", "none");
    MPI_Info_set(info, "same_disp_unit", "true");
    void *mem;
    MPI_Win win;
    MPI_Win_allocate(size, 1, info, comm, &mem, &win);
    return win;
  }

 public:
  const MPI_Win win;
  Window(const Window &) = delete;
  Window(const MPI_Aint size, const MPI_Comm comm) : win{allocate_win(size, comm)} {}

  ~Window() {
    MPI_Win lwin = win;
    MPI_Win_free(&lwin);
  }

  inline void lock_all() const { MPI_Win_lock_all(0, win); }

  inline void unlock_all() const { MPI_Win_unlock_all(win); }

  inline void flush_all() const { MPI_Win_flush_all(win); }

#define USE_FAO
  //#define USE_REQUEST_BASED

  template <typename T>
  inline T get(const int target_rank, const MPI_Aint target_disp) const {
    T result;
    MPI_Datatype type = get_mpi_type<T>();
#ifndef USE_REQUEST_BASED
    MPI_Get(&result, 1, type, target_rank, target_disp, 1, type, win);
    MPI_Win_flush_local(target_rank, win);
#else
    MPI_Request request;
    MPI_Rget(&result, 1, type, target_rank, target_disp, 1, type, win, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
#endif
    return result;
  }

  template <typename T>
  inline void set(const int target_rank, const MPI_Aint target_disp, const T &value) const {
    MPI_Datatype type = get_mpi_type<T>();
#ifndef USE_REQUEST_BASED
    MPI_Put(&value, 1, type, target_rank, target_disp, 1, type, win);
    MPI_Win_flush_local(target_rank, win);
#else
    MPI_Request request;
    MPI_Rput(&value, 1, type, target_rank, target_disp, 1, type, win, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
#endif
  }

  template <typename T>
  inline T atomic_get(const int target_rank, const MPI_Aint target_disp) const {
    MPI_Datatype type = get_mpi_type<T>();
    T dummy;
    T result;
#ifndef USE_REQUEST_BASED
#ifdef USE_FAO
    MPI_Fetch_and_op(&dummy, &result, type, target_rank, target_disp, MPI_NO_OP, win);
#else
    MPI_Get_accumulate(
        &dummy, 1, type, &result, 1, type, target_rank, target_disp, 1, type, MPI_NO_OP, win);
#endif
    MPI_Win_flush_local(target_rank, win);
#else
    MPI_Request request;
    MPI_Rget_accumulate(
        &dummy, 1, type, &result, 1, type, target_rank, target_disp, 1, type, MPI_NO_OP, win,
        &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
#endif
    return result;
  }

  template <typename T>
  inline void atomic_set(const int target_rank, const MPI_Aint target_disp, const T &value) const {
    MPI_Datatype type = get_mpi_type<T>();
#ifndef USE_REQUEST_BASED
    MPI_Accumulate(&value, 1, type, target_rank, target_disp, 1, type, MPI_REPLACE, win);
    MPI_Win_flush_local(target_rank, win);
#else
    MPI_Request request;
    MPI_Raccumulate(&value, 1, type, target_rank, target_disp, 1, type, MPI_REPLACE, win, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
#endif
  }

  template <typename T>
  inline T swap(const int target_rank, const MPI_Aint target_disp, const T &value) const {
    MPI_Datatype type = get_mpi_type<T>();
    T result;
#ifndef USE_REQUEST_BASED
#ifdef USE_FAO
    MPI_Fetch_and_op(&value, &result, type, target_rank, target_disp, MPI_REPLACE, win);
#else
    MPI_Get_accumulate(
        &value, 1, type, &result, 1, type, target_rank, target_disp, 1, type, MPI_REPLACE, win);
#endif
    MPI_Win_flush_local(target_rank, win);
#else
    MPI_Request request;
    MPI_Rget_accumulate(
        &value, 1, type, &result, 1, type, target_rank, target_disp, 1, type, MPI_REPLACE, win,
        &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
#endif
    return result;
  }

  template <typename T>
  inline bool compare_and_swap(
      const int target_rank, const MPI_Aint target_disp, const T &old_value,
      const T &new_value) const {
    MPI_Datatype type = get_mpi_type<T>();
    T result;
    MPI_Compare_and_swap(&new_value, &old_value, &result, type, target_rank, target_disp, win);
    MPI_Win_flush_local(target_rank, win);
    return result == old_value;
  }
};
