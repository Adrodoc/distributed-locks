#include <libdash.h>

#include "benchmarks.cpp"
#include "lock/baseline/DMcsLock.cpp"
#include "lock/baseline/DashLock.cpp"
#include "lock/baseline/MpiWinLock.cpp"
#include "lock/baseline/RmaMcsLock.cpp"
#include "lock/clh/ClhLock.cpp"
#include "lock/clh/ClhLockAtomic.cpp"
#include "lock/clh/ClhLockAtomicWithCohortDetection.cpp"
#include "lock/clh/ClhLockNuma.cpp"
#include "lock/cohort/CohortLock.cpp"
#include "lock/cohort/CohortLockDirectCounter.cpp"
#include "lock/cohort/CohortLockInlineCounter.cpp"
#include "lock/cohort/CohortLockLocalCounter.cpp"
#include "lock/hem/HemLockAtomic.cpp"
#include "lock/hem/HemLockCtrAhAtomic.cpp"
#include "lock/hem/HemLockCtrAhAtomicWithCohortDetection.cpp"
#include "lock/hem/HemLockCtrAtomic.cpp"
#include "lock/hem/HemLockCtrOverlapAtomic.cpp"
#include "lock/hem/HemLockOverlapAtomic.cpp"
#include "lock/mcs/McsLock.cpp"
#include "lock/mcs/McsLockAccumulate.cpp"
#include "lock/mcs/McsLockAtomic.cpp"
#include "lock/mcs/McsLockAtomicWithCohortDetection.cpp"
#include "lock/mcs/McsLockMpi.cpp"
#include "lock/mcs/McsLockWithCohortDetection.cpp"
#include "lock/mcs/McsLockWithTasStealing.cpp"
#include "lock/mcs/McsLockWithTtsStealing.cpp"
#include "lock/mcs/p2p/McsLockDashStyle.cpp"
#include "lock/mcs/p2p/McsLockDashStyleCheckNextInRelease.cpp"
#include "lock/mcs/p2p/McsLockDashStyleDirectSpinning.cpp"
#include "lock/mcs/p2p/McsLockMpiTwoSided.cpp"
#include "lock/mcs/p2p/McsLockTwoSided.cpp"
#include "lock/mcs/p2p/McsLockTwoSidedAtomic.cpp"
#include "lock/mcs/p2p/McsLockTwoSidedSync.cpp"
#include "lock/mcs/p2p/McsLockTwoSidedSync2.cpp"
#include "lock/mcs/p2p/McsLockTwoSidedWithCohortDetection.cpp"
#include "lock/rh/RhLock.cpp"
// #include "lock/rh/RhLockAtomicRef.cpp"
#include "lock/shfl/MyShuffleLock.cpp"
#include "lock/shfl/ShflLock.cpp"
#include "lock/shfl/ShflLockGlobalTas.cpp"
#include "lock/tas/TasLock.cpp"
#include "lock/tas/TasLockAtomic.cpp"
#include "lock/tas/TasLockAtomicWithCohortDetection.cpp"
#include "lock/tas/TasLockBo.cpp"
#include "lock/tas/TasLockBoAtomicWithCohortDetection.cpp"
#include "lock/tas/TasLockCas.cpp"
#include "lock/tas/TasLockCasBo.cpp"
#include "lock/tkt/TktLock.cpp"
#include "lock/tkt/TktLockAtomic.cpp"
#include "lock/tkt/TktLockAtomicWithCohortDetection.cpp"
#include "lock/tkt/TktLockBoAtomic.cpp"
#include "lock/tts/TtsLock.cpp"
#include "lock/tts/TtsLockAtomicWithCohortDetection.cpp"
#include "lock/tts/TtsLockBo.cpp"
#include "lock/tts/TtsLockBoAtomicWithCohortDetection.cpp"
#include "lock/tts/TtsLockCas.cpp"
#include "lock/tts/TtsLockCasBo.cpp"
#include "log.cpp"

std::string get_mpi_memory_model() {
  void *mem;
  MPI_Win win;
  MPI_Win_allocate(1, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &mem, &win);
  int *memory_model;
  int flag;
  MPI_Win_get_attr(win, MPI_WIN_MODEL, &memory_model, &flag);
  MPI_Win_free(&win);
  if (!flag) {
    return "UNKNOWN";
  }
  switch (*memory_model) {
    case MPI_WIN_SEPARATE: return "MPI_WIN_SEPARATE";
    case MPI_WIN_UNIFIED: return "MPI_WIN_UNIFIED";
    default: return "UNKNOWN";
  }
}

void print_processor() {
  char processor_name_array[MPI_MAX_PROCESSOR_NAME];
  int processor_name_len;
  MPI_Get_processor_name(processor_name_array, &processor_name_len);
  std::string processor_name = std::string{processor_name_array, (std::size_t)processor_name_len};
  log() << "Running on processor: " << processor_name << std::endl;
}

template <class C, class G, class L>
void run_cohort_lock_benchmarks(
    const std::function<G()> &new_global_lock, const std::function<L()> &new_local_lock) {
  run_mpi_lock_benchmarks<C>([=] { return C{new_global_lock(), new_local_lock()}; });
}

template <class G, class L>
void run_cohort_locks_benchmarks(
    const std::function<G()> &new_global_lock, const std::function<L()> &new_local_lock) {
  run_cohort_lock_benchmarks<CohortLock<G, L>, G, L>(new_global_lock, new_local_lock);
  run_cohort_lock_benchmarks<CohortLockLocalCounter<G, L>, G, L>(new_global_lock, new_local_lock);
  run_cohort_lock_benchmarks<CohortLockDirectCounter<G, L>, G, L>(new_global_lock, new_local_lock);
  run_cohort_lock_benchmarks<CohortLockInlineCounter<G, L>, G, L>(new_global_lock, new_local_lock);
}

int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);
  mpi_lock_bench::initialize(argc, argv);
  dash::init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // freopen(("err/" + std::to_string(rank) + ".err").c_str(), "w", stderr);
  auto mpi_memory_model = get_mpi_memory_model();
  if (rank == 0) log() << "MPI memory model: " << mpi_memory_model << std::endl;
  print_processor();
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Comm global_comm = MPI_COMM_WORLD;
  MPI_Comm local_comm = split_comm_shared(global_comm);

  run_mpi_lock_benchmarks<ClhLock>();
  run_mpi_lock_benchmarks<ClhLockAtomic>();
  run_mpi_lock_benchmarks<ClhLockAtomicWithCohortDetection>();
  run_mpi_lock_benchmarks<ClhLockNuma>();

  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLock::per_node(global_comm, local_comm); }},
      std::function{[=] { return McsLockWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLock::per_node(global_comm, local_comm); }},
      std::function{[=] { return McsLockAtomicWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLockTwoSided::per_node(global_comm, local_comm); }},
      std::function{[=] { return McsLockAtomicWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TktLock{global_comm}; }},
      std::function{[=] { return McsLockAtomicWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TasLockCasBo{global_comm}; }},
      std::function{[=] { return McsLockAtomicWithCohortDetection{local_comm}; }});

  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLockTwoSided::per_node(global_comm, local_comm); }},
      std::function{[=] { return TktLockAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TktLock{global_comm}; }},
      std::function{[=] { return TktLockAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TasLockCasBo{global_comm}; }},
      std::function{[=] { return TktLockAtomic{local_comm}; }});

  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLockTwoSided::per_node(global_comm, local_comm); }},
      std::function{[=] { return TktLockAtomicWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TktLock{global_comm}; }},
      std::function{[=] { return TktLockAtomicWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TasLockCasBo{global_comm}; }},
      std::function{[=] { return TktLockAtomicWithCohortDetection{local_comm}; }});

  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLockTwoSided::per_node(global_comm, local_comm); }},
      std::function{[=] { return TktLockBoAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TktLock{global_comm}; }},
      std::function{[=] { return TktLockBoAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TasLockCasBo{global_comm}; }},
      std::function{[=] { return TktLockBoAtomic{local_comm}; }});

  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLockTwoSided::per_node(global_comm, local_comm); }},
      std::function{[=] { return TasLockBoAtomicWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TktLock{global_comm}; }},
      std::function{[=] { return TasLockBoAtomicWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TasLockCasBo{global_comm}; }},
      std::function{[=] { return TasLockBoAtomicWithCohortDetection{local_comm}; }});

  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLockTwoSided::per_node(global_comm, local_comm); }},
      std::function{[=] { return TtsLockBoAtomicWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TktLock{global_comm}; }},
      std::function{[=] { return TtsLockBoAtomicWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TasLockCasBo{global_comm}; }},
      std::function{[=] { return TtsLockBoAtomicWithCohortDetection{local_comm}; }});

  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLockTwoSided::per_node(global_comm, local_comm); }},
      std::function{[=] { return ClhLockAtomicWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TktLock{global_comm}; }},
      std::function{[=] { return ClhLockAtomicWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TasLockCasBo{global_comm}; }},
      std::function{[=] { return ClhLockAtomicWithCohortDetection{local_comm}; }});

  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLockTwoSided::per_node(global_comm, local_comm); }},
      std::function{[=] { return HemLockAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TktLock{global_comm}; }},
      std::function{[=] { return HemLockAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TasLockCasBo{global_comm}; }},
      std::function{[=] { return HemLockAtomic{local_comm}; }});

  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLockTwoSided::per_node(global_comm, local_comm); }},
      std::function{[=] { return HemLockCtrAhAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TktLock{global_comm}; }},
      std::function{[=] { return HemLockCtrAhAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TasLockCasBo{global_comm}; }},
      std::function{[=] { return HemLockCtrAhAtomic{local_comm}; }});

  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLockTwoSided::per_node(global_comm, local_comm); }},
      std::function{[=] { return HemLockCtrAhAtomicWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TktLock{global_comm}; }},
      std::function{[=] { return HemLockCtrAhAtomicWithCohortDetection{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TasLockCasBo{global_comm}; }},
      std::function{[=] { return HemLockCtrAhAtomicWithCohortDetection{local_comm}; }});

  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLockTwoSided::per_node(global_comm, local_comm); }},
      std::function{[=] { return HemLockCtrAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TktLock{global_comm}; }},
      std::function{[=] { return HemLockCtrAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TasLockCasBo{global_comm}; }},
      std::function{[=] { return HemLockCtrAtomic{local_comm}; }});

  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLockTwoSided::per_node(global_comm, local_comm); }},
      std::function{[=] { return HemLockCtrOverlapAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TktLock{global_comm}; }},
      std::function{[=] { return HemLockCtrOverlapAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TasLockCasBo{global_comm}; }},
      std::function{[=] { return HemLockCtrOverlapAtomic{local_comm}; }});

  run_cohort_locks_benchmarks(
      std::function{[=] { return McsLockTwoSided::per_node(global_comm, local_comm); }},
      std::function{[=] { return HemLockOverlapAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TktLock{global_comm}; }},
      std::function{[=] { return HemLockOverlapAtomic{local_comm}; }});
  run_cohort_locks_benchmarks(
      std::function{[=] { return TasLockCasBo{global_comm}; }},
      std::function{[=] { return HemLockOverlapAtomic{local_comm}; }});

  run_mpi_lock_benchmarks<DashLock>();
  run_mpi_lock_benchmarks<DMcsLock>();
  run_mpi_lock_benchmarks<McsLock>();
  run_mpi_lock_benchmarks<McsLockAccumulate>();
  run_mpi_lock_benchmarks<McsLockAtomic>();
  run_mpi_lock_benchmarks<McsLockDashStyle>();
  run_mpi_lock_benchmarks<McsLockDashStyleCheckNextInRelease>();
  run_mpi_lock_benchmarks<McsLockDashStyleDirectSpinning>();
  //   run_mpi_lock_benchmarks<McsLockMpi>();
  // run_mpi_lock_benchmarks<McsLockMpiTwoSided>();
  run_mpi_lock_benchmarks<McsLockTwoSided>();
  // run_mpi_lock_benchmarks<McsLockTwoSidedAtomic>();
  // run_mpi_lock_benchmarks<McsLockTwoSidedSync>();
  // run_mpi_lock_benchmarks<McsLockTwoSidedSync2>();
  // run_mpi_lock_benchmarks<McsLockTwoSidedWithCohortDetection>();
  // run_mpi_lock_benchmarks<McsLockAtomicWithCohortDetection>();
  // run_mpi_lock_benchmarks<McsLockWithCohortDetection>();
  run_mpi_lock_benchmarks<McsLockWithTasStealing>();
  run_mpi_lock_benchmarks<McsLockWithTtsStealing>();
  run_mpi_lock_benchmarks<MpiWinLock>();
  // run_mpi_lock_benchmarks<MyShuffleLock>();

  run_mpi_lock_benchmarks<RhLock>(
      std::function{[=] {
        return RhLock{global_comm, local_comm, RhLockConfig{1, 1, 16, 0, 0}};
      }},
      "RhLock-local_max=16");

  run_mpi_lock_benchmarks<RhLock>(
      std::function{[=] {
        return RhLock{global_comm, local_comm, RhLockConfig{1, 1, 32, 0, 0}};
      }},
      "RhLock-local_max=32");

  run_mpi_lock_benchmarks<RhLock>(
      std::function{[=] {
        return RhLock{global_comm, local_comm, RhLockConfig{2, 1, 32, 0, 0}};
      }},
      "RhLock-local_max=32-fair_factor=2");

  run_mpi_lock_benchmarks<RhLock>(
      std::function{[=] {
        return RhLock{global_comm, local_comm, RhLockConfig{100, 1, 32, 0, 0}};
      }},
      "RhLock-local_max=32-fair_factor=100");

  run_mpi_lock_benchmarks<RhLock>(
      std::function{[=] {
        return RhLock{global_comm, local_comm, RhLockConfig{1, 1, 64, 0, 0}};
      }},
      "RhLock-local_max=64");

  // run_mpi_lock_benchmarks<RhLockAtomicRef>();
  run_mpi_lock_benchmarks<RmaMcsLock>();
  run_mpi_lock_benchmarks<ShflLock>();
  run_mpi_lock_benchmarks<ShflLockGlobalTas>();
  run_mpi_lock_benchmarks<TasLock>();
  run_mpi_lock_benchmarks<TasLockAtomicWithCohortDetection>();
  run_mpi_lock_benchmarks<TasLockBo>();
  run_mpi_lock_benchmarks<TasLockBoAtomicWithCohortDetection>();
  run_mpi_lock_benchmarks<TasLockCas>();
  run_mpi_lock_benchmarks<TasLockCasBo>();
  run_mpi_lock_benchmarks<TktLock>();
  run_mpi_lock_benchmarks<TktLockAtomic>();
  run_mpi_lock_benchmarks<TtsLock>();
  run_mpi_lock_benchmarks<TtsLockAtomicWithCohortDetection>();
  run_mpi_lock_benchmarks<TtsLockBo>();
  run_mpi_lock_benchmarks<TtsLockBoAtomicWithCohortDetection>();
  run_mpi_lock_benchmarks<TtsLockCas>();
  run_mpi_lock_benchmarks<TtsLockCasBo>();

  dash::finalize();
  MPI_Finalize();
  return 0;
}
