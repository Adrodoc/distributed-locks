#include "NullReporter.cpp"
#include "benchmarks.cpp"
#include "lock/AdvancedMcsLock.cpp"
#include "lock/CohortLock.cpp"
#include "lock/DMcsLock.cpp"
#include "lock/DisableableTasLock.cpp"
#include "lock/DisableableTtsLock.cpp"
#include "lock/HMCSLock.cpp"
#include "lock/HybridLock.cpp"
#include "lock/McsLock.cpp"
#include "lock/McsLockAccumulate.cpp"
#include "lock/McsLockBcl.cpp"
#include "lock/McsLockFlushEveryOperation.cpp"
#include "lock/McsLockFlushLocal.cpp"
#include "lock/McsLockMpiFlushLocal.cpp"
#include "lock/McsLockMpiFlushLocalAll.cpp"
#include "lock/McsLockMpiRequest.cpp"
#include "lock/McsLockOwnWindow.cpp"
#include "lock/McsLockTwoSided.cpp"
#include "lock/McsLockTwoSidedSync.cpp"
#include "lock/McsLockTwoSidedSync2.cpp"
#include "lock/McsLockWithCohortDetection.cpp"
#include "lock/MpiWinLock.cpp"
#include "lock/MyShuffleLock.cpp"
#include "lock/ShflLock.cpp"
#include "lock/TasLock.cpp"
#include "lock/TasLockOwnWindow.cpp"
#include "lock/TasLockOwnWindowCas.cpp"
#include "lock/TktLock.cpp"
#include "lock/TtsLock.cpp"
#include "lock/TtsLockOwnWindow.cpp"
#include "lock/TtsLockOwnWindowCas.cpp"
#include "log.cpp"

std::string get_mpi_memory_model()
{
    int *memory_model;
    int flag;
    MPI_Win_get_attr(BCL::win, MPI_WIN_MODEL, &memory_model, &flag);
    if (!flag)
    {
        return "UNKNOWN";
    }
    switch (*memory_model)
    {
    case MPI_WIN_SEPARATE:
        return "MPI_WIN_SEPARATE";
    case MPI_WIN_UNIFIED:
        return "MPI_WIN_UNIFIED";
    default:
        return "UNKNOWN";
    }
}

void print_processor()
{
    char processor_name_array[MPI_MAX_PROCESSOR_NAME];
    int processor_name_len;
    MPI_Get_processor_name(processor_name_array, &processor_name_len);
    std::string processor_name = std::string{processor_name_array, (std::size_t)processor_name_len};
    log() << "Running on processor: " << processor_name << std::endl;
}

extern std::string FLAGS_benchmark_out;

int main(int argc, char *argv[])
{
    BCL::init();
    // MPI_Init(NULL, NULL);
    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // BCL::my_rank = rank;
    // int size;
    // MPI_Comm_size(MPI_COMM_WORLD, &size);
    // BCL::my_nprocs = size;
    // freopen(("err/" + std::to_string(BCL::rank()) + ".err").c_str(), "w", stderr);
    if (BCL::rank() == 0)
        log() << "MPI memory model: " << get_mpi_memory_model() << std::endl;
    print_processor();
    MPI_Barrier(MPI_COMM_WORLD);

    // REGISTER_LOCK_BENCHMARKS(AdvancedMcsLock);
    typedef CohortLock<TktLock, McsLockWithCohortDetection> CTktMcsLock;
    REGISTER_LOCK_BENCHMARKS(CTktMcsLock);
    // REGISTER_LOCK_BENCHMARKS(DisableableTasLock);
    // REGISTER_LOCK_BENCHMARKS(DisableableTtsLock);
    // REGISTER_LOCK_BENCHMARKS(DMcsLock);
    // REGISTER_LOCK_BENCHMARKS(HMCSLock);
    // REGISTER_LOCK_BENCHMARKS(HybridLock);
    REGISTER_LOCK_BENCHMARKS(McsLock);
    // REGISTER_LOCK_BENCHMARKS(McsLockAccumulate);
    // REGISTER_LOCK_BENCHMARKS(McsLockBcl);
    // REGISTER_LOCK_BENCHMARKS(McsLockFlushEveryOperation);
    // REGISTER_LOCK_BENCHMARKS(McsLockFlushLocal);
    // REGISTER_LOCK_BENCHMARKS(McsLockMpiFlushLocal);
    // REGISTER_LOCK_BENCHMARKS(McsLockMpiFlushLocalAll);
    // REGISTER_LOCK_BENCHMARKS(McsLockMpiRequest);
    // REGISTER_LOCK_BENCHMARKS(McsLockOwnWindow);
    // REGISTER_LOCK_BENCHMARKS(McsLockTwoSided);
    // REGISTER_LOCK_BENCHMARKS(McsLockTwoSidedSync);
    // REGISTER_LOCK_BENCHMARKS(McsLockTwoSidedSync2);
    // REGISTER_LOCK_BENCHMARKS(McsLockWithCohortDetection);
    // REGISTER_LOCK_BENCHMARKS(MpiWinLock);
    // REGISTER_LOCK_BENCHMARKS(MyShuffleLock);
    // REGISTER_LOCK_BENCHMARKS(ShflLock);
    // REGISTER_LOCK_BENCHMARKS(TasLock);
    // REGISTER_LOCK_BENCHMARKS(TasLockOwnWindow);
    // REGISTER_LOCK_BENCHMARKS(TasLockOwnWindowCas);
    REGISTER_LOCK_BENCHMARKS(TktLock);
    // REGISTER_LOCK_BENCHMARKS(TtsLock);
    // REGISTER_LOCK_BENCHMARKS(TtsLockOwnWindow);
    // REGISTER_LOCK_BENCHMARKS(TtsLockOwnWindowCas);

    benchmark::Initialize(&argc, argv);
    if (BCL::rank() == 0)
        benchmark::RunSpecifiedBenchmarks();
    else
    {
        // Removing this command line flag to avoid recreating the file when it is already in use.
        FLAGS_benchmark_out = "";
        NullReporter null;
        benchmark::RunSpecifiedBenchmarks(&null);
    }
    // MPI_Finalize();
    BCL::finalize();
    return 0;
}
