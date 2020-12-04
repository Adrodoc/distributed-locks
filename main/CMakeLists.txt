OPTION(STATS "Gather additional lock specific statistics" ON)
IF(STATS)
    ADD_DEFINITIONS(-DSTATS)
ENDIF(STATS)

# add_executable(main src/main.cpp)
add_executable(main src/benchmark/main.cpp)
target_include_directories(main PUBLIC src)

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
target_link_libraries(main MPI::MPI_CXX)

include_directories($ENV{HOME}/opt/include)

find_library(DASH dash-mpi $ENV{HOME}/opt/lib)
message(STATUS ${DASH})
target_link_libraries(main ${DASH})

find_library(DART_MPI dart-mpi $ENV{HOME}/opt/lib)
message(STATUS ${DART_MPI})
target_link_libraries(main ${DART_MPI})

find_library(DART_BASE dart-base $ENV{HOME}/opt/lib)
message(STATUS ${DART_BASE})
target_link_libraries(main ${DART_BASE})
