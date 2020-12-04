# Developing High Performance RMA Locks by Porting and Optimizing NUMA-aware Algorithms

* This repository contains the code used in my [master thesis](https://github.com/Adrodoc/Masterarbeit).
* The main file is [main/src/benchmark/main.cpp](main/src/benchmark/main.cpp).
* The uncontested performance benchmark (UPB) can be found [here](main/src/benchmark/mpi_lock_bench.cpp#L289-L371).
* The other benchmarks can be found in [main/src/benchmark/benchmarks.cpp](main/src/benchmark/benchmarks.cpp)
* The locks can be found in [main/src/lock](main/src/lock).

# Setup

## Install prerequisites

### On Ubuntu
```bash
sudo apt install cmake
# MPICH is recommended, but other MPI implementations might work too. Open-MPI does not work properly at the time of this writing.
sudo apt install mpich
sudo apt install python3-venv
sudo apt install g++-10
sudo update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-10 100
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-10 100
```

### At LRZ

```bash
module load cmake/3.16.5
module unload intel-mpi
module switch intel intel/20.0.2
module load intel-mpi/2019.10.317-intel
# If you want to use OpenMPI:
# module switch intel-mpi openmpi/4.0.4-intel19
```

### Install Dash
```bash
git clone --recursive https://github.com/dash-project/dash.git
cd dash
git switch --detach 99f299572bbb42f6668fbe704b179b9b5d0da92b
cmake -E make_directory build
cmake -E chdir build cmake -DDART_IMPLEMENTATIONS=mpi -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DENABLE_HWLOC=OFF -DENABLE_LIBNUMA=OFF ..
cmake --build build --target install
cd ..
```

## Setup project
```bash
# Check out the repository:
git clone https://github.com/Adrodoc/distributed-locks.git
# Go to the root directory:
cd distributed-locks
# Make a build directory to place the build output:
cmake -E make_directory "build"
# Generate build system files with cmake:
cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release ..
# or, starting with CMake 3.13, use a simpler form:
# cmake -DCMAKE_BUILD_TYPE=Release -S . -B "build"
```

# Build
```bash
cmake --build build --config Release
```

# Run

## Locally

### All benchmarks
```bash
# When using Open-MPI (see https://github.com/open-mpi/ompi/issues/2080):
# export OMPI_MCA_osc=pt2pt
./locally-run-benchmarks.sh --include_locks='McsLock|RmaMcsLock'
```

### A specific configuration
```bash
# When using Open-MPI (see https://github.com/open-mpi/ompi/issues/2080):
# export OMPI_MCA_osc=pt2pt
mkdir benchmarks
mpirun -n 4 build/main/main --out=benchmarks
```

## At LRZ

### All benchmarks
```bash
./sbatch-run-benchmarks.sh
```

### A specific configuration
```bash
mkdir benchmarks
salloc --ntasks=4 --partition=cm2_inter
mpiexec build/main/main --out=benchmarks
exit
```
