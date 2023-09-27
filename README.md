This repository has moved to https://codeberg.org/Adrodoc/distributed-locks to avoid GitHubs two factor authentication (2FA) requirement. We believe that Microsofts decision to force all code contributors to use 2FA is very problematic for the following reasons:

1. 2FA significantly increases the risk of irreversible account loss. This is very different to 2FA for something like online banking where in the worst case you can contact your bank and verify your identity to regain access. With GitHub however, if you loose your phone and backup codes (both of which is possible), you will never gain access to your account again.
2. The decision to require 2FA for every code contributor seems very needless. Yes software supply chain attacks are a thing, but not every code contributor on GitHub is responsible for widely used libraries. It's quite the opposite: most code contributors are not responsible for widely used libraries and their code is reviewed and merged by those that are. Also, the details of the 2FA requirement seem arbitrary. Why for example is email not accepted as a second factor or why can WebAuth only be a second second factor and not a primary second factor? Just to make it really hard to not use a phone for 2FA? It feels like a "trust us, we know what's good for you" attitude from Microsoft and it is scary to think what arbitrary decision could come next.
3. Depending on how you use passwords the account security is not necessary improved that much by using 2FA, especially if it is forced onto people that don't want to use it. So why is there no opt out?
4. Many other developers publicly stated that they are leaving GitHub because of this, so staying on GitHub would prevent any code contributions from these people. This makes finding good contributors even harder than before. By moving to https://codeberg.org everyone can continue to contribute to this project.
5. Unfortunately Microsoft does not allow mail as a second factor and some companies do not allow you to bring your private phone to work or install proprietary software (such authenticators) for security reasons. This means 2FA can actually completely prevent you from logging into the website in some circumstances. This is really sad, because it can make it harder for professional developers at companies that use free and open source software to return something to the community.
6. Not everyone owns/can afford a smartphone or dedicated authenticator hardware and Microsoft makes it very inconvenient to use 2FA without that by requiring you to install authenticator software on every development machine. This discourages code contributions from poor people.

2FA is a good technology, but it should be up to repository owners to decide whether it is appropriate for the project at hand. Requiring 2FA for all code contributions, even for code that is reviewed and merged by other people, is completely unnecessary and discourages contributions.

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
