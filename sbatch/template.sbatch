#!/bin/bash
#SBATCH -J dist-locks
#SBATCH -o sbatch/out/%x.%j.%N.out
#SBATCH --clusters=$clusters
#SBATCH --partition=$partition
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$ntasks-per-node
#SBATCH --export=NONE
#SBATCH --time=00:20:00
module load slurm_setup
module switch intel-mpi openmpi/4.0.4-intel19

json_dir=reports/$commit-lrz-$partition/json
mkdir -p $json_dir
mpiexec -n $SLURM_NTASKS build/main/main --benchmark_repetitions=16 --benchmark_out=$json_dir/lrz-$partition-$nodes-$ntasks-per-node-$SLURM_NTASKS.json --benchmark_display_aggregates_only=true
