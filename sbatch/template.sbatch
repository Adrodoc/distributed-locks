#!/bin/bash
#SBATCH -J dist-locks
#SBATCH -o sbatch/out/%x.%j.%N.out
#SBATCH --clusters=$clusters
#SBATCH --partition=$partition
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$ntasks-per-node
#SBATCH --exclusive
#SBATCH --export=NONE
#SBATCH --time=00:10:00
module load slurm_setup
module switch intel-mpi intel-mpi/2019.10.317-intel

json_dir=reports/$commit-lrz-$partition/json
mkdir -p $json_dir

mpiexec -n $SLURM_NTASKS build/main/main \
    --out=$json_dir \
    --file_suffix=-lrz-$partition-$nodes-$ntasks-per-node-$SLURM_NTASKS \
    --repetitions=8 $args
