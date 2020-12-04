#!/bin/bash

IFS='|' include_locks="$*"

if [ "$include_locks" == "" ]
then
  echo "Usage: $0 <lock-regex> [<lock-regex> ...]"
  exit 1
fi

git pull
commit=$(git rev-parse --short HEAD)
echo Commit: $commit
cmake --build build --config Release

mkdir -p sbatch/out
mkdir -p sbatch/tmp
rm -f sbatch/tmp/*

clusters=cm2_tiny
partition=cm2_tiny

submit_job () {
  local file_name="sbatch/tmp/$commit-$partition-$nodes-$tasks-include_benchmarks=$include_benchmarks-exclude_locks=$exclude_locks-include_locks=$include_locks.sbatch"
  sed "
  s/\$commit/$commit/g
  s/\$clusters/$clusters/g
  s/\$partition/$partition/g
  s/\$nodes/$nodes/g
  s/\$ntasks-per-node/$tasks/g
  s/\$args/--include_benchmarks='$include_benchmarks' --exclude_locks='$exclude_locks' --include_locks='$include_locks'/g
  " sbatch/template.sbatch > "$file_name"
  sbatch "$file_name"
}

# UPB:
nodes=2
tasks=2
include_benchmarks=UPB
exclude_locks='McsLockAtomic|McsLockTwoSidedAtomic'
submit_job

# nodes=1
# tasks=2
# args="--include_benchmarks=UPB --include_locks='McsLockAtomic|McsLockTwoSidedAtomic'"
# submit_job

# ECSB:
for i in 1,14 1,28 2,15 2,28 3,19 3,28 4,22 4,28
do
  nodes=${i%,*}
  tasks=${i#*,}
  include_benchmarks=ECSB
  exclude_locks=
  submit_job
done

# CCWB & WBAB:
nodes=1
tasks=28
include_benchmarks='CCWB|WBAB'
exclude_locks=
submit_job

nodes=4
tasks=28
include_benchmarks='CCWB|WBAB'
exclude_locks='McsLockAtomic|McsLockTwoSidedAtomic'
submit_job
