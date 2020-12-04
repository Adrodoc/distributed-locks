#!/bin/bash

trap 'exit' INT
trap 'exit' ERR

commit=$(git rev-parse --short HEAD)
mkdir -p reports/$commit/json

# ECSB:
declare -a arr=(2 4 6 8)
for n in "${arr[@]}"
do
    mpirun -n $n build/main/main \
        --out=reports/$commit/json \
        --include_benchmarks=ECSB \
        "$@"
done

# CCWB & WBAB:
mpirun -n 6 build/main/main \
    --out=reports/$commit/json \
    --include_benchmarks='CCWB|WBAB' \
    "$@"

cd plot
if [ ! -d .venv ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

echo Plotting benchmarks
python plot.py
