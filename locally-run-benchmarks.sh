#!/bin/bash

commit=$(git rev-parse --short HEAD)

declare -a arr=(2 3 4 5 6 7)
for n in "${arr[@]}"
do
    mkdir -p reports/$commit/json
    mpirun -n $n build/main/main --benchmark_repetitions=8 --benchmark_display_aggregates_only=true --benchmark_out=reports/$commit/json/$n.json
done

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
