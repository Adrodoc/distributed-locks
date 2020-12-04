from pathlib import Path
import json
import numpy as np
import pandas as pd


def read_benchmark_json(path):
    with open(path) as file:
        content = json.load(file)
    runs_df = pd.json_normalize(content['runs'])
    context_df = pd.json_normalize([content['context']])
    return runs_df.merge(context_df, how='cross')


plot_dir = Path(__file__).resolve().parent
reports_dir = plot_dir / '../reports'

dir = '9ef1c84-lrz-cm2_tiny-wbab-baseline-thesis'
# file = 'WBAB-3-DMcsLock-avg_wait_ns=5000-mpi_progress=1-processes=112-lrz-cm2_tiny-4-28-112.json'
file = 'WBAB-4-RmaMcsLock-avg_wait_ns=3000-mpi_progress=1-processes=112-lrz-cm2_tiny-4-28-112.json'

data = read_benchmark_json(reports_dir / dir / 'json' / file)

data['latency'] = data['duration_ns'] / data['iterations']
print('mean(latency)', data['latency'].mean())
print('mean(min)', data['iterations_per_process_min'].mean())
print('mean(max)', data['iterations_per_process_max'].mean())
print('mean(median)', data['iterations_per_process_median'].mean())
