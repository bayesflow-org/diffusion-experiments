import os
import logging

import pickle
import numpy as np
import pandas as pd
import itertools
from pathlib import Path
from scipy.stats import median_abs_deviation as mad
import sbibm

from case_study1.model_settings_benchmark import MODELS, SAMPLER_SETTINGS, is_compatible

import warnings
warnings.filterwarnings(
    "ignore",
    message="JULIA_SYSIMAGE_DIFFEQTORCH not set"
)

def model_sampler_key(_model: str, _sampler: str) -> str:
    return f"{_model}-{_sampler}"

benchmarks = list(
    (m, t, s) for s, m, t in itertools.product(
    SAMPLER_SETTINGS, MODELS.keys(), sbibm.get_available_tasks()
) if is_compatible(m, s)
)
BASE = Path(__file__).resolve().parent
lustre_metrics_dir = Path("/lustre/scratch/data/jarruda_hpc-diffusion_experiments/case_study1/metrics")
metrics_dir = lustre_metrics_dir if lustre_metrics_dir.exists() else BASE / "metrics"

model_names = list(MODELS.keys())
sampler_names = list(SAMPLER_SETTINGS.keys())
tasks = sbibm.get_available_tasks()
task_to_idx = {t: i for i, t in enumerate(tasks)}
n_tasks = len(sbibm.get_available_tasks())
keys = [model_sampler_key(m, s) for m, _, s in benchmarks]
def nan_matrix():
    return {k: np.full(n_tasks, np.nan, dtype=float) for k in keys}

results_mean = nan_matrix()
results_std = nan_matrix()
results_median = nan_matrix()
results_mad = nan_matrix()
times = nan_matrix()
times_std = nan_matrix()

for model_name, task_name, sampler_name in benchmarks:
    model_name = model_name
    pkl = metrics_dir / f'c2st_results_{model_name}_{task_name}_{sampler_name}.pkl'
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            c2st_results = pickle.load(f)
    else:
        logging.warning(f"Missing results for {model_name} on {task_name} with {sampler_name}, skipping.")
        continue

    task_i = task_to_idx.get(task_name)
    if task_i is None:
        logging.warning("Unknown task %s; skipping.", task_name)
        continue

    key = model_sampler_key(model_name, sampler_name)
    results_mean[key][task_i] = np.mean([max(v['c2st'], 1-v['c2st']) for v in c2st_results])
    results_std[key][task_i] = np.std([max(v['c2st'], 1-v['c2st']) for v in c2st_results])
    results_median[key][task_i] = np.median([max(v['c2st'], 1 - v['c2st']) for v in c2st_results])
    results_mad[key][task_i] = mad([max(v['c2st'], 1 - v['c2st']) for v in c2st_results])
    times[key][task_i] = np.mean([v['time'] for v in c2st_results])
    times_std[key][task_i] = np.std([v['time'] for v in c2st_results])

# Build long format dataframe
rows = []
duplicates = []
for key in keys:
    if key in duplicates:
        continue
    model = key.split("-")[0]
    rest = key.split("-")[1:]
    sampler = "-".join(rest)
    duplicates.append(key)
    for task_idx, task_name in enumerate(tasks):
        rows.append({
            'model': model,
            'sampler': sampler,
            'subnet': 'time_mlp' if 'transformer' not  in model else 'time_transformer',
            'task': task_name,
            'c2st': results_mean[key][task_idx],
            'c2st_std': results_std[key][task_idx],
            'c2st_median': results_median[key][task_idx],
            'c2st_mad': results_mad[key][task_idx],
            'time': times[key][task_idx],
            'time_std': times_std[key][task_idx]
        })

df = pd.DataFrame(rows)
df.to_csv(BASE / "plots" / "c2st_benchmark_results.csv", index=False)
