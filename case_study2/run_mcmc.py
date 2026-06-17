import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # jax might deadlock joblib here
# Redirect PyTensor compile cache to local /tmp to avoid NFS stale file handle errors on HPC
_pytensor_compiledir = f"/tmp/pytensor_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
os.environ.setdefault("PYTENSOR_FLAGS", f"compiledir={_pytensor_compiledir}")

import sys
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Union
import time

import pypesto.optimize as optimize
import pypesto.result
import pypesto.sample as sample
import amici
import fides

from case_study2.helper_pypesto import load_problem, create_pypesto_problem
from case_study2.pymc_sampler import PymcSampler2

logging.basicConfig(level=logging.INFO)
logging.getLogger("pypesto").setLevel(logging.ERROR)
amici.swig_wrappers.logger.setLevel(logging.CRITICAL)

n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 10))
dataset_idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
BASE = Path(__file__).resolve().parent

logging.info(f'AMICI OpenMP {amici.compiledWithOpenMP()}')  # True
logging.info(f"SLURM_CPUS_PER_TASK={n_cpus}")
logging.info(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")  # 4

problem_name = "Beer_MolBioSystems2014"
mcmc_type = 'NUTS'
n_mcmc_samples = 1_000
n_final_samples = 1_000
n_chains = n_cpus // 4  # 4 cpus per AMICI simulation
n_tune = 1_000  # default
n_starts = 20

lustre_models_dir = Path("/lustre/scratch/data/jarruda_hpc-diffusion_experiments/case_study2/models")
models_dir = lustre_models_dir if lustre_models_dir.exists() else BASE / "models"

validation_data_path = models_dir / f'validation_data_petab_{problem_name}.pkl'
result_path = models_dir / f'mcmc_samples_{problem_name}_dataset{dataset_idx}.pkl'


def run_mcmc(
    petab_problem,
    data_df=None,
    verbose: bool = True,
) -> Union[pypesto.result.Result, tuple]:
    if data_df is None:
        _pypesto_problem = create_pypesto_problem(petab_problem)
        _petab_problem = None
    else:
        _measurement_df = data_df
        if 'measurement' not in _measurement_df.columns:
            _measurement_df['measurement'] = _measurement_df['simulation']
        _pypesto_problem, _petab_problem = create_pypesto_problem(petab_problem, _measurement_df)

    if n_starts == 0:
        _result = None
    else:
        logging.info(f"Starting optimization with {n_starts} starts.")
        _result = optimize.minimize(
            problem=_pypesto_problem,
            optimizer=optimize.FidesOptimizer(
                verbose=0,
                hessian_update=fides.hessian_approximation.BFGS(),
            ),
            n_starts=n_starts,
            progress_bar=verbose,
            engine=pypesto.engine.MultiProcessEngine(n_procs=n_cpus),
        )

    logging.info('Starting MCMC')
    if mcmc_type == 'parallel_tempering':
        _sampler = sample.AdaptiveParallelTemperingSampler(
            internal_sampler=sample.AdaptiveMetropolisSampler(),
            n_chains=n_chains,
            options=dict(show_progress=verbose),
        )
    elif mcmc_type == 'NUTS':
        # The dominant cost of NUTS here is the number of leapfrog steps, each of
        # which triggers an AMICI gradient simulation.
        _sampler = PymcSampler2(
            progressbar=verbose,
            tune=n_tune,
            chains=n_chains,
            cores=n_chains,
            init="adapt_full",
            target_accept=0.9,
            max_treedepth=8,
        )
    else:
        raise ValueError(f"Unknown mcmc_type: {mcmc_type}")

    start_time = time.time()
    _result = sample.sample(
        problem=_pypesto_problem,
        n_samples=n_mcmc_samples,
        sampler=_sampler,
        result=_result,
    )
    end_time = time.time()
    logging.info(f"Finished MCMC in {end_time - start_time} seconds.")
    if mcmc_type != 'NUTS':
        sample.geweke_test(_result)

    if data_df is None:
        return _result
    return _result, _petab_problem, _pypesto_problem


def get_mcmc_posterior_samples(res):
    if mcmc_type == 'parallel_tempering':
        burn_in = sample.geweke_test(res)
        if burn_in == res.sample_result.trace_x.shape[1]:
            logging.warning("All samples are burn-in; using first chain.")
            return res.sample_result.trace_x[0]
        return res.sample_result.trace_x[0, burn_in:]
    elif mcmc_type == 'NUTS':
        return res.sample_result.trace_x
    else:
        raise ValueError(f"Unknown mcmc_type: {mcmc_type}")


def _sample_per_chain(ps, n_final):
    """Subsample n_final draws per chain. ps shape: (n_chains, n_samples, n_params)."""
    n_c, _, n_p = ps.shape
    out = np.empty((n_c, n_final, n_p), dtype=ps.dtype)
    for i in range(n_c):
        idx = np.random.choice(ps.shape[1], size=n_final, replace=True)
        out[i] = ps[i, idx]
    return out


def run_mcmc_for_dataset(petab_prob, pypesto_prob, sim_data_df) -> np.ndarray:
    """Run MCMC for one dataset. Returns array of shape (n_chains, n_final_samples, n_params)."""
    n_params = len(pypesto_prob.x_free_indices)
    n_return_chains = n_chains if mcmc_type == 'NUTS' else 1

    if all(np.isnan(sim_data_df['simulation'])):
        return np.full((n_return_chains, n_final_samples, n_params), np.nan)

    r, _, _ = run_mcmc(
        petab_problem=petab_prob,
        data_df=sim_data_df,
    )

    if r is None:
        return np.full((n_return_chains, n_final_samples, n_params), np.nan)

    ps = get_mcmc_posterior_samples(r)
    if mcmc_type == 'parallel_tempering':
        ps = ps[None, ...]  # add chain axis: (1, n_samples, n_params)
    return _sample_per_chain(ps, n_final_samples)


if __name__ == "__main__":
    if not validation_data_path.exists():
        logging.error(f"Validation data not found at {validation_data_path}")
        sys.exit(1)

    with open(validation_data_path, 'rb') as f:
        validation_data = pickle.load(f)

    n_datasets = len(validation_data['sim_data_df'])
    if dataset_idx >= n_datasets:
        logging.error(f"dataset_idx={dataset_idx} out of range (n_datasets={n_datasets})")
        sys.exit(1)

    if result_path.exists():
        logging.info(f"Result for dataset {dataset_idx} already exists, skipping.")
        sys.exit(0)

    logging.info(f"Running MCMC for dataset {dataset_idx}/{n_datasets} ({problem_name})")

    pypesto_problem, petab_problem, _, _ = load_problem(problem_name)
    sim_data_df = validation_data['sim_data_df'][dataset_idx]

    samples = run_mcmc_for_dataset(petab_problem, pypesto_problem, sim_data_df)
    # shape: (n_chains, n_final_samples, n_params)

    with open(result_path, 'wb') as f:
        pickle.dump(samples, f)

    logging.info(f"Saved to {result_path}")
