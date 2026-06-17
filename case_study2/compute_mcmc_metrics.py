import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import sys
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import median_abs_deviation

import bayesflow as bf
from bayesflow.diagnostics.metrics import (
    root_mean_squared_error,
    posterior_contraction,
    calibration_error,
    classifier_two_sample_test,
    accuracy_random_points,
)

from case_study2.helper_pypesto import (
    load_problem,
    get_samples_from_dict,
    compute_likelihood_parallel,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("pypesto").setLevel(logging.ERROR)

n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
BASE = Path(__file__).resolve().parent

problem_name = "Beer_MolBioSystems2014"

lustre_models_dir = Path("/lustre/scratch/data/jarruda_hpc-diffusion_experiments/case_study2/models")
lustre_metrics_dir = Path("/lustre/scratch/data/jarruda_hpc-diffusion_experiments/case_study2/metrics")
models_dir = lustre_models_dir if lustre_models_dir.exists() else BASE / "models"
metrics_dir = lustre_metrics_dir if lustre_metrics_dir.exists() else BASE / "metrics"

validation_data_path = models_dir / f'validation_data_petab_{problem_name}.pkl'
merged_samples_path = models_dir / f'mcmc_samples_{problem_name}.pkl'
metrics_path = metrics_dir / f'{problem_name}_mcmc_metrics.csv'


def load_all_dataset_results(models_dir: Path, problem_name: str, n_datasets: int) -> np.ndarray:
    """
    Load per-dataset MCMC results and stack into (n_datasets, n_chains, n_samples, n_params).
    Datasets with missing files are filled with NaN.
    """
    samples_list = []
    missing = []

    for i in range(n_datasets):
        path = models_dir / f'mcmc_samples_{problem_name}_dataset{i}.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                samples_list.append(pickle.load(f))  # (n_chains, n_samples, n_params)
        else:
            missing.append(i)
            samples_list.append(None)

    if missing:
        logging.warning(f"Missing results for {len(missing)} datasets: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # Determine shape from first available result
    ref = next(s for s in samples_list if s is not None)
    n_chains, n_final_samples, n_params = ref.shape

    stacked = np.full((n_datasets, n_chains, n_final_samples, n_params), np.nan)
    for i, s in enumerate(samples_list):
        if s is not None:
            stacked[i] = s

    return stacked


if __name__ == "__main__":
    # Load validation data
    if not validation_data_path.exists():
        logging.error(f"Validation data not found at {validation_data_path}")
        sys.exit(1)

    with open(validation_data_path, 'rb') as f:
        validation_data = pickle.load(f)

    n_datasets = len(validation_data['sim_data_df'])
    logging.info(f"Loaded validation data: {n_datasets} datasets")

    # Load or assemble merged samples
    if merged_samples_path.exists():
        logging.info(f"Loading merged samples from {merged_samples_path}")
        with open(merged_samples_path, 'rb') as f:
            mcmc_samples = pickle.load(f)
    else:
        logging.info("Assembling per-dataset results...")
        mcmc_samples = load_all_dataset_results(models_dir, problem_name, n_datasets)
        # shape: (n_datasets, n_chains, n_final_samples, n_params)
        with open(merged_samples_path, 'wb') as f:
            pickle.dump(mcmc_samples, f)
        logging.info(f"Merged samples saved to {merged_samples_path}")

    n_available_datasets = int((~np.isnan(mcmc_samples[:, 0, 0, 0])).sum())
    logging.info(f"Valid datasets: {n_available_datasets}/{n_datasets}")

    pypesto_problem, petab_problem, _, _ = load_problem(problem_name)
    param_names = [
        name for i, name in enumerate(pypesto_problem.x_names)
        if i in pypesto_problem.x_free_indices
    ]
    targets_all = pypesto_problem.get_reduced_vector(validation_data['amici_params'].T).T

    n_chains_available = mcmc_samples.shape[1]
    # valid mask per chain: (n_datasets, n_chains)
    valid_mask = ~np.isnan(mcmc_samples.sum(axis=(2, 3)))

    # Diagnostic plots (first chain with valid data)
    plots_dir = BASE / "plots"
    plots_dir.mkdir(exist_ok=True)

    for chain_idx in range(n_chains_available):
        mask = valid_mask[:, chain_idx]
        if not np.any(mask):
            logging.warning(f"No valid datasets for chain {chain_idx}, skipping plots.")
            continue

        chain_samples = mcmc_samples[mask, chain_idx]   # (n_valid, n_samples, n_params)
        chain_targets = targets_all[mask]

        fig = bf.diagnostics.recovery(
            estimates=chain_samples,
            targets=chain_targets,
            variable_names=param_names,
        )
        fig.savefig(plots_dir / f"{problem_name}_mcmc_recovery_chain{chain_idx}.png")

        fig = bf.diagnostics.calibration_ecdf(
            estimates=chain_samples,
            targets=chain_targets,
            variable_names=param_names,
            stacked=True,
        )
        fig.savefig(plots_dir / f"{problem_name}_mcmc_calibration_chain{chain_idx}.png")
        break  # one set of diagnostic plots is enough

    # Compute metrics
    if metrics_path.exists():
        logging.info("Metrics already computed, loading.")
        mcmc_df = pd.read_csv(metrics_path)
    else:
        metric_rows = []
        for chain_idx in range(n_chains_available):
            mask = valid_mask[:, chain_idx]
            if not np.any(mask):
                logging.warning(f"No valid datasets for chain {chain_idx}, skipping metrics.")
                continue

            chain_samples = mcmc_samples[mask, chain_idx]

            # Subset validation data to valid datasets
            test_data = {}
            for key, values in validation_data.items():
                if key == 'sim_data_df':
                    test_data[key] = [v for i, v in enumerate(values) if mask[i]]
                else:
                    test_data[key] = values[mask]

            test_targets = get_samples_from_dict(test_data, pypesto_problem)

            rand_idx = np.random.choice(chain_samples.shape[1])
            workflow_samples_aug = compute_likelihood_parallel(
                petab_problem, chain_samples[:, rand_idx], test_data, n_jobs=n_cpus
            )
            test_data_aug = compute_likelihood_parallel(
                petab_problem, test_data['amici_params'], test_data, n_jobs=n_cpus
            )

            workflow_samples_aug = workflow_samples_aug[~np.isnan(workflow_samples_aug).any(axis=1)]
            test_data_aug = test_data_aug[~np.isnan(test_data_aug).any(axis=1)]
            logging.info(
                f"chain {chain_idx}: {workflow_samples_aug.shape[0]} workflow samples, "
                f"{test_data_aug.shape[0]} test data samples"
            )

            tarp_out = accuracy_random_points(chain_samples, test_targets)
            metric_rows.append({
                'model': 'MCMC',
                'sampler': 'MCMC',
                'chain_idx': chain_idx,
                'n_datasets': int(chain_samples.shape[0]),
                'nrmse': root_mean_squared_error(
                    chain_samples, test_targets, aggregation=np.nanmedian
                )['values'].mean(),
                'nrmse_mad': root_mean_squared_error(
                    chain_samples, test_targets, aggregation=median_abs_deviation
                )['values'].mean(),
                'posterior_contraction': posterior_contraction(
                    chain_samples, test_targets, aggregation=np.nanmedian
                )['values'].mean(),
                'posterior_contraction_mad': posterior_contraction(
                    chain_samples, test_targets, aggregation=median_abs_deviation
                )['values'].mean(),
                'posterior_calibration_error': calibration_error(
                    chain_samples, test_targets, aggregation=np.nanmedian
                )['values'].mean(),
                'posterior_calibration_error_mad': calibration_error(
                    chain_samples, test_targets, aggregation=median_abs_deviation
                )['values'].mean(),
                'posterior_tarp': tarp_out['values'],
                'posterior_tarp_p': tarp_out['ks_pvalue'],
                'c2st': classifier_two_sample_test(workflow_samples_aug, test_data_aug),
            })

        mcmc_df = pd.DataFrame(metric_rows)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        mcmc_df.to_csv(metrics_path, index=False)
        logging.info(f"Metrics saved to {metrics_path}")

    logging.info("Done!")
