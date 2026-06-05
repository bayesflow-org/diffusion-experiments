# # PEtab benchmark model with BayesFlow
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # jax might deadlock joblib here
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed
from pathlib import Path

import petab
import pypesto.petab
import pypesto.optimize as optimize
import pypesto.sample as sample
import pypesto.visualize as visualize
from pypesto.visualize.model_fit import visualize_optimized_model_fit
from scipy.stats import median_abs_deviation

import bayesflow as bf
from bayesflow.diagnostics.metrics import root_mean_squared_error, posterior_contraction, calibration_error, classifier_two_sample_test, accuracy_random_points

import logging
pypesto.logging.log(level=logging.ERROR, name="pypesto.petab", console=True)
logging.getLogger("pypesto").setLevel(logging.ERROR)

from case_study2.helper_pypesto import load_problem, simulate_parallel, get_samples_from_dict, compute_likelihood_parallel, create_pypesto_problem, sample_from_prior, simulator_amici

job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 2))
BASE = Path(__file__).resolve().parent
num_training_sets = 512 * 64
num_validation_sets = 1000
problem_name = "Beer_MolBioSystems2014"
lustre_metrics_dir = Path("/lustre/scratch/data/jarruda_hpc-diffusion_experiments/case_study2/metrics")
metrics_dir = lustre_metrics_dir if lustre_metrics_dir.exists() else BASE / "metrics"
lustre_models_dir = Path("/lustre/scratch/data/jarruda_hpc-diffusion_experiments/case_study2/models")
models_dir = lustre_models_dir if lustre_models_dir.exists() else BASE / "models"
mcmc_path = models_dir / f'mcmc_samples_{problem_name}.pkl'
mcmc_metrics_path = metrics_dir / f'{problem_name}_mcmc_metrics.csv'
mcmc_type = ['parallel_tempering', 'NUTS'][1]
mcmc_chain_path = models_dir / f'mcmc_samples_{problem_name}_chain{job_id}.pkl'
n_chains = 10
RUN_TEST = False


def run_mcmc(petab_problem, data_df=None, n_optimization_starts=0, n_chains=10, n_samples=10000, n_procs=10,
             verbose=False) -> Union[pypesto.result.Result, tuple[pypesto.result.Result, petab.Problem, pypesto.Problem]]:
    if data_df is None:
        # use true data
        _pypesto_problem = create_pypesto_problem(petab_problem)
        _petab_problem = None
    else:
        _measurement_df = data_df
        if not 'measurement' in _measurement_df.columns:
            _measurement_df['measurement'] = _measurement_df['simulation']  # pypesto expects measurement column
        _pypesto_problem, _petab_problem = create_pypesto_problem(petab_problem, _measurement_df)

    if n_optimization_starts == 0:
        logging.info("Skipping optimization, sample start points for chains from prior")
        _result = None
    else:
        # do the optimization
        _result = optimize.minimize(
            problem=_pypesto_problem,
            optimizer=optimize.FidesOptimizer(verbose=0),
            n_starts=n_optimization_starts,
            engine=pypesto.engine.MultiProcessEngine(n_procs=n_procs) if n_procs > 1 else None,
            progress_bar=verbose
        )

    if mcmc_type == 'parallel_tempering':
        _sampler = sample.AdaptiveParallelTemperingSampler(
           internal_sampler=sample.AdaptiveMetropolisSampler(),
           n_chains=n_chains,
           options=dict(show_progress=verbose)
        )
    elif mcmc_type == 'NUTS':
        _sampler = sample.PymcSampler(
            progressbar=verbose,
            tune=10 if RUN_TEST else 1000,  # default
            chains=n_chains,
            cores=1, #min(n_procs, n_chains)
        )
    else:
        raise ValueError("Unknown mcmc_type {}".format(mcmc_type))

    _result = sample.sample(
        problem=_pypesto_problem,
        n_samples=n_samples,
        sampler=_sampler,
        result=_result,
    )
    if mcmc_type != 'NUTS':
        sample.geweke_test(_result)

    if data_df is None:
        return _result
    return _result, _petab_problem, _pypesto_problem


def get_mcmc_posterior_samples(res, mcmc_type):
    if mcmc_type == 'parallel_tempering':
        burn_in = sample.geweke_test(res)
        if burn_in == res.sample_result.trace_x.shape[1]:
            logging.warning("All samples are considered burn-in.")
            _samples = res.sample_result.trace_x[0]  # only use first chain
        else:
            _samples = res.sample_result.trace_x[0, burn_in:]  # only use first chain
        return _samples
    elif mcmc_type == 'NUTS':
        # return all chains, pymc discards burn_in automatically
        return res.sample_result.trace_x
    else:
        raise ValueError("Unknown mcmc_type {}".format(mcmc_type))


def _sample_per_chain(ps, n_final_samples):
    """Sample n_final_samples per chain from a (n_chains, n_samples, n_params) tensor."""
    n_chains, _, n_params = ps.shape
    final_ps = np.empty((n_chains, n_final_samples, n_params), dtype=ps.dtype)
    for i in range(n_chains):
        idx = np.random.choice(ps[i].shape[0], size=n_final_samples, replace=True)
        final_ps[i] = ps[i][idx]
    return final_ps


def _flatten_chain_axis(ps):
    """Convert (n_datasets, n_chains, n_samples, n_params) -> (n_datasets, n_chains*n_samples, n_params)."""
    return ps.reshape(ps.shape[0], ps.shape[1] * ps.shape[2], ps.shape[3])

#%%
def run_mcmc_single(petab_prob, pypesto_prob, sim_data_df, n_starts,
                    n_mcmc_samples, n_final_samples, _n_chains):
    import amici
    import logging
    amici.swig_wrappers.logger.setLevel(logging.CRITICAL)
    pypesto.logging.log(level=logging.ERROR, name="pypesto.petab", console=True)
    n_params = len(pypesto_prob.x_free_indices)
    n_return_chains = _n_chains if mcmc_type == 'NUTS' else 1

    if all(np.isnan(sim_data_df['simulation'])):
        return np.full((n_return_chains, n_final_samples, n_params), np.nan)

    r, _, _ = run_mcmc(
        petab_problem=petab_prob,
        data_df=sim_data_df,
        n_optimization_starts=n_starts,
        n_samples=n_mcmc_samples,
        n_chains=_n_chains,
        n_procs=1,
    )

    if r is None:
        return np.full((n_return_chains, n_final_samples, n_params), np.nan)

    ps = get_mcmc_posterior_samples(r, mcmc_type)
    if mcmc_type == 'parallel_tempering':
        # Use single posterior chain and keep an explicit chain axis for downstream consistency.
        ps = ps[None, ...]  # (1, n_samples, n_params)
        return _sample_per_chain(ps, n_final_samples)
    elif mcmc_type == 'NUTS':
        return _sample_per_chain(ps, n_final_samples)
    else:
        raise ValueError("Unknown mcmc_type {}".format(mcmc_type))


#%%
if __name__ == "__main__":
    logging.info(f"job_id={job_id}, n_chains={n_chains}")
    pypesto_problem, petab_problem, factory, amici_predictor = load_problem(problem_name)
    param_names = [name for i, name in enumerate(pypesto_problem.x_names) if i in pypesto_problem.x_free_indices]
    lbs = np.array([lb for i, lb in enumerate(petab_problem.lb_scaled) if i in pypesto_problem.x_free_indices])
    ubs = np.array([ub for i, ub in enumerate(petab_problem.ub_scaled) if i in pypesto_problem.x_free_indices])

    if os.path.exists(models_dir / f"validation_data_petab_{problem_name}.pkl"):
        with open(models_dir / f'validation_data_petab_{problem_name}.pkl', 'rb') as f:
            validation_data = pickle.load(f)
        try:
            with open(models_dir / f'training_data_petab_{problem_name}.pkl', 'rb') as f:
                training_data = pickle.load(f)
        except FileNotFoundError:
            training_data = None
            logging.info("Training data not found")
    else:
        logging.info('Generate data')
        training_data = simulate_parallel(num_training_sets, amici_predictor, factory, petab_problem, pypesto_problem)
        validation_data = simulate_parallel(num_validation_sets, amici_predictor, factory, petab_problem,
                                            pypesto_problem, return_df=True)

        with open(models_dir / f'training_data_petab_{problem_name}.pkl', 'wb') as f:
            pickle.dump(training_data, f)
        with open(models_dir / f'validation_data_petab_{problem_name}.pkl', 'wb') as f:
            pickle.dump(validation_data, f)

    if RUN_TEST:
        n_optimization_starts = 0 if mcmc_type == 'NUTS' else 1
        test_params = sample_from_prior(petab_problem=petab_problem, pypesto_problem=pypesto_problem)
        logging.info(f'test_params {test_params}')
        test = simulator_amici(test_params['amici_params'], amici_predictor, factory, petab_problem, pypesto_problem, return_df=True)

        new_result, new_petab_problem, new_pypesto_problem = run_mcmc(
            petab_problem=petab_problem,
            data_df=test['sim_data_df'],
            n_optimization_starts=n_optimization_starts,
            n_samples=10,
            n_procs=n_cpus,
            n_chains=2,
            verbose=True
        )

        if n_optimization_starts > 0:
            visualize.waterfall(new_result, size=(6, 4))
            plt.show()
            visualize.parameters(new_result, size=(6, 25))
            plt.show()
            visualize.sampling_fval_traces(new_result, size=(6, 25))
            plt.show()
            sim_dict = visualize_optimized_model_fit(
                petab_problem=new_petab_problem,
                result=new_result,
                pypesto_problem=new_pypesto_problem,
                return_dict=True
            )
            plt.show()
            logging.info(f'error: {test_params["amici_params"]-new_result.optimize_result.x[0]}')

            fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(10, 3), layout='constrained')
            obs_name = ['Bac', 'Ind']
            for i, obs_id in enumerate(sim_dict['simulation_df']['observableId'].unique()):
                df_obs = sim_dict['simulation_df'][sim_dict['simulation_df']['observableId'] == obs_id]
                cmap = plt.get_cmap('tab20', len(df_obs['simulationConditionId'].unique()))
                for j, sim_con in enumerate(df_obs['simulationConditionId'].unique()):
                    color = cmap(j)
                    df = df_obs[df_obs['simulationConditionId'] == sim_con]
                    ax[i].plot(df['time'].values, df['simulation'].values[:, 0], 'o', color=color,
                               markersize=0.7, label=f'Condition {j}')
                ax[i].set_ylabel(f'{obs_name[i]} [a.u.]', fontsize=14)
                ax[i].set_xlabel('Time [min]', fontsize=14)
                ax[i].tick_params(axis='x', labelsize=12)
                ax[i].spines['top'].set_visible(False)
                ax[i].spines['right'].set_visible(False)
            plt.savefig(BASE / "plots" / f'petab_benchmark_model_{problem_name}.png')
            plt.show()

        test = run_mcmc_single(
            petab_prob=petab_problem,
            pypesto_prob=pypesto_problem,
            sim_data_df=test['sim_data_df'],
            n_starts=0 if mcmc_type == 'NUTS' else 1,
            n_mcmc_samples=20,
            n_final_samples=10,
            _n_chains=2,
        )
        print(test.shape)
        exit()

    # %%
    if not os.path.exists(mcmc_chain_path):
        logging.info(f"Running MCMC chain {job_id} over {len(validation_data['sim_data_df'])} datasets...")
        chain_samples = Parallel(n_jobs=n_cpus)(
            delayed(run_mcmc_single)(
                petab_prob=petab_problem,
                pypesto_prob=pypesto_problem,
                sim_data_df=sim_data_df,
                n_starts=0 if mcmc_type == 'NUTS' else 10,
                n_mcmc_samples=1e5,
                n_final_samples=1000,
                _n_chains=1,
            ) for sim_data_df in validation_data['sim_data_df']
        )
        # shape: (n_datasets, 1, n_samples, n_parameters)
        chain_samples = np.array(chain_samples)
        with open(mcmc_chain_path, 'wb') as f:
            pickle.dump(chain_samples, f)
        logging.info(f"Saved chain {job_id} to {mcmc_chain_path}")
    else:
        logging.info(f"Chain {job_id} already exists, skipping.")

    # Merge all chains once every job has finished.
    all_chain_paths = [models_dir / f'mcmc_samples_{problem_name}_chain{i}.pkl' for i in range(n_chains)]
    if not all(os.path.exists(p) for p in all_chain_paths):
        logging.info("Not all chains available yet — skipping merge and metrics.")
        logging.info("Done!")
        exit(0)

    if os.path.exists(mcmc_path):
        with open(mcmc_path, 'rb') as f:
            mcmc_posterior_samples = pickle.load(f)
    else:
        logging.info("All chains available — merging...")
        chains = []
        for p in all_chain_paths:
            with open(p, 'rb') as f:
                chains.append(pickle.load(f))
        # concatenate along chain axis: (n_datasets, n_array_jobs, n_samples, n_parameters)
        mcmc_posterior_samples = np.concatenate(chains, axis=1)
        with open(mcmc_path, 'wb') as f:
            pickle.dump(mcmc_posterior_samples, f)
        logging.info(f"Merged samples saved to {mcmc_path}")

    # Keep chain axis and compute valid dataset masks per chain.
    # Shape: (n_datasets, n_chains, n_samples, n_parameters)
    mcmc_mask_per_chain = ~np.isnan(mcmc_posterior_samples.sum(axis=(2, 3)))
    n_available_chains = mcmc_posterior_samples.shape[1]
    targets_all = pypesto_problem.get_reduced_vector(validation_data['amici_params'].T).T

    #%%
    for chain_idx in range(n_available_chains):
        chain_mask = mcmc_mask_per_chain[:, chain_idx]
        if not np.any(chain_mask):
            logging.warning(f"No valid datasets for chain {chain_idx}, skipping diagnostics plot.")
            continue
        chain_samples = mcmc_posterior_samples[chain_mask, chain_idx]
        chain_targets = targets_all[chain_mask]
        fig = bf.diagnostics.recovery(
            estimates=chain_samples,
            targets=chain_targets,
            variable_names=param_names,
        )
        fig.savefig(BASE / "plots" / f"{problem_name}_mcmc_recovery_chain{chain_idx}.png")

        fig = bf.diagnostics.calibration_ecdf(
            estimates=chain_samples,
            targets=chain_targets,
            variable_names=param_names,
            stacked=True
        )
        fig.savefig(BASE / "plots" / f"{problem_name}_mcmc_calibration_chain{chain_idx}.png")
        break

    #%%
    if os.path.exists(mcmc_metrics_path):
        with open(mcmc_metrics_path, 'rb') as f:
            mcmc_df = pd.read_csv(f)
        logging.info("MCMC metrics already computed.")
    else:
        metric_rows = []
        for chain_idx in range(n_available_chains):
            chain_mask = mcmc_mask_per_chain[:, chain_idx]
            if not np.any(chain_mask):
                logging.warning(f"No valid datasets for chain {chain_idx}, skipping metrics.")
                continue
            chain_samples = mcmc_posterior_samples[chain_mask, chain_idx]

            test_data = {}
            for key, values in validation_data.items():
                if key == 'sim_data_df':
                    test_data[key] = [v for i, v in enumerate(values) if chain_mask[i]]
                else:
                    test_data[key] = values[chain_mask]

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
                f"chain {chain_idx}: {workflow_samples_aug.shape[0]} workflow samples and {test_data_aug.shape[0]} test data samples."
            )

            tarp_out = accuracy_random_points(chain_samples, test_targets)
            metric_rows.append({
                'model': f'MCMC',
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
        with open(mcmc_metrics_path, 'wb') as f:
            mcmc_df.to_csv(f)

logging.info("Done!")
