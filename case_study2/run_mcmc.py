# # PEtab benchmark model with BayesFlow
import os
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "tensorflow"
else:
    print(f"Using '{os.environ['KERAS_BACKEND']}' backend")

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
from bayesflow.diagnostics.metrics import root_mean_squared_error, posterior_contraction, calibration_error, classifier_two_sample_test

import logging
pypesto.logging.log(level=logging.ERROR, name="pypesto.petab", console=True)
logging.getLogger("pypesto").setLevel(logging.ERROR)

from case_study2.helper_pypesto import load_problem, simulate_parallel, get_samples_from_dict, compute_likelihood_parallel, create_pypesto_problem, sample_from_prior, simulator_amici

job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
BASE = Path(__file__).resolve().parent
num_training_sets = 512 * 64
num_validation_sets = 1000
problem_name = "Beer_MolBioSystems2014"
models_dir = BASE / "models"
metrics_dir = BASE / 'metrics'
#metrics_dir = Path('/lustre/scratch/data/jarruda_hpc-diffusion_experiments/case_study2/metrics')
#models_dir = Path('/lustre/scratch/data/jarruda_hpc-diffusion_experiments/case_study2/models')
mcmc_path = BASE / "models" / f'mcmc_samples_{problem_name}.pkl'
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
        print("Skipping optimization, sample start points for chains from prior")
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

    _sampler = sample.AdaptiveParallelTemperingSampler(
        internal_sampler=sample.AdaptiveMetropolisSampler(),
        n_chains=n_chains,
        options=dict(show_progress=verbose)
    )

    _result = sample.sample(
        problem=_pypesto_problem,
        n_samples=n_samples,
        sampler=_sampler,
        result=_result,
    )
    sample.geweke_test(_result)

    if data_df is None:
        return _result
    return _result, _petab_problem, _pypesto_problem


def get_mcmc_posterior_samples(res):
    burn_in = sample.geweke_test(res)
    if burn_in == res.sample_result.trace_x.shape[1]:
        print("Warning: All samples are considered burn-in.")
        _samples = res.sample_result.trace_x[0]  # only use first chain
    else:
        _samples = res.sample_result.trace_x[0, burn_in:]  # only use first chain
    return _samples


#%%
def run_mcmc_single(petab_prob, pypesto_prob, sim_data_df, n_starts,
                    n_mcmc_samples, n_final_samples, n_chains):
    import amici
    import logging
    amici.swig_wrappers.logger.setLevel(logging.CRITICAL)
    pypesto.logging.log(level=logging.ERROR, name="pypesto.petab", console=True)

    if all(np.isnan(sim_data_df['simulation'])):
        return np.full((n_final_samples, len(pypesto_prob.x_free_indices)), np.nan)

    r, _, _ = run_mcmc(
        petab_problem=petab_prob,
        data_df=sim_data_df,
        n_optimization_starts=n_starts,
        n_samples=n_mcmc_samples,
        n_chains=n_chains,
        n_procs=1,
    )

    if r is None:
        return np.full((n_final_samples, len(pypesto_prob.x_free_indices)), np.nan)

    ps = get_mcmc_posterior_samples(r)
    # num_samples random samples from posterior
    idx = np.random.choice(ps.shape[0], size=n_final_samples)
    return ps[idx]

#%%
if __name__ == "__main__":
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
            print("Training data not found")
    else:
        print('Generate data')
        training_data = simulate_parallel(num_training_sets, amici_predictor, factory, petab_problem, pypesto_problem)
        validation_data = simulate_parallel(num_validation_sets, amici_predictor, factory, petab_problem,
                                            pypesto_problem, return_df=True)

        with open(models_dir / f'training_data_petab_{problem_name}.pkl', 'wb') as f:
            pickle.dump(training_data, f)
        with open(models_dir / f'validation_data_petab_{problem_name}.pkl', 'wb') as f:
            pickle.dump(validation_data, f)

    if RUN_TEST:
        n_optimization_starts = 1
        test_params = sample_from_prior(petab_problem=petab_problem, pypesto_problem=pypesto_problem)
        print('test_params', test_params)
        test = simulator_amici(test_params['amici_params'], amici_predictor, factory, petab_problem, pypesto_problem, return_df=True)

        new_result, new_petab_problem, new_pypesto_problem = run_mcmc(
            petab_problem=petab_problem,
            data_df=test['sim_data_df'],
            n_optimization_starts=n_optimization_starts,
            n_samples=1e2,
            n_procs=n_cpus,
            n_chains=2,
            verbose=True
        )

        if n_optimization_starts > 0:
            visualize.waterfall(new_result, size=(6, 4))
            visualize.parameters(new_result, size=(6, 25))
            sim_dict = visualize_optimized_model_fit(
                petab_problem=new_petab_problem,
                result=new_result,
                pypesto_problem=new_pypesto_problem,
                return_dict=True
            )
            print('error:', test_params['amici_params']-new_result.optimize_result.x[0])

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
        exit()

    if os.path.exists(mcmc_path):
        with open(mcmc_path, 'rb') as f:
            mcmc_posterior_samples = pickle.load(f)
    else:
        print("Running MCMC...")
        mcmc_posterior_samples = Parallel(n_jobs=n_cpus, verbose=10)(
            delayed(run_mcmc_single)(
                petab_prob=petab_problem,
                pypesto_prob=pypesto_problem,
                sim_data_df=sim_data_df,
                n_starts=10,
                n_mcmc_samples=1e5,
                n_final_samples=1000,
                n_chains=5,
            ) for sim_data_df in validation_data['sim_data_df']
        )
        mcmc_posterior_samples = np.array(mcmc_posterior_samples)

        with open(mcmc_path, 'wb') as f:
            pickle.dump(mcmc_posterior_samples, f)
    mcmc_mask = ~np.isnan(mcmc_posterior_samples.sum(axis=(1, 2)))

    #%%
    fig = bf.diagnostics.recovery(
        estimates=mcmc_posterior_samples[mcmc_mask],
        targets=pypesto_problem.get_reduced_vector(validation_data['amici_params'].T).T[mcmc_mask],
        variable_names=param_names,
    )
    fig.savefig(BASE / "plots" / f"{problem_name}_mcmc_recovery.png")

    fig = bf.diagnostics.calibration_ecdf(
        estimates=mcmc_posterior_samples[mcmc_mask],
        targets=pypesto_problem.get_reduced_vector(validation_data['amici_params'].T).T[mcmc_mask],
        variable_names=param_names,
        difference=True,
        stacked=True
    )
    fig.savefig(BASE / "plots" / f"{problem_name}_mcmc_calibration.png")

    test_data = {}
    for key, values in validation_data.items():
        if key == 'sim_data_df':
            test_data[key] = [v for i, v in enumerate(values) if mcmc_mask[i]]
        else:
            test_data[key] = values[mcmc_mask]

    #%%
    if os.path.exists(metrics_dir / f'{problem_name}_mcmc_metrics.csv'):
        with open(metrics_dir / f'{problem_name}_mcmc_metrics.csv', 'rb') as f:
            mcmc_df = pd.read_csv(f)
        print("MCMC metrics already computed.")
    else:
        test_targets = get_samples_from_dict(test_data, pypesto_problem)

        rand_idx = np.random.choice(mcmc_posterior_samples.shape[1])
        workflow_samples_aug = compute_likelihood_parallel(petab_problem, mcmc_posterior_samples[mcmc_mask, rand_idx],
                                                           test_data, n_jobs=n_cpus)

        # augment test data
        test_data_aug = compute_likelihood_parallel(petab_problem, test_data['amici_params'], test_data,
                                                    n_jobs=n_cpus)

        workflow_samples_aug = workflow_samples_aug[~np.isnan(workflow_samples_aug).any(axis=1)]
        test_data_aug = test_data_aug[~np.isnan(test_data_aug).any(axis=1)]
        print(f"{workflow_samples_aug.shape[0]} workflow samples and {test_data_aug.shape[0]} test data samples.")

        mcmc_df = pd.DataFrame([{
            'model': 'MCMC',
            'sampler': 'MCMC',
            'nrmse': root_mean_squared_error(
                mcmc_posterior_samples[mcmc_mask], test_targets, aggregation=np.nanmedian
            )['values'].mean(),
            'nrmse_mad': root_mean_squared_error(
                mcmc_posterior_samples[mcmc_mask], test_targets, aggregation=median_abs_deviation
            )['values'].mean(),
            'posterior_contraction': posterior_contraction(
                mcmc_posterior_samples[mcmc_mask], test_targets, aggregation=np.nanmedian
            )['values'].mean(),
            'posterior_contraction_mad': posterior_contraction(
                mcmc_posterior_samples[mcmc_mask], test_targets, aggregation=median_abs_deviation
            )['values'].mean(),
            'posterior_calibration_error': calibration_error(
                mcmc_posterior_samples[mcmc_mask], test_targets, aggregation=np.nanmedian
            )['values'].mean(),
            'posterior_calibration_error_mad': calibration_error(
                mcmc_posterior_samples[mcmc_mask], test_targets, aggregation=median_abs_deviation
            )['values'].mean(),
            'c2st': classifier_two_sample_test(workflow_samples_aug, test_data_aug)
        }], index=[0])
        with open(metrics_dir / f'{problem_name}_mcmc_metrics.csv', 'wb') as f:
            mcmc_df.to_csv(f)
