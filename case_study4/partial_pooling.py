import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import median_abs_deviation

import bayesflow as bf
import keras

from bayesflow.diagnostics.metrics import calibration_error as ece, root_mean_squared_error as nrmse

import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

from case_study4.settings import EPOCHS, BATCH_SIZE, N_TRAINING_BATCHES, N_TRIALS, N_SUBJECTS, N_TEST, N_SAMPLES, BASE, METHOD, STEPS, MAX_STEP
from case_study4.ddm_simulator import simulator_hierarchical, prior_global_score, beta_from_normal


param_names_global = ['mu_nu', 'mu_log_alpha', 'mu_log_t0',
                      'log_sigma_nu', 'log_sigma_log_alpha', 'log_sigma_log_t0',
                      'beta_raw']
pretty_param_names_global = [r'$\mu_\nu$', r'$\mu_{\log \alpha}$', r'$\mu_{\log t_0}$',
                              r'$\log \sigma_\nu$', r'$\log \sigma_{\log \alpha}$', r'$\log \sigma_{\log t_0}$',
                              r'$\beta$']
param_names_local = ['nu', 'alpha', 't0']
param_metrics = ['nu', 'alpha', 't0', 'beta']
pretty_param_names_local = [r'$\nu_p$', r'$\alpha_p$', r'$t_{0,p}$'] + [r'$\beta$']


#%%
adapter = (
    bf.adapters.Adapter()
    .to_array()
    .convert_dtype("float64", "float32")
    .concatenate(param_names_global, into="inference_variables")
    .rename("sim_data", "summary_variables")
)

workflow = bf.BasicWorkflow(
    adapter=adapter,
    summary_network=bf.networks.SetTransformer(summary_dim=16, dropout=0.1),
    inference_network=bf.networks.DiffusionModel(),
)

model_path = BASE / 'models' / 'partial_pooling_global.keras'
if not os.path.exists(model_path):
    training_data = simulator_hierarchical.sample_parallel((N_TRAINING_BATCHES * BATCH_SIZE), n_trials=N_TRIALS)

    history = workflow.fit_offline(
        training_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    workflow.approximator.save(model_path)
else:
    workflow.approximator = keras.models.load_model(model_path)

#%%
test_data = simulator_hierarchical.sample_parallel(N_TEST, n_subjects=N_SUBJECTS, n_trials=N_TRIALS)

#%%
logging.info("Starting Partial-Pooling (global) inference...")
workflow_global = bf.CompositionalWorkflow.from_basic_workflow(workflow)
global_posterior = workflow_global.compositional_sample(
    num_samples=N_SAMPLES,
    conditions={'sim_data': test_data['sim_data']},
    compute_prior_score=prior_global_score,
    mini_batch_size=3,
    method=METHOD,
    steps=STEPS,
    max_steps=MAX_STEP,
    batch_size=BATCH_SIZE,
    compositional_bridge_d0=0.1
)
ps = global_posterior.copy()
ps['beta'] = beta_from_normal(ps['beta_raw'])
ps.pop('beta_raw')

fig = bf.diagnostics.recovery(
    estimates=ps,
    targets=test_data,
    variable_names=pretty_param_names_global
)
fig.savefig(BASE / "plots" / "partial_pooling_global_recovery.png")
plt.show()

fig = bf.diagnostics.calibration_ecdf(
    estimates=ps,
    targets=test_data,
    variable_names=pretty_param_names_global
)
fig.savefig(BASE / "plots" / "partial_pooling_global_calibration.png")
plt.show()

metrics = {
    'NRMSE': nrmse(ps, test_data)['values'],
    'NRMSE-mad': nrmse(ps, test_data, aggregation=median_abs_deviation)['values'],
    'calibration_error': ece(ps, test_data)['values'],
    'calibration_error-mad': ece(ps, test_data,
                                 aggregation=median_abs_deviation)['values'],
}
with open(BASE / 'metrics' / 'partial_pooling_global_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

#%% Local model
adapter_subjects = (
    bf.adapters.Adapter()
    .to_array()
    .convert_dtype("float64", "float32")
    .log(["alpha", "t0"])  # log-transform alpha and t0 to make them unbounded
    .concatenate(param_names_local, into="inference_variables")
    .concatenate(param_names_global, into="inference_conditions")
    .rename("sim_data", "summary_variables")
)

workflow_local = bf.BasicWorkflow(
    adapter=adapter_subjects,
    summary_network=bf.networks.SetTransformer(summary_dim=16, dropout=0.1),
    inference_network=bf.networks.StableConsistencyModel(),
)

model_path = BASE / 'models' / 'partial_pooling_local.keras'
if not os.path.exists(model_path):
    training_data = simulator_hierarchical.sample_parallel((N_TRAINING_BATCHES * BATCH_SIZE), n_trials=N_TRIALS)

    history = workflow_local.fit_offline(
        training_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    workflow_local.approximator.save(model_path)
else:
    workflow_local.approximator = keras.models.load_model(model_path)

#%%
logging.info("Starting Partial-Pooling (local) inference...")

samples_local = workflow_local.ancestral_sample(
    conditions={"sim_data": test_data['sim_data']},
    ancestral_conditions=global_posterior,
    batch_size=BATCH_SIZE*10,
)

#%%
samples = samples_local.copy()
test_params_local = {}
for p in param_names_local:
    test_params_local[p] = test_data[p].reshape(-1, 1)

test_params_local['beta'] = np.repeat(test_data['beta'], N_SUBJECTS, axis=-1).reshape(-1, 1)
samples['beta'] = np.repeat(ps['beta'][:, None], N_SUBJECTS, axis=1)

for k, v in samples.items():
    samples[k] = v.reshape(-1, N_SAMPLES, 1)

fig = bf.diagnostics.recovery(
    estimates=samples,
    targets=test_params_local,
    variable_names=pretty_param_names_local,
    variable_keys=param_metrics
)
fig.savefig(BASE / "plots" / "partial_pooling_local_recovery.png")
plt.show()

fig = bf.diagnostics.calibration_ecdf(
    estimates=samples,
    targets=test_params_local,
    variable_names=pretty_param_names_local,
    variable_keys=param_metrics
)
fig.savefig(BASE / "plots" / "partial_pooling_local_calibration.png")
plt.show()

metrics = {
    'NRMSE': nrmse(samples, test_params_local, variable_keys=param_metrics)['values'],
    'NRMSE-mad': nrmse(samples, test_params_local, variable_keys=param_metrics,
                       aggregation=median_abs_deviation)['values'],
    'calibration_error': ece(samples, test_params_local, variable_keys=param_metrics)['values'],
    'calibration_error-mad': ece(samples, test_params_local, variable_keys=param_metrics,
                                 aggregation=median_abs_deviation)['values'],
}

with open(BASE / 'metrics' / f'partial_pooling_local_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
logging.info('Done.')
