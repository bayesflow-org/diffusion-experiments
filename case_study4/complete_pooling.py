# %%
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import median_abs_deviation

import bayesflow as bf
import keras

from bayesflow.diagnostics.metrics import calibration_error as ece, root_mean_squared_error as nrmse

from case_study4.settings import EPOCHS, BATCH_SIZE, N_TRAINING_BATCHES, N_TRIALS, N_SUBJECTS, N_TEST, N_SAMPLES, BASE, METHOD, STEPS, MAX_STEP
from case_study4.ddm_simulator import simulator_flat, beta_from_normal, prior_flat_score

import logging
logging.getLogger('bayesflow').setLevel(logging.DEBUG)

param_names = ['nu', 'log_alpha', 'log_t0', 'beta_raw']
param_metrics = ['nu', 'alpha', 't0', 'beta']
pretty_param_names = [r'$\nu$', r'$\log \alpha$', r'$\log t_0$', r'$\beta_\text{raw}$']
pretty_param_names_p = [r'$\nu_p$', r'$\log \alpha_p$', r'$\log t_{0,p}$', r'$\beta_p$']

# %%
adapter = (
    bf.adapters.Adapter()
    .to_array()
    .convert_dtype("float64", "float32")
    .concatenate(param_names, into="inference_variables")
    .rename("sim_data", "summary_variables")
)

workflow_trials = bf.CompositionalWorkflow(
    adapter=adapter,
    summary_network=bf.networks.SetTransformer(summary_dim=16, dropout=0.1),
    inference_network=bf.networks.DiffusionModel(),
)

# %%
model_path = BASE / 'models' / f'flat_trial_{N_TRIALS}.keras'
if not os.path.exists(model_path):
    training_data_trials = simulator_flat.sample_parallel((N_TRAINING_BATCHES * BATCH_SIZE), n_trials=N_TRIALS)

    history = workflow_trials.fit_offline(
        training_data_trials,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    workflow_trials.approximator.save(model_path)
else:
    workflow_trials.approximator = keras.models.load_model(model_path)

#%%
test_data = simulator_flat.sample_parallel(N_TEST, n_subjects=N_SUBJECTS, n_trials=N_TRIALS)

#%%
print("Starting No-Pooling inference...")
no_pooling_data = test_data.copy()
no_pooling_data['sim_data'] = test_data['sim_data'][:, 0]  # only first subject, no pooling
no_pooling_ps = workflow_trials.sample(conditions=no_pooling_data, num_samples=N_SAMPLES,
                                       method=METHOD, steps=STEPS, max_steps=MAX_STEP, batch_size=BATCH_SIZE)
no_pooling_ps['beta'] = beta_from_normal(no_pooling_ps['beta_raw'])
no_pooling_ps.pop('beta_raw')

fig = bf.diagnostics.recovery(
    estimates=no_pooling_ps,
    targets=no_pooling_data,
    variable_names=pretty_param_names_p
)
fig.savefig(BASE / 'plots' / f'no_pooling_recovery_{N_TRIALS}.png')
plt.show()

fig = bf.diagnostics.calibration_ecdf(
    estimates=no_pooling_ps,
    targets=no_pooling_data,
    variable_names=pretty_param_names_p
)
fig.savefig(BASE / 'plots' / f'no_pooling_calibration_{N_TRIALS}.png')
plt.show()

no_pooling_ps['alpha'] = np.exp(no_pooling_ps['log_alpha'])
no_pooling_ps['t0'] = np.exp(no_pooling_ps['log_t0'])
metrics = {
    'NRMSE': nrmse(no_pooling_ps, no_pooling_data, variable_keys=param_metrics)['values'],
    'NRMSE-mad': nrmse(no_pooling_ps, no_pooling_data, variable_keys=param_metrics,
                       aggregation=median_abs_deviation)['values'],
    'calibration_error': ece(no_pooling_ps, no_pooling_data, variable_keys=param_metrics)['values'],
    'calibration_error-mad': ece(no_pooling_ps, no_pooling_data, variable_keys=param_metrics,
                                 aggregation=median_abs_deviation)['values'],
}

with open(BASE / 'metrics' / f'no_pooling_metrics_{N_TRIALS}.pkl', 'wb') as f:
    pickle.dump(metrics, f)


#%%
print("Starting Complete-Pooling inference...")
## Complete Pooling
test_posterior_comp = workflow_trials.compositional_sample(
    num_samples=N_SAMPLES,
    conditions={'sim_data': test_data['sim_data']},
    compute_prior_score=prior_flat_score,
    mini_batch_size=3,
    method=METHOD,
    steps=STEPS,
    max_steps=MAX_STEP,
    batch_size=BATCH_SIZE,
)
test_posterior_comp['beta'] = beta_from_normal(test_posterior_comp['beta_raw'])
test_posterior_comp.pop('beta_raw')

fig = bf.diagnostics.recovery(
    estimates=test_posterior_comp,
    targets=test_data,
    variable_names=pretty_param_names_p
)
fig.savefig(BASE / 'plots' / f"complete_pooling_recovery_{N_TRIALS}.png")
plt.show()

fig = bf.diagnostics.calibration_ecdf(
    estimates=test_posterior_comp,
    targets=test_data,
    difference=True,
    variable_names=pretty_param_names_p
)
fig.savefig(BASE / 'plots' / f"complete_pooling_calibration_{N_TRIALS}.png")
plt.show()

test_posterior_comp['alpha'] = np.exp(test_posterior_comp['log_alpha'])
test_posterior_comp['t0'] = np.exp(test_posterior_comp['log_t0'])
metrics = {
    'NRMSE': nrmse(test_posterior_comp, test_data, variable_keys=param_metrics)['values'],
    'NRMSE-mad': nrmse(test_posterior_comp, test_data, variable_keys=param_metrics,
                       aggregation=median_abs_deviation)['values'],
    'calibration_error': ece(test_posterior_comp, test_data, variable_keys=param_metrics)['values'],
    'calibration_error-mad': ece(test_posterior_comp, test_data, variable_keys=param_metrics,
                                 aggregation=median_abs_deviation)['values'],
}

with open(BASE / 'metrics' / f'complete_pooling_metrics_{N_TRIALS}.pkl', 'wb') as f:
    pickle.dump(metrics, f)
