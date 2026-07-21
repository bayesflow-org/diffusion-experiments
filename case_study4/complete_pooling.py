# %%
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
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
logging.info("Starting No-Pooling inference...")
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
logging.info("Starting Complete-Pooling inference...")
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
    compositional_bridge_d0=0.01
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

#%%
# analysis of different choices
class StepCountHandler(logging.Handler):
    """Capture integration step counts emitted by BayesFlow."""

    _pattern = re.compile(r"Finished integration after (\d+)")

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.steps = []

    def emit(self, record):
        match = self._pattern.search(record.getMessage())
        if match:
            self.steps.append(int(match.group(1)))

    def reset(self):
        self.steps.clear()

bf_logger = logging.getLogger("bayesflow")
step_handler = StepCountHandler()
bf_logger.addHandler(step_handler)
bf_logger.setLevel(logging.DEBUG)

def metric_with_spread(estimates, targets, keys):
    """Point estimate (default aggregation) plus MAD spread over datasets, averaged over params."""
    point_nrmse = np.mean(nrmse(estimates, targets, variable_keys=keys)['values'])
    err_nrmse = np.mean(nrmse(estimates, targets, variable_keys=keys,
                              aggregation=median_abs_deviation)['values'])
    point_cal = np.mean(ece(estimates, targets, variable_keys=keys)['values'])
    err_cal = np.mean(ece(estimates, targets, variable_keys=keys,
                          aggregation=median_abs_deviation)['values'])
    return point_nrmse, err_nrmse, point_cal, err_cal


#%%
mini_batch_fractions = [0.1, 0.5, 1.0]  # mini-batch size as a percentage of the number of factors
damping_factors = [0.001, 0.01, 0.1, 1.0]
n_factors_list = [2, 10, 20, 50, 75, 100]

if os.path.exists(BASE / 'metrics' / f'detailed_analysis_{N_TRIALS}.pkl'):
    with open(BASE / 'metrics' / f'detailed_analysis_{N_TRIALS}.pkl', 'rb') as f:
        detailed_results = pickle.load(f)
else:
    logging.info("Starting detailed compositional analysis...")
    detailed_results = {}  # (mini_batch_fraction, damping) -> {'n_factors', 'nrmse', 'calibration'}
    for mini_batch_fraction in mini_batch_fractions:
        for damping in damping_factors:
            n_factors_used = []
            nrmse_vals, nrmse_err, cal_vals, cal_err = [], [], [], []
            steps_vals, steps_err = [], []
            for n_factors in n_factors_list:
                mini_batch = max(1, int(round(mini_batch_fraction * n_factors)))

                step_handler.reset()  # capture the per-batch integration steps for this run
                post = workflow_trials.compositional_sample(
                    num_samples=N_SAMPLES,
                    conditions={'sim_data': test_data['sim_data'][:, :n_factors]},
                    compute_prior_score=prior_flat_score,
                    mini_batch_size=mini_batch,
                    method=METHOD,
                    steps=STEPS,
                    max_steps=MAX_STEP,
                    batch_size=BATCH_SIZE,
                    compositional_bridge_d0=damping,
                )
                post['beta'] = beta_from_normal(post['beta_raw'])
                post.pop('beta_raw')
                post['alpha'] = np.exp(post['log_alpha'])
                post['t0'] = np.exp(post['log_t0'])

                # point estimate over all datasets; error bar = MAD spread over datasets
                p_nrmse, e_nrmse, p_cal, e_cal = metric_with_spread(post, test_data, param_metrics)

                # per-batch integration steps captured from the DEBUG logs during this run
                batch_steps = np.asarray(step_handler.steps)

                n_factors_used.append(n_factors)
                nrmse_vals.append(p_nrmse)
                nrmse_err.append(e_nrmse)
                cal_vals.append(p_cal)
                cal_err.append(e_cal)
                steps_vals.append(np.mean(batch_steps) if batch_steps.size else np.nan)
                steps_err.append(np.std(batch_steps) if batch_steps.size else np.nan)

            detailed_results[(mini_batch_fraction, damping)] = {
                'n_factors': np.array(n_factors_used),
                'nrmse': np.array(nrmse_vals),
                'nrmse_err': np.array(nrmse_err),
                'calibration': np.array(cal_vals),
                'calibration_err': np.array(cal_err),
                'steps': np.array(steps_vals),  # mean per-batch integration steps
                'steps_err': np.array(steps_err),  # std of per-batch integration steps
            }

    with open(BASE / 'metrics' / f'detailed_analysis_{N_TRIALS}.pkl', 'wb') as f:
        pickle.dump(detailed_results, f)

#%%
# single hue = create_figure.py's 'Complete Pooling' color; damping factors are told apart
# by shade (light -> dark), line style, and marker instead of by different hues
pooling_color = "#7570B3"  # muted purple
fontsize = 11

def _shades(hex_color, n):
    """n shades of hex_color, ordered light -> dark."""
    base = np.array(mcolors.to_rgb(hex_color))
    light = 1 - 0.6 * (1 - base)  # blended toward white
    dark = 0.4 * base             # blended toward black
    cmap = mcolors.LinearSegmentedColormap.from_list('pooling_shades', [light, dark])
    return [mcolors.to_hex(cmap(x)) for x in np.linspace(0, 1, n)]

# grid of small multiples: one row per metric, one column per mini-batch fraction.
metric_rows = [
    ('nrmse', 'nrmse_err', 'NRMSE'),
    ('calibration', 'calibration_err', 'Calibration Error'),
    ('steps', 'steps_err', 'Steps'),
]
damping_markers = ['o', 's', '^', 'D']
damping_linestyles = ['-', '--', '-.', ':']
damping_shades = _shades(pooling_color, len(damping_factors))

fig, axes = plt.subplots(len(metric_rows), len(mini_batch_fractions),
                         figsize=(10, 5),
                         sharex=True, sharey='row', squeeze=False, layout='constrained')

for r, (key, err_key, ylabel) in enumerate(metric_rows):
    for c, mini_batch_fraction in enumerate(mini_batch_fractions):
        ax = axes[r, c]
        for damping, marker, linestyle, shade in zip(damping_factors, damping_markers,
                                                      damping_linestyles, damping_shades):
            res = detailed_results[(mini_batch_fraction, damping)]
            # clip lower error bars so they don't extend below zero
            yerr = [np.minimum(res[err_key], res[key]), res[err_key]]
            ax.errorbar(res['n_factors'], res[key], yerr=yerr,
                        marker=marker, linestyle=linestyle, color=shade,
                        capsize=3, label=fr'$d_0={damping}$')

        if key == 'steps':
            # dashed black line marking the integration step ceiling
            ax.axhline(MAX_STEP, color='black', linestyle='--', linewidth=1)
            ax.text(0.65, MAX_STEP-20, 'max steps', color='black', fontsize=fontsize - 1,
                    ha='right', va='top', transform=ax.get_yaxis_transform())


        ax.set_xticks(n_factors_list)  # ticks only where we have data
        ax.set_xlim(0, N_SUBJECTS)
        ax.tick_params(labelsize=fontsize)
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if r == 0:
            ax.set_title(f'Mini-Batch = {int(mini_batch_fraction * 100)}% of Groups', fontsize=fontsize)
        if r == len(metric_rows) - 1:
            ax.set_xlabel('Number of Groups', fontsize=fontsize)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize)
    axes[r, 0].set_ylim(bottom=0)  # all three metrics are non-negative (shared within the row)

fig.align_ylabels(axes[:, 0])  # line up the row labels at the same x-position

# single legend: one line per damping factor, distinguished by shade + style + marker
handles = [Line2D([0], [0], color=shade, marker=marker, linestyle=linestyle, label=fr'$d_0={damping}$')
           for damping, marker, linestyle, shade in zip(damping_factors, damping_markers,
                                                         damping_linestyles, damping_shades)]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.08),
           ncol=len(handles), fontsize=fontsize, frameon=False)

fig.savefig(BASE / 'plots' / f'compositional_detailed_analysis.pdf', bbox_inches='tight')
plt.show()
