import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pickle

from case_study4.settings import BASE, N_TRIALS

with open(BASE / 'metrics' / f'no_pooling_metrics_{N_TRIALS}.pkl', 'rb') as f:
    no_pooling_metrics = pickle.load(f)

with open(BASE / 'metrics' / f'complete_pooling_metrics_{N_TRIALS}.pkl', 'rb') as f:
    complete_pooling_metrics = pickle.load(f)

with open(BASE / 'metrics' / 'partial_pooling_local_metrics.pkl', 'rb') as f:
    partial_pooling_local_metrics = pickle.load(f)

pretty_param_names = [r'$\nu^{(r)}$', r'$\alpha^{(r)}$', r'$t_{0}^{(r)}$', r'$\beta^{(r)}$']
n_params = len(pretty_param_names)

pooling_models = [
    'No-Pooling',  # trials
    'Complete Pooling',  # subjects
    'Partial Pooling' # local values
]
colors = [
    "#E7298A",  # magenta pink
    "#7570B3",  # muted purple
    "#1B9E77",  # teal green
    "#D95F02",  # deep orange
]
fontsize = 11

metrics_names = list(complete_pooling_metrics.keys())
metrics_names_pretty = [r'NRMSE', "", r'Calibration Error']

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(5, 3), layout='constrained')

metrics_rows = [0, 2]  # which metrics to plot, one per row
model_metrics = [
    no_pooling_metrics,
    complete_pooling_metrics,
    partial_pooling_local_metrics,
]
markers = ['o', 's', '^']  # one per pooling model
group_x = np.arange(n_params)
width = 0.18
offsets = np.linspace(-width, width, len(model_metrics))

for r, mi in enumerate(metrics_rows):
    ax = axes[r]
    metric = metrics_names[mi]

    for k in range(len(model_metrics)):
        vals = [model_metrics[k][metric][j] for j in range(n_params)]
        yerrs = [model_metrics[k][metric + '-mad'][j] for j in range(n_params)]
        # clip lower error bars so they don't extend below 0
        lower_errs = [min(yerrs[j], vals[j]) for j in range(n_params)]
        container = ax.errorbar(
            group_x + offsets[k],
            vals,
            yerr=[lower_errs, yerrs],
            fmt=markers[k],
            linestyle='',
            alpha=0.8,
            color=colors[k],
            label=pooling_models[k] if r == 0 else None,
            clip_on=False,
        )
        # keep error bars semi-transparent but draw the marker fully opaque
        container.lines[0].set_alpha(1.0)
        # draw markers on top of the axis spines
        container.lines[0].set_zorder(10)

    ax.set_xlim(-0.5, n_params - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks(group_x, pretty_param_names, fontsize=fontsize)
    ax.set_ylabel(metrics_names_pretty[mi], fontsize=fontsize)
    ax.grid(True)

for a in axes:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

handles = [
    Patch(color=colors[i], label=pooling_models[i], alpha=0.8) for i in range(len(pooling_models))
]

fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.1),
           ncols=len(model_metrics), fontsize=fontsize, frameon=False)
plt.savefig(BASE / 'plots' / 'pooling_metrics.pdf', bbox_inches='tight')
plt.show()
