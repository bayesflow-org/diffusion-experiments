import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sbibm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings(
    "ignore",
    message="JULIA_SYSIMAGE_DIFFEQTORCH not set"
)

BASE = Path(__file__).resolve().parent


def get_lueckmann_results(algorithm='NPE'):
    results_lueckmann = pd.read_csv(BASE / 'plots' / 'lueckmann_results.csv')
    results_lueckmann = results_lueckmann.loc[results_lueckmann['num_simulations'] == max(results_lueckmann['num_simulations']),
            ['task', 'algorithm', 'C2ST']]
    results_lueckmann = results_lueckmann.groupby(['algorithm', 'task'])['C2ST'].mean().reset_index()
    if algorithm == 'min':
        results_lueckmann = results_lueckmann.groupby(['task']).min('C2ST').reset_index()
    else:
        results_lueckmann = results_lueckmann[results_lueckmann.algorithm == algorithm].reset_index(drop=True)
    return results_lueckmann


def create_model_config():
    colors = {
        # Flow Matching family
        "flow_matching": "#1B9E77",
        "ot_flow_matching": "#2E7D32",  # dark medium green
        "ot_partial_flow_matching": "#43A047",  # balanced green
        "cot_flow_matching": "#4CAF50",  # standard green
        "cot_0_02_flow_matching": "#66BB6A",  # light green
        "cot_0_05_flow_matching": "#81C784",  # lighter green
        "cot_0_1_flow_matching": "#A5D6A7",  # very light green
        "cot_partial_flow_matching": "#66BB6A",  # light green
        "flow_matching_edm": "#81C784",  # pale green

        # Consistency family
        "consistency_model": "#D95F02",
        "stable_consistency_model": "#E6AB02",

        # EDM family
        "diffusion_edm_vp": "#E7298A",
        "diffusion_edm_vp_ema": "#AD1457",
        "diffusion_edm_vp_transformer": "#F768A1",  # bright pink
        "diffusion_edm_ve": "#FB9A99",

        # Cosine family
        "diffusion_cosine_F": "#7570B3",
        "diffusion_cosine_v": "#9E9AC8",
        "diffusion_cosine_v_lw": "#BCBDDC",
        "diffusion_cosine_noise": "#54278F",
    }
    return {"visualization": {"colors": colors}}


def get_model_name(k=None):
    names = {
        "flow_matching": r"Flow Matching$\,$",
        "ot_flow_matching": r"Flow Matching (OT)$\,$",
        "ot_partial_flow_matching": r"Flow Matching (POT)$\,$",
        "cot_flow_matching": r"Flow Matching (COT)$\,$",  # 0.01
        "cot_0_02_flow_matching": r"Flow Matching (COT, $r=0.02$)$\,$",
        "cot_0_05_flow_matching": r"Flow Matching (COT, $r=0.05$)$\,$",
        "cot_0_1_flow_matching": r"Flow Matching (COT, $r=0.1$)$\,$",
        "cot_partial_flow_matching": r"Flow Matching (CPOT)$\,$",
        "flow_matching_edm": r"Flow Matching ($\rho = -0.6$)$\,$",
        "consistency_model": r"Discrete Consistency$\,$",
        "stable_consistency_model": r"Continuous Consistency$\,$",
        "diffusion_edm_vp": r"VP EDM$\,$",
        "diffusion_edm_vp_ema": r"VP EDM EMA$\,$",
        "diffusion_edm_vp_transformer": r"VP EDM (Transformer)$\,$",
        "diffusion_edm_ve": r"VE EDM$\,$",
        "diffusion_cosine_F": r"Cosine $\mathbf{F}$-pred.$\,$",
        "diffusion_cosine_v": r"Cosine $\boldsymbol{v}$-pred.$\,$",
        "diffusion_cosine_v_lw": r"Cosine $\boldsymbol{v}$-pred. (lw)$\,$",
        "diffusion_cosine_noise": r"Cosine $\boldsymbol{\epsilon}$-pred.$\,$",
    }
    if k is None:
        return list(names.keys())
    return names.get(k, k.replace("_", " ").title())


def get_model_name_plotly(k=None):  # without LaTeX
    names = {
        "flow_matching": "Flow Matching",
        "ot_flow_matching": "Flow Matching (OT)",
        "ot_partial_flow_matching": "Flow Matching (POT)",
        "cot_flow_matching": "Flow Matching (COT)",
        "cot_0_02_flow_matching": "Flow Matching (COT, r=0.02)",
        "cot_0_05_flow_matching": "Flow Matching (COT, r=0.05)",
        "cot_0_1_flow_matching": "Flow Matching (COT, r=0.1)",
        "cot_partial_flow_matching": "Flow Matching (CPOT)",
        "flow_matching_edm": "Flow Matching (ρ = −0.6)",
        "consistency_model": "Discrete Consistency",
        "stable_consistency_model": "Continuous Consistency",
        "diffusion_edm_vp": "VP EDM",
        "diffusion_edm_vp_ema": "VP EDM (EMA)",
        "diffusion_edm_vp_transformer": "VP EDM (Transformer)",
        "diffusion_edm_ve": "VE EDM",
        "diffusion_cosine_F": "Cosine F-pred.",
        "diffusion_cosine_v": "Cosine v-pred.",
        "diffusion_cosine_v_lw": "Cosine v-pred. (lw)",
        "diffusion_cosine_noise": "Cosine ε-pred.",
    }

    if k is None:
        return list(names.keys())

    return names.get(k, k.replace("_", " ").title())


def get_sampler_name(k=None):
    names = {
        'ode-euler': 'Euler',
        'ode-euler-mini': r'Euler ($2\%$ steps)',
        'ode-euler-small': r'Euler ($10\%$ steps)',
        'ode-euler-scheduled': r'Euler (Scheduled)',
        'ode-rk45': 'RK45',
        'ode-rk45-adaptive': 'RK45-Adaptive',
        'ode-rk45-adaptive-joint': 'RK45-Adaptive (Batch)',
        'ode-tsit5': 'TSIT5',
        'ode-tsit5-adaptive': 'TSIT5-Adaptive',
        'ode-tsit5-adaptive-joint': 'TSIT5-Adaptive (Batch)',
        'sde-euler-adaptive': 'EulerM-Adaptive',
        'sde-euler': 'EulerM',
        'sde-euler-pc': 'EulerM-PC',
        'sde-sea': 'SEA',
        'sde-shark': 'ShARK',
        'sde-two_step-adaptive': '2Step-Adaptive',
        'sde-langevin': 'Langevin',
        'sde-langevin-pc': 'Langevin-PC',
    }
    if k is None:
        return list(names.keys())
    return names.get(k, k.replace("_", " ").title())


def pareto_best_sampler(
        df: pd.DataFrame,
        model_col="model",
        sampler_col="sampler",
        c2st_mean_col="c2st",
        c2st_std_col="std",
        time_mean_col="time",
        time_std_col="time_std",
        n_runs: int = 10,
        z: float = 1.96,
        return_all_front: bool = False,
) -> pd.DataFrame:
    """
    Minimizes (c2st_mean, time_mean) with dominance tolerances derived from SEs
    computed from std over repeated runs.
    Selects best sampler(s) per model (aggregated across all tasks), then returns
    all rows for those (model, sampler) combinations across all tasks.
    """
    if n_runs <= 0:
        raise ValueError("n_runs must be a positive integer")

    def se(std):
        if pd.isna(std):
            return 0.0
        return float(std) / np.sqrt(float(n_runs))

    def tau(std_a, std_b):
        se_a, se_b = se(std_a), se(std_b)
        return z * np.sqrt(se_a * se_a + se_b * se_b)

    def dominates(a, b) -> bool:
        """
        For MINIMIZATION: a dominates b if:
        - a is no worse than b in both objectives (within tolerance)
        - a is strictly better than b in at least one objective (beyond tolerance)
        """
        c_mu_a, c_sd_a, t_mu_a, t_sd_a = a
        c_mu_b, c_sd_b, t_mu_b, t_sd_b = b

        tau_c = tau(c_sd_a, c_sd_b)
        tau_t = tau(t_sd_a, t_sd_b)

        # For minimization: a is no worse if a <= b + tolerance
        no_worse_c2st = (c_mu_a <= c_mu_b + tau_c)
        no_worse_time = (t_mu_a <= t_mu_b + tau_t)

        # For minimization: a is strictly better if a < b - tolerance
        strictly_better_c2st = (c_mu_a < c_mu_b - tau_c)
        strictly_better_time = (t_mu_a < t_mu_b - tau_t)

        return (no_worse_c2st and no_worse_time and
                (strictly_better_c2st or strictly_better_time))

    # Step 1: Aggregate metrics per (model, sampler) across all tasks
    agg_dict = {
        c2st_mean_col: 'mean',
        time_mean_col: 'mean',
    }

    # Aggregate std columns if they exist (using RMS)
    if c2st_std_col in df.columns:
        agg_dict[c2st_std_col] = lambda x: np.sqrt(np.mean(x ** 2))
    if time_std_col in df.columns:
        agg_dict[time_std_col] = lambda x: np.sqrt(np.mean(x ** 2))

    sampler_stats = df.groupby([model_col, sampler_col]).agg(agg_dict).reset_index()

    # Step 2: Find Pareto-optimal samplers per model
    best_samplers_list = []

    for model, model_group in sampler_stats.groupby(model_col):
        g = model_group.dropna(subset=[c2st_mean_col, time_mean_col])

        if g.empty:
            # Fallback: pick best available
            best = model_group.sort_values([c2st_mean_col, time_mean_col]).head(1)
            best_samplers_list.append(best[[model_col, sampler_col]])
            continue

        # Fill missing std columns with 0.0
        c_std = g[c2st_std_col].fillna(0.0) if c2st_std_col in g.columns else pd.Series(0.0, index=g.index)
        t_std = g[time_std_col].fillna(0.0) if time_std_col in g.columns else pd.Series(0.0, index=g.index)

        X = np.column_stack([
            g[c2st_mean_col].values,
            c_std.values,
            g[time_mean_col].values,
            t_std.values
        ])

        n = len(g)
        dominated = np.zeros(n, dtype=bool)

        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[i]:
                    continue
                if dominates(X[j], X[i]):
                    dominated[i] = True
                    break

        front = g[~dominated].copy()

        # Ensure at least one sampler
        if front.empty:
            front = g.sort_values([c2st_mean_col, time_mean_col]).head(1)

        if len(front) > 1:
            cols = [sampler_col, c2st_mean_col, c2st_std_col, time_mean_col, time_std_col]
            cols = [c for c in cols if c in front.columns]
            preview = front.sort_values([c2st_mean_col, time_mean_col])[cols].head(10).to_string(index=False)
            logging.info(
                "Multiple Pareto-front samplers for model=%s, k=%d (z=%.3g, n_runs=%d)\n%s",
                model, len(front), z, n_runs, preview
            )

        if not return_all_front:
            front = front.sort_values([c2st_mean_col, time_mean_col]).head(1)

        best_samplers_list.append(front[[model_col, sampler_col]])

    # Step 3: Merge back to get all rows for selected (model, sampler) pairs
    best_samplers = pd.concat(best_samplers_list, ignore_index=True)
    result = df.merge(best_samplers, on=[model_col, sampler_col], how='inner')

    return result


def plot_benchmark_results(df, show_sampler, save_path=None):
    """Plot C2ST benchmark results across tasks with robust handling of varying data subsets."""
    if df['problem'].iloc[0] == 'all':
        return

    problem_names = sbibm.get_available_tasks()
    problem_names_nice = np.array([sbibm.get_task(p).name_display for p in problem_names])
    problem_dim = [sbibm.get_task(p).dim_parameters for p in problem_names]
    data_dim = [sbibm.get_task(p).dim_data for p in problem_names]
    problem_names_nice = np.array([
        f'{n}\n' + r'$\mathrm{dim}(\boldsymbol{\theta})$' + f'$={p_dim}$' +
        r', $\mathrm{dim}(\mathbf{y})$' + f'$={d_dim}$'
        for n, p_dim, d_dim in zip(problem_names_nice, problem_dim, data_dim)
    ])
    problem_order = np.lexsort((data_dim, problem_dim))

    n_problems = len(problem_order)
    results_lueckmann = get_lueckmann_results()
    colors = create_model_config()['visualization']['colors']
    fig, axes = plt.subplots(2, n_problems // 2, figsize=(12, 4), sharex=True, sharey=True, layout='constrained')
    axes = axes.flatten()

    # Collect all unique model-sampler combinations across all problems
    all_combinations = []
    for model_key in get_model_name():
        # Single sampler per model
        model_data = df[df['model'] == model_key]
        if len(model_data) > 0:
            sampler = model_data['sampler'].iloc[0]
            label = get_model_name(model_key)
            if 'consistency' not in model_key and show_sampler != 'all':
                label += f' ({get_sampler_name(sampler)})'
            all_combinations.append({
                'model': model_key,
                'sampler': sampler,
                'label': label,
                'color': colors.get(model_key, 'gray')
            })

    # Create a consistent mapping from combination to x-position
    combination_to_x = {
        (combo['model'], combo['sampler']): idx
        for idx, combo in enumerate(all_combinations)
    }

    # Collect handles and labels for legend
    all_labels = []
    all_handles = []
    seen_labels = set()

    for idx, problem_idx in enumerate(problem_order):
        ax = axes[idx]
        subset = df[df['problem'] == problem_names[problem_idx]]

        # Plot each combination if it exists for this problem
        for combo in all_combinations:
            model_key = combo['model']
            sampler = combo['sampler']

            # Find matching data
            model_data = subset[
                (subset['model'] == model_key) &
                (subset['sampler'] == sampler)
                ]

            if len(model_data) == 0:
                continue

            # Get consistent x-position
            x_pos = combination_to_x[(model_key, sampler)]

            # Plot error bar
            handle = ax.errorbar(
                x_pos,
                model_data['c2st'].iloc[0],
                yerr=model_data['std'].iloc[0],
                fmt='o', markersize=6, capsize=3,
                color=combo['color'],
                markeredgewidth=0.5,
                label=combo['label']
            )

            # Collect unique labels for legend (only from first subplot)
            if idx == 0 and combo['label'] not in seen_labels:
                all_labels.append(combo['label'])
                all_handles.append(handle)
                seen_labels.add(combo['label'])

        # Add reference line (if available for this problem)
        ref_data = results_lueckmann.loc[
            results_lueckmann['task'] == problem_names[problem_idx]
            ]

        if len(ref_data) > 0:
            ref_val = ref_data['C2ST'].iloc[0]
            ref_line = ax.axhline(
                y=ref_val, color='black', linestyle='--', linewidth=1,
                label='Lueckmann et al. NPE', zorder=-1
            )

            if idx == 0 and 'Lueckmann et al. NPE' not in seen_labels:
                all_handles.append(ref_line)
                all_labels.append(r'Lueckmann et al. NPE')
                seen_labels.add('Lueckmann et al. NPE')

        # Styling
        ax.set_title(problem_names_nice[problem_idx], fontsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.48, 1.0)
        ax.set_xticks([])

        # Set consistent x-axis limits across all subplots
        ax.set_xlim(-0.5, len(all_combinations) - 0.5)

    # Set y-labels
    axes[0].set_ylabel(r'$\mathrm{C2ST}$', fontsize=10)
    axes[5].set_ylabel(r'$\mathrm{C2ST}$', fontsize=10)

    # Add legend
    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.3),
        ncol=3,
        fancybox=False,
        fontsize=10
    )

    # Save figure
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()
    return


def plot_benchmark_results_plotly(df):
    """Plot C2ST benchmark results across tasks using Plotly with hover support."""
    if df['problem'].iloc[0] == 'all':
        return
    show_sampler = 'best'

    FONT_AXIS = 11
    FONT_TITLE = 11
    MARKER_SIZE = 4

    problem_names = sbibm.get_available_tasks()
    problem_names_nice = np.array([sbibm.get_task(p).name_display for p in problem_names])
    problem_dim = [sbibm.get_task(p).dim_parameters for p in problem_names]
    data_dim = [sbibm.get_task(p).dim_data for p in problem_names]

    problem_order = np.lexsort((data_dim, problem_dim))
    n_problems = len(problem_order)

    results_lueckmann = get_lueckmann_results()
    colors = create_model_config()['visualization']['colors']

    fig = make_subplots(
        rows=n_problems // 2,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[problem_names_nice[i] for i in problem_order]
    )

    all_combinations = []
    for model_key in get_model_name_plotly():
        model_data = df[df['model'] == model_key]
        if len(model_data) > 0:
            sampler = model_data['sampler'].iloc[0]
            label = get_model_name_plotly(model_key)
            if 'consistency' not in model_key and show_sampler != 'all':
                label += f' ({get_sampler_name(sampler)})'

            all_combinations.append({
                'model': model_key,
                'sampler': sampler,
                'label': label,
                'color': colors.get(model_key, 'gray')
            })

    combination_to_x = {
        (combo['model'], combo['sampler']): idx
        for idx, combo in enumerate(all_combinations)
    }

    shown_labels = set()

    for plot_idx, problem_idx in enumerate(problem_order):
        row = plot_idx // 2 + 1
        col = plot_idx % 2 + 1

        task_name = problem_names[problem_idx]
        subset = df[df['problem'] == task_name]

        for combo in all_combinations:
            model_key = combo['model']
            sampler = combo['sampler']

            model_data = subset[
                (subset['model'] == model_key) &
                (subset['sampler'] == sampler)
            ]

            if len(model_data) == 0:
                continue

            x_pos = combination_to_x[(model_key, sampler)]
            show_legend = combo['label'] not in shown_labels

            fig.add_trace(
                go.Scatter(
                    x=[x_pos],
                    y=[model_data['c2st'].iloc[0]],
                    error_y=dict(
                        type='data',
                        array=[model_data['std'].iloc[0]],
                        visible=True
                    ),
                    mode='markers',
                    marker=dict(
                        size=MARKER_SIZE,
                        color=combo['color'],
                        line=dict(width=0.5)
                    ),
                    name=combo['label'],
                    legendgroup=combo['label'],
                    showlegend=show_legend,
                    hovertemplate=(
                        '<b>Task:</b> %{customdata[0]}<br>'
                        '<b>Model:</b> %{customdata[1]}<br>'
                        '<b>Sampler:</b> %{customdata[2]}<br>'
                        '<b>C2ST:</b> %{y:.3f}<br>'
                        '<b>Std:</b> %{customdata[3]:.3f}'
                        '<extra></extra>'
                    ),
                    customdata=[[
                        problem_names_nice[problem_idx],
                        get_model_name_plotly(model_key),
                        get_sampler_name(sampler),
                        model_data['std'].iloc[0]
                    ]]
                ),
                row=row,
                col=col
            )

            if show_legend:
                shown_labels.add(combo['label'])

        ref_data = results_lueckmann.loc[
            results_lueckmann['task'] == task_name
        ]

        if len(ref_data) > 0:
            ref_val = ref_data['C2ST'].iloc[0]
            show_legend = 'Lueckmann et al. NPE' not in shown_labels

            fig.add_trace(
                go.Scatter(
                    x=[-0.5, len(all_combinations) - 0.5],
                    y=[ref_val, ref_val],
                    mode='lines',
                    line=dict(color='black', dash='dash', width=1),
                    name='Lueckmann et al. NPE',
                    showlegend=show_legend,
                    legendgroup='reference',
                    hovertemplate=(
                        '<b>Task:</b> %{customdata}<br>'
                        '<b>Reference C2ST:</b> %{y:.3f}'
                        '<extra></extra>'
                    ),
                    customdata=[task_name]
                ),
                row=row,
                col=col
            )

            if show_legend:
                shown_labels.add('Lueckmann et al. NPE')

        fig.update_xaxes(
            range=[-0.5, len(all_combinations) - 0.5],
            showticklabels=False,
            row=row,
            col=col
        )

        fig.update_yaxes(
            range=[0.48, 1.0],
            row=row,
            col=col
        )

    fig.update_layout(
        height=600,
        width=700,
        hovermode='closest',
        legend=dict(
            orientation="v",
            x=1.0,
            y=1.0,
            xanchor='left',
            yanchor='top',
        ),
        margin=dict(
            t=40,
            b=80,
            l=60,
            r=260,
        ),
    )

    fig.update_yaxes(
        title_text=r'$\mathrm{C2ST}$',
        row=1,
        col=1,
        title_font=dict(size=FONT_AXIS),
        tickfont=dict(size=FONT_AXIS),
    )
    fig.update_yaxes(
        title_text=r'$\mathrm{C2ST}$',
        row=2,
        col=1,
        title_font=dict(size=FONT_AXIS),
        tickfont=dict(size=FONT_AXIS),
    )
    fig.update_yaxes(
        title_text=r'$\mathrm{C2ST}$',
        row=3,
        col=1,
        title_font=dict(size=FONT_AXIS),
        tickfont=dict(size=FONT_AXIS),
    )
    fig.update_yaxes(
        title_text=r'$\mathrm{C2ST}$',
        row=4,
        col=1,
        title_font=dict(size=FONT_AXIS),
        tickfont=dict(size=FONT_AXIS),
    )
    fig.update_yaxes(
        title_text=r'$\mathrm{C2ST}$',
        row=5,
        col=1,
        title_font=dict(size=FONT_AXIS),
        tickfont=dict(size=FONT_AXIS),
    )

    fig.update_annotations(font_size=FONT_TITLE)

    return fig


def plot_by_sampler(df, col='c2st', col_std='std', save_path=None):
    """Plot benchmark results grouped by sampler, showing different models."""
    if df['problem'].iloc[0] != 'all':
        return

    unique_samplers = get_sampler_name()
    unique_samplers = [u for u in unique_samplers if u in df.sampler.unique()]
    colors = create_model_config()['visualization']['colors']
    n_samplers = len(unique_samplers)
    n_cols = np.ceil(n_samplers / 2).astype(int)

    fig, axes = plt.subplots(2, n_cols, figsize=(12, 4), sharex=True, sharey=True, layout='constrained')
    axes = axes.flatten()

    # Collect all unique models across all samplers
    all_models = []
    for model_key in get_model_name():
        if model_key in df['model'].unique():
            all_models.append({
                'key': model_key,
                'label': get_model_name(model_key),
                'color': colors.get(model_key, 'gray')
            })

    # Create consistent mapping from model to x-position
    model_to_x = {model['key']: idx for idx, model in enumerate(all_models)}

    all_labels = []
    all_handles = []
    seen_labels = set()

    for idx, sampler in enumerate(unique_samplers):
        ax = axes[idx]
        subset = df[df['sampler'] == sampler]

        # Plot each model if it exists for this sampler
        for model_info in all_models:
            model_key = model_info['key']
            model_data = subset[subset['model'] == model_key]

            if len(model_data) == 0:
                continue

            sampler_val = model_data['sampler'].iloc[0]

            # Skip consistency models with non-Euler samplers
            if sampler_val != 'ode-euler' and 'consistency' in model_key:
                continue

            val = model_data[col].iloc[0]
            std = model_data[col_std].iloc[0]

            # Asymmetric error bars
            if col == 'c2st':  # (bounded by [0.5, 1.0])
                yerr = np.array([
                    [np.minimum(abs(0.5 - val), std)],
                    [np.minimum(abs(1.0 - val), std)]
                ])
            else:
                yerr = np.array([
                    [np.minimum(val, std)],
                    [std]
                ])

            # Get consistent x-position
            x_pos = model_to_x[model_key]

            # Plot data
            handle = ax.errorbar(
                x_pos, val, yerr=yerr,
                fmt='o', markersize=6, capsize=3,
                color=model_info['color'], markeredgewidth=0.5,
                label=model_info['label']
            )

            # Collect labels from first subplot
            if idx == 0 and model_info['label'] not in seen_labels:
                all_labels.append(model_info['label'])
                all_handles.append(handle)
                seen_labels.add(model_info['label'])

        # Styling
        sampler_display = get_sampler_name(sampler)
        ax.set_title(sampler_display, fontsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])

        # Set consistent x-axis limits across all subplots
        ax.set_xlim(-0.5, len(all_models) - 0.5)

    # Set y-labels and scale
    if col == 'c2st':
        axes[0].set_ylabel(r'$\mathrm{C2ST}$', fontsize=10)
        axes[n_cols].set_ylabel(r'$\mathrm{C2ST}$', fontsize=10)
    else:
        axes[0].set_ylabel(r'Time [s]', fontsize=10)
        axes[n_cols].set_ylabel(r'Time [s]', fontsize=10)
    axes[0].set_yscale('log')

    # Hide unused subplots
    if n_samplers < len(axes):
        for i in range(n_samplers, len(axes)):
            axes[i].set_visible(False)

    # Legend
    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.3),
        ncol=4,
        fancybox=False,
        fontsize=10
    )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_by_model(df, col='c2st', col_std='std', save_path=None):
    """Plot scores grouped by model, showing different samplers."""
    if df['problem'].iloc[0] != 'all':
        return

    unique_models = get_model_name()
    unique_models = [u for u in unique_models if u in df.model.unique() and 'consistency' not in u]
    n_models = len(unique_models)
    n_cols = np.ceil(n_models / 2).astype(int)
    colors = create_model_config()['visualization']['colors']

    fig, axes = plt.subplots(2, n_cols, figsize=(12, 4), sharex=True, sharey=True, layout='constrained')
    axes = axes.flatten()

    # Collect all unique samplers across all models
    samplers_list = get_sampler_name()
    all_samplers = []
    for sampler in samplers_list:
        if sampler in df['sampler'].unique():
            all_samplers.append({
                'key': sampler,
                'label': get_sampler_name(sampler)
            })

    # Create consistent mapping from sampler to x-position
    sampler_to_x = {s['key']: idx for idx, s in enumerate(all_samplers)}

    # Collect all x-tick labels for consistent display
    all_labels_x = [s['label'] for s in all_samplers]

    for idx, model_key in enumerate(unique_models):
        ax = axes[idx]
        subset = df[df['model'] == model_key]

        # Plot each sampler if it exists for this model
        for sampler_info in all_samplers:
            sampler = sampler_info['key']
            model_data = subset[subset['sampler'] == sampler]

            if len(model_data) == 0:
                continue

            model_name_val = model_data['model'].iloc[0]
            sampler_val = model_data['sampler'].iloc[0]

            # Skip consistency models with non-Euler samplers
            if sampler_val != 'ode-euler' and 'consistency' in model_name_val:
                continue

            val = model_data[col].iloc[0]
            std = model_data[col_std].iloc[0]

            # Asymmetric error bars
            if col == 'c2st':   # (bounded by [0.5, 1.0])
                yerr = np.array([
                    [np.minimum(abs(0.5 - val), std)],
                    [np.minimum(abs(1.0 - val), std)]
                ])
            else:
                yerr = np.array([
                    [np.minimum(val, std)],
                    [std]
                ])

            # Get consistent x-position
            x_pos = sampler_to_x[sampler]

            # Plot data
            ax.errorbar(
                x_pos, val, yerr=yerr,
                fmt='o', markersize=6, capsize=3,
                color=colors.get(model_name_val, 'gray'),
                markeredgewidth=0.5
            )

        # Styling
        model_display = get_model_name(model_key)
        ax.set_title(model_display, fontsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True, alpha=0.3)

        # Set consistent x-ticks across all subplots
        ax.set_xticks(ticks=np.arange(len(all_labels_x)), labels=all_labels_x, rotation=90, fontsize=10)
        ax.set_xlim(-0.5, len(all_samplers) - 0.5)

    # Set y-labels
    if col == 'c2st':
        axes[0].set_ylabel(r'$\mathrm{C2ST}$', fontsize=10)
        axes[n_cols].set_ylabel(r'$\mathrm{C2ST}$', fontsize=10)
    else:
        axes[0].set_ylabel(r'Time [s]', fontsize=10)
        axes[n_cols].set_ylabel(r'Time [s]', fontsize=10)

    # Hide unused subplots
    if n_models < len(axes):
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_low_budget_c2st_by_problem(df, save_path=None):
    """Plot C2ST results for low-budget samplers, one subplot per problem."""
    if df['problem'].iloc[0] == 'all':
        return
    samplers = ['ode-euler-mini', 'ode-euler-small', 'ode-euler']
    colors = ["#0072B2", "#E69F00", "#009E73"]

    problem_names = sbibm.get_available_tasks()
    problem_names_nice = np.array([sbibm.get_task(p).name_display for p in problem_names])
    problem_dim = [sbibm.get_task(p).dim_parameters for p in problem_names]
    data_dim = [sbibm.get_task(p).dim_data for p in problem_names]
    problem_order = np.lexsort((data_dim, problem_dim))
    n_problems = len(problem_names)

    fig, axs = plt.subplots(
        ncols=n_problems // 2,
        nrows=2,
        figsize=(12, 4),
        layout='constrained',
        sharey=True, sharex=True
    )

    # Ensure axs is iterable for n_problems == 1
    if n_problems == 1:
        axs = np.array([axs])
    axs = axs.flatten()

    for p_i, problem_idx in enumerate(problem_order):
        ax = axs[p_i]
        df_p = df[df['problem'] == problem_names[problem_idx]]

        # Model labels from the first sampler (assumed consistent)
        subset_models = df_p[df_p.sampler == samplers[0]]
        model_labels = [get_model_name(m) for m in subset_models.model.values]
        x = np.arange(len(model_labels))

        for s_i, s in enumerate(samplers):
            subset = df_p[df_p.sampler == s]
            ax.errorbar(
                x=x,
                y=subset.c2st.values,
                yerr=subset['std'].values,
                marker='o',
                markersize=5,
                color=colors[s_i],
                label=get_sampler_name(s) if p_i == 0 else None
            )

        ax.set_title(problem_names_nice[problem_idx], fontsize=10)
        ax.set_xticks(x, model_labels, rotation=90, ha='right')
        ax.grid(True, alpha=0.3)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if p_i == 0 or p_i == n_problems // 2:
            ax.set_ylabel(r'$\mathrm{C2ST}$', fontsize=10)

    fig.legend(
        fontsize=10,
        ncols=3,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.15),
        fancybox=False
    )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()
