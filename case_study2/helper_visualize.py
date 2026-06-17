import os
import numpy as np
import matplotlib.pyplot as plt


METRICS = ['NRMSE', 'Posterior Contraction', 'Calibration Error', 'c2st', 'TARP']
METRIC_LABELS = {
    'NRMSE': r'$1-$NRMSE',
    'Posterior Contraction': r'Contraction',
    'Calibration Error': 'Calibration',
    'c2st': r'$1-\vert 0.5-\text{C2ST}\vert$',
    'TARP': r'$1-\vert\text{TARP}\vert$',
}


def _create_model_config():
    colors = {
        # Flow Matching family
        "flow_matching": "#1B9E77",
        "ot_flow_matching": "#33A02C",
        "flow_matching_edm": "#66C2A5",
        "flow_matching_ft": "#1B9E77",
        "ot_flow_matching_ft": "#33A02C",
        "flow_matching_edm_ft": "#66C2A5",

        # EDM family
        "diffusion_edm_vp": "#E7298A",
        "diffusion_edm_vp_ema": "#AD1457",
        "diffusion_edm_ve": "#FB9A99",
        "diffusion_edm_vp_ft": "#E7298A",
        "diffusion_edm_ve_ft": "#FB9A99",

        # Cosine family
        "diffusion_cosine_F": "#7570B3",
        "diffusion_cosine_v": "#9E9AC8",
        "diffusion_cosine_noise": "#54278F",
        "diffusion_cosine_F_ft": "#7570B3",
        "diffusion_cosine_v_ft": "#9E9AC8",
        "diffusion_cosine_noise_ft": "#54278F",

        # Consistency family
        "consistency_model": "#D95F02",
        "stable_consistency_model": "#E6AB02",
        "consistency_model_ft": "#D95F02",
        "stable_consistency_model_ft": "#E6AB02",

        # Other
        "ode": "#A6761D",
        "sde": "#FFD92F",
        "sde-pc": "#E6AB02",
        "MCMC": "#000000",
    }
    return {"visualization": {"colors": colors}}

SAMPLER_STYLES = {'ode': '-', 'sde': '--', 'sde-pc': '-.', 'MCMC': '-'}

def _model_name(k):
    if k.endswith('ft'):
        k = k[:-2]
    names = {
        "flow_matching":r"Flow Matching$\,$",
        "ot_flow_matching":r"Flow Matching (OT)$\,$",
        "flow_matching_edm":r"Flow Matching ($\rho = -0.6$)$\,$",
        "consistency_model":r"Discrete Consistency Model$\,$",
        "stable_consistency_model":r"Continuous Consistency Model$\,$",
        "diffusion_edm_vp":r"VP EDM$\,$",
        "diffusion_edm_ve":r"VE EDM$\,$",
        "diffusion_edm_vp_ema":r"VP EDM EMA$\,$",
        "diffusion_cosine_F":r"Cosine $\boldsymbol{F}$-pred.$\,$",
        "diffusion_cosine_v":r"Cosine $\boldsymbol{v}$-pred.$\,$",
        "diffusion_cosine_noise":r"Cosine $\boldsymbol{\epsilon}$-pred.$\,$",
        "MCMC":"MCMC",
    }
    return names.get(k, k.replace("_"," ").title())

def _sampler_name(k):
    names = {'ode':'(ODE RK45)', 'sde':'(SDE Euler)', 'sde-pc':'(SDE PC Euler)', 'MCMC': ''}
    return names.get(k, k.upper())

def _prep_metrics(df):
    d = df.copy()
    if 'posterior_contraction' in d: d['Posterior Contraction'] = d['posterior_contraction']
    if 'c2st' in d: d['c2st'] = 1 - np.abs(0.5 - d['c2st'])
    if 'nrmse' in d: d['NRMSE'] = 1 - np.minimum(d['nrmse'], 1)  # noise prediction explodes sometimes
    if 'posterior_calibration_error' in d: d['Calibration Error'] = 1 - 2*d['posterior_calibration_error']
    if 'posterior_tarp' in d: d['TARP'] = 1 - np.abs(d['posterior_tarp'])
    return d

def _angles(n):
    a = [i / float(n) * 2 * np.pi for i in range(n)]
    return a + a[:1]

def _group_by_method(df, key):
    key = key[:-2] if key.endswith('ft') else key
    flow = ['flow_matching','ot_flow_matching','flow_matching_edm']
    cons = ['consistency_model','stable_consistency_model']
    diff = ['diffusion_edm_vp','diffusion_edm_ve','diffusion_edm_vp_ema',
            'diffusion_cosine_F','diffusion_cosine_v','diffusion_cosine_noise']
    if key == 'flow': return {'Flow Methods': df[df['model'].isin(flow)]}
    if key == 'consistency': return {'Consistency Methods': df[df['model'].isin(cons)]}
    if key == 'diffusion': return {'Diffusion Methods': df[df['model'].isin(diff)]}
    if key == 'flow_consistency': return {'Overview': df[df['model'].isin(flow+cons)]}
    if key == 'overview': return {'Overview': df[df['model'].isin(['MCMC','flow_matching_edm','diffusion_cosine_v','consistency_model'])]}
    if key == 'edm': return {'EDM': df[df['model'].isin(['diffusion_edm_vp','diffusion_edm_ve'])]}
    if key == 'cosine': return {'Cosine': df[df['model'].isin(['diffusion_cosine_F','diffusion_cosine_v','diffusion_cosine_noise'])]}
    return {}


def _plot_one(ax, df, colors, alpha, include_sampler, sampler_filter=None, plot_shade=False):
    metrics = [m for m in METRICS if m in df.columns]
    if not metrics:
        return
    ang = _angles(len(metrics))

    def ok_sampler(s):
        if sampler_filter is None:
            return True
        if isinstance(sampler_filter, str):
            return s == sampler_filter
        if isinstance(sampler_filter, (list, tuple, set)):
            return s in sampler_filter
        return True

    # determine if all non MCMC traces share one sampler
    non_mcmc_samplers = set()
    if include_sampler and 'sampler' in df.columns:
        for model in df['model'].unique():
            if model == 'MCMC':
                continue
            for sp in df['sampler'].unique():
                if not ok_sampler(sp):
                    continue
                sub = df[(df['model'] == model) & (df['sampler'] == sp)]
                if not sub.empty:
                    non_mcmc_samplers.add(sp)
    uniform_sampler = (len(non_mcmc_samplers) == 1)

    # plotting
    for model in df['model'].unique() if 'model' in df.columns else ['All']:
        if (include_sampler and 'sampler' in df.columns) or model == 'MCMC':
            for sp in df['sampler'].unique() if 'sampler' in df.columns else ['']:
                if not ok_sampler(sp):
                    continue
                if model == 'MCMC':
                    sub = df[df['model'] == model]
                else:
                    sub = df[(df['model'] == model) & (df['sampler'] == sp)]
                if sub.empty:
                    continue

                vals = [float(sub[m].mean()) for m in metrics]
                v = vals + vals[:1]

                # legend label rules
                if model == 'MCMC':
                    label = _model_name(model)            # never show sampler
                else:
                    if include_sampler and not uniform_sampler:
                        label = f"{_model_name(model)} {_sampler_name(sp)}"
                    else:
                        label = _model_name(model)

                ax.plot(
                    ang, v,
                    label=label,
                    color=colors.get(model, "#000000"),
                    linestyle=SAMPLER_STYLES.get(sp, '-') if model != 'MCMC' else '-',
                    marker='o', linewidth=2, alpha=alpha, markersize=4
                )
                if plot_shade:
                    # shade under curve
                    ax.fill(ang, v, color=colors.get(model, "#000000"), alpha=alpha * 0.25)
        else:
            sub = df[df['model'] == model] if model != 'All' else df
            if sub.empty:
                continue
            vals = [float(sub[m].mean()) for m in metrics]
            v = vals + vals[:1]
            ax.plot(
                ang, v,
                label=_model_name(model) if model != 'All' else 'All',
                color=colors.get(model, "#000000"),
                linestyle='-', marker='o', linewidth=2, alpha=alpha, markersize=4
            )
            if plot_shade:
                ax.fill(ang, v, color=colors.get(model, "#000000"), alpha=alpha * 0.25)

    # axes and labels
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels([])
    ax.set_ylim(0, 1)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='y', labelsize=10, pad=-12)
    ax.grid(True)
    # Keep labels centered and readable.
    for i, m in enumerate(metrics):
        theta = ang[i]
        deg = np.degrees(theta) % 360

        # Keep text upright: flip orientation on left half of the plot.
        rotation = deg - 90
        if 180 < deg < 360:
            rotation += 180

        ax.text(
            theta,
            1.17,
            METRIC_LABELS.get(m, m),
            ha="center",
            va="center",
            size=12,
            color="black",
            rotation=rotation,
            rotation_mode="anchor",
        )


def plot_model_comparison_radar(
    metrics_df,
    save_path=None,
    alpha=0.7,
    title=None,
    group='overview',
    sampler='all',           # 'all', 'ode', ['ode','MCMC'], etc.
    include_sampler=True,
):
    cfg = _create_model_config()
    d = _prep_metrics(metrics_df)
    groups = _group_by_method(d, group)

    for name, data in groups.items():
        if data.empty:
            print(f'No metrics for {name}')
            continue
        fig, ax = plt.subplots(figsize=(5,4), subplot_kw=dict(polar=True), layout='constrained')
        sampler_filter = None if sampler == 'all' else sampler
        _plot_one(ax, data, cfg["visualization"]["colors"], alpha, include_sampler, sampler_filter)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0),
                       ncol=min(len(handles), 2), fontsize=12)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fname = (title or name).lower().replace(' ','_') if (title or name) else group
            fig.savefig(os.path.join(save_path, f"model_comparison_{fname}.pdf"), bbox_inches='tight')
        plt.show()


def plot_model_comparison_grid(
    metrics_df,
    save_path=None,
    alpha=0.7,
    panels=None,
    plot_shade=False
):

    cfg = _create_model_config()
    d = _prep_metrics(metrics_df)

    if panels is None:
        panels = [
            ('overview', 'Overview', 'ode'),
            #('flow_consistency', 'Flow and Consistency', 'ode'),
            ('cosine', 'Cosine', ['ode']),
            ('edm', 'EDM', ['ode']),
        ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4),
                             subplot_kw=dict(polar=True),
                             layout='constrained')
    axes = axes.ravel()

    handles = []
    labels = []
    for ax, (grp_key, panel_title, sampler) in zip(axes, panels):
        groups = _group_by_method(d, grp_key)
        data = next(iter(groups.values())) if groups else None
        if data is None or data.empty:
            ax.set_axis_off()
            continue
        _plot_one(ax, data, cfg["visualization"]["colors"], alpha,
                  include_sampler=True, sampler_filter=(None if sampler == 'all' else sampler),
                  plot_shade=plot_shade)

        handles += ax.get_legend_handles_labels()[0]
        labels += ax.get_legend_handles_labels()[1]
    # unique handles and labels
    unique = dict()
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    handles = list(unique.values())
    fig.legend(handles=handles,
        loc='lower center',
                  bbox_to_anchor=(0.5, -0.17),
                  ncol=4,
                  fontsize=12,
                  frameon=False,
                  handleheight=1.6,  # fix symbol line height
                  handletextpad=0.8,  # consistent padding
                  labelspacing=0.2,  # fixed vertical spacing between rows
    )

    if save_path:
        fig.savefig(os.path.join(save_path, "model_comparison_grid.pdf"), bbox_inches='tight')
    plt.show()
