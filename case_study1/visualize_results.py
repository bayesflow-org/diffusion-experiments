import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    import sbibm
    import sys

    BASE = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE.parent 
    sys.path.append(str(PROJECT_ROOT))

    from case_study1.model_settings_benchmark import SAMPLER_SETTINGS
    from case_study1.helper_visualize import plot_benchmark_results, plot_by_sampler, plot_by_model, plot_low_budget_c2st_by_problem, pareto_best_sampler, plot_benchmark_results_plotly

    return (
        BASE,
        SAMPLER_SETTINGS,
        np,
        pareto_best_sampler,
        pd,
        plot_benchmark_results,
        plot_benchmark_results_plotly,
        plot_by_model,
        plot_by_sampler,
        plot_low_budget_c2st_by_problem,
        plt,
        sbibm,
    )


@app.cell
def _(BASE, pd):
    # Load the dataset
    results = pd.read_csv(BASE / 'plots' / 'c2st_benchmark_results.csv')
    results.reset_index(inplace=True, drop=True)
    results
    return (results,)


@app.cell
def _(SAMPLER_SETTINGS):
    all_samplers= ['best', 'merge_problems'] + [k for k in SAMPLER_SETTINGS.keys()]
    SHOW_SAMPLER = all_samplers[0]
    print(SHOW_SAMPLER)
    return (SHOW_SAMPLER,)


@app.cell
def _(SHOW_SAMPLER, pareto_best_sampler, results):
    long_df = results.copy()
    long_df = long_df.rename(columns={'task': 'problem', 'c2st': 'c2st', 'c2st_std': 'std'})
    long_df_copy = long_df.copy()

    if SHOW_SAMPLER == 'best':
        long_df = long_df[
            (long_df["sampler"] != 'ode-euler-small') &
            (long_df["sampler"] != 'ode-euler-mini') &
            ~(long_df["sampler"].str.contains('joint'))
        ]
        long_df = pareto_best_sampler(long_df)
        long_df = long_df.sort_values(['problem', 'model'])
    elif SHOW_SAMPLER == 'merge_problems':
        long_df = (
            long_df[["model", "sampler", "c2st", "std", "time", "time_std"]]
            .groupby(["model", "sampler"])

            .mean()
        ).reset_index()
        long_df['problem'] = 'all'
    else:
        long_df = long_df[long_df['sampler'] == SHOW_SAMPLER]
    long_df = long_df.dropna()

    # remove too much info 
    long_df_reduced = long_df[
        ((long_df["model"].str.contains('diffusion')) |
        (long_df["model"].str.contains('consistency')) |
        (long_df["model"] == 'flow_matching') |
        (long_df["model"] == 'cot_flow_matching') |
        (long_df["model"] == 'flow_matching_edm')) & (
        (long_df["sampler"] != 'ode-euler-small') &
        (long_df["sampler"] != 'ode-euler-mini'))
    ].reset_index(drop=True)

    # only flow matching
    long_df_reduced_fm = long_df[
        long_df["model"].str.contains('flow_matching')
    ].reset_index(drop=True)
    long_df_fm = long_df_copy[
        long_df_copy["model"].str.contains('flow_matching')
    ].reset_index(drop=True)
    return long_df, long_df_fm, long_df_reduced, long_df_reduced_fm


@app.cell
def _(long_df):
    vp_edm_df = long_df[long_df['model'] == 'diffusion_edm_vp']
    vp_edm_df = vp_edm_df[['sampler', 'c2st', 'time']].sort_values(by='c2st')
    vp_edm_df.head()
    return


@app.cell
def _(long_df):
    fm_df = long_df[long_df['model'] == 'flow_matching']
    fm_df = fm_df[['sampler', 'c2st', 'time']].sort_values(by='c2st')
    fm_df.head()
    return


@app.cell
def _(BASE, pd, plot_benchmark_results_plotly):
    #plotly_df = long_df_reduced[long_df_reduced.model != 'diffusion_cosine_v_lw']
    #plotly_df.to_csv(BASE / 'plots' / f'plotly_df.csv', index=False)
    plotly_df = pd.read_csv(BASE / 'plots' / f'plotly_df.csv')

    plot_benchmark_results_plotly(
        plotly_df,
    )
    return


@app.cell
def _(
    BASE,
    SHOW_SAMPLER,
    long_df_reduced,
    plot_benchmark_results,
    plot_by_model,
    plot_by_sampler,
):
    plot_benchmark_results(
        long_df_reduced[long_df_reduced.model != 'diffusion_cosine_v_lw'],
        SHOW_SAMPLER,
        BASE / 'plots' / f"c2st_benchmark_boxplot_{SHOW_SAMPLER}.pdf"
    )

    plot_by_model(
        long_df_reduced,
        col='c2st',
        col_std='std',
        save_path=BASE / 'plots' / f"c2st_benchmark_boxplot_{SHOW_SAMPLER}.pdf"
    )

    #plot_by_model(
    #    long_df_reduced,
    #    col='time',
    #    col_std='time_std',
    #    save_path=BASE / 'plots' / f"time_benchmark_boxplot_{SHOW_SAMPLER}.pdf"
    #)

    plot_by_sampler(
        long_df_reduced,
        col='time',
        col_std='time_std',
        save_path=BASE / 'plots' / f"time_benchmark_boxplot_{SHOW_SAMPLER}.pdf"
    )
    return


@app.cell
def _(
    BASE,
    SHOW_SAMPLER,
    long_df_fm,
    long_df_reduced_fm,
    plot_by_sampler,
    plot_low_budget_c2st_by_problem,
):
    plot_by_sampler(
        long_df_reduced_fm,
        col='time',
        col_std='time_std',
        save_path=BASE / 'plots' / f"time_benchmark_boxplot_{SHOW_SAMPLER}_fm.pdf"
    )

    plot_low_budget_c2st_by_problem(
        long_df_fm,
        BASE / 'plots' / f"euler_benchmark_boxplot_fm.pdf"
    )
    return


@app.cell
def _():


    return


@app.cell
def _():
    import bayesflow as bf

    from model_settings_benchmark import load_model, MODELS

    return MODELS, bf, load_model


@app.cell
def _(MODELS, sbibm):
    model_i = 4
    task_name = sbibm.get_available_tasks()[0]
    task = sbibm.get_task(task_name)
    conf_tuple = list(MODELS.values())[model_i]
    model_name_test = list(MODELS.keys())[model_i]
    priors = task.get_prior()(1_000).numpy()
    print(model_name_test)
    print(task_name)
    storage = ''
    return conf_tuple, model_name_test, priors, storage, task, task_name


@app.cell
def _(conf_tuple, load_model, model_name_test, storage, task_name):
    workflow = load_model(conf_tuple=conf_tuple,
                          training_data=None,
                          simulator=None,
                          storage=storage,
                          problem_name=task_name, model_name=model_name_test,
                          use_ema=True)
    return (workflow,)


@app.cell
def _(np, task, task_name):
    test_data = {
        'observables': np.concatenate([task.get_observation(num_observation=i).numpy() 
                                       for i in range(1, 11)]),
        'parameters': np.concatenate([task.get_true_parameters(num_observation=i).numpy()
                                      for i in range(1, 11)])
    }

    if task_name == 'lotka_volterra':
        new_data = []
        for obs in test_data['observables']:
            new_data.append(np.array([obs[:10], obs[10:]]).T.flatten()[np.newaxis])
        test_data['observables'] = np.concatenate(new_data)
    return (test_data,)


@app.cell
def _(plt, test_data, workflow):
    diagnostics = workflow.plot_default_diagnostics(test_data=test_data)
    plt.show()
    return


@app.cell
def _(np, task, workflow):
    num_observation = 1
    observation = task.get_observation(num_observation=num_observation).numpy()
    observation = np.array([observation[0, :10], observation[0, 10:]]).T.flatten()[np.newaxis]
    reference_samples = task.get_reference_posterior_samples(num_observation=num_observation).numpy()[:1000]
    posterior_samples_dict = workflow.sample(conditions={'observables': observation}, num_samples=1_000)
    posterior_samples = posterior_samples_dict['parameters'][0]

    param_names = task.get_labels_parameters()
    true_params = task.get_true_parameters(num_observation=num_observation).numpy()[0]
    return param_names, posterior_samples, reference_samples, true_params


@app.cell
def _(bf, param_names, plt, posterior_samples, priors, true_params):
    bf.diagnostics.pairs_posterior(
        estimates=posterior_samples,
        priors=priors,
        targets=true_params,
        variable_names=param_names
    )
    plt.show()
    return


@app.cell
def _(bf, param_names, plt, priors, reference_samples, true_params):
    bf.diagnostics.pairs_posterior(
        estimates=reference_samples,
        priors=priors,
        targets=true_params,
        variable_names=param_names
    )
    plt.show()
    return


if __name__ == "__main__":
    app.run()
