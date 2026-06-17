import marimo

__generated_with = "0.23.9"
app = marimo.App(
    app_title="Tutorial: Simulation-Based Inference using Diffusion Models",
    css_file="",
    auto_download=["html"],
)

with app.setup(hide_code=True):
    import marimo as mo

    import os
    os.environ["KERAS_BACKEND"] = "jax"

    import logging
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import keras

    from intro_example.inverse_kinematics import InverseKinematicsModel

    BASE = Path(__file__).resolve().parent
    logging.getLogger("bayesflow").setLevel(logging.ERROR)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Diffusion Models In Simulation-Based Inference: A Tutorial

    This notebook introduces answers the following questions:

    1. **What is simulation-based inference (SBI)?**
    2. **What are diffusion models?**
    3. **Why are diffusion models so special for SBI?**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Let's start with an example inference problem: [Inverse Kinematics](https://arxiv.org/pdf/2101.10763.pdf)
    """)
    return


@app.cell(hide_code=True)
def _(bf, simulator):
    np.random.seed(26)
    _adapter_plot = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .rename("parameters",  "inference_variables")
        .rename("observables", "inference_conditions")
    )
    _training_data = simulator.sample(3)
    _prior_samples = _adapter_plot.forward(_training_data)

    _, _ax = plt.subplots(1, 3, figsize=(10, 4),
                           subplot_kw=dict(box_aspect=0.9), squeeze=False,
                           layout='constrained')
    _ax = _ax.flatten()

    _m = InverseKinematicsModel(linecolors=['#E7298A']*3)
    _m.update_plot_ax(_ax[0], 
                     _prior_samples["inference_variables"][0:1], 
                     _prior_samples["inference_conditions"][0][::-1], 
                     exemplar_color="#e6e7eb")
    _m.update_plot_ax(_ax[1], 
                     _prior_samples["inference_variables"][1:2], 
                     _prior_samples["inference_conditions"][1][::-1], 
                     exemplar_color="#e6e7eb")
    _m.update_plot_ax(_ax[2], 
                     _prior_samples["inference_variables"][2:3], 
                     _prior_samples["inference_conditions"][2][::-1], 
                     exemplar_color="#e6e7eb")
    _ax[0].set_title('Simulation 1')
    _ax[1].set_title('Simulation 2')
    _ax[2].set_title('Simulation 3')
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We consider a simple 3-segment planar robot arm with unknown configuration:
    - one scalar **height offset** $h$,
    - three **joint angles** $\alpha_1$, $\alpha_2$, $\alpha_3$.


    The goal is: given an end position $\red{\mathbf{x}}$ of a robot arm, infer the unknown arm configuration $\boldsymbol{\theta}=(h, \alpha_1, \alpha_2, \alpha_3)$.
    Hence, we want to learn a mapping $\mathbf{x} \mapsto \boldsymbol\theta= (h, \alpha_1, \alpha_2, \alpha_3)$.

    **Important:** this is a deliberately *non-identifiable* setting:
    different angle combinations can yield very similar end positions.
    That makes the inference task naturally **multimodal** and therefore a good stress test
    for flexible inference methods such as diffusion models.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How to solve this problem?

    We will train a neural network to approximate the posterior distribution for the simple **inverse kinematics** problem.

    Concretely, we will see how to:
    - define a *prior* over parameters and a *simulator* producing synthetic observations,
    - generate a training dataset by forward simulation,
    - train an amortized inference model using **diffusion-based SBI** in [BayesFlow](https://bayesflow.org),
    - draw approximate posterior samples for a new observation,
    - compare diffusion models to *flow matching* and *consistency models*,
    - make *post-hoc modifications* to inference via guided sampling.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Simulation-Based Inference (SBI)

    In Bayesian inference, we want to infer the **posterior distribution**
    $p(\boldsymbol\theta \mid \mathbf{x})$: a probability distribution over parameters
    $\boldsymbol\theta$ given observed data $\mathbf{x}$.

    A standard Bayesian workflow needs a prior $p(\boldsymbol\theta)$ and the likelihood $p(\mathbf{x}\mid \boldsymbol\theta)$ to be able to compute the posterior:

    $p(\boldsymbol\theta\mid \mathbf{x}) \propto p(\mathbf{x}\mid\boldsymbol\theta)\,p(\boldsymbol\theta)$.


    #### Key idea in **simulation-based inference (SBI)**:

    Often we can **simulate** realistic data from the model, without evaluating the likelihood density.


    ### SBI in one sentence

    If we can sample from $p(\mathbf{x}\mid \boldsymbol\theta) p(\boldsymbol\theta)$, we can train a neural network
    to approximate $p(\boldsymbol\theta \mid \mathbf{x})$ directly.


    ### Training a neural posterior estimator

    1. We generate many simulated training pairs:
    $(\boldsymbol\theta_i, \mathbf{x}_i) \sim p(\boldsymbol\theta)p(\mathbf{x}\mid\boldsymbol\theta)$
    2. We train a neural network to learn the mapping
    $\mathbf{x} \mapsto p(\boldsymbol\theta\mid \mathbf{x})$ (or something related to that mapping).

    After training, inference for a new observation $\mathbf{x}_{\mathrm{obs}}$
    becomes a fast forward-pass procedure:
    $\boldsymbol\theta^{(1)},\dots,\boldsymbol\theta^{(N)} \sim q_\phi(\boldsymbol\theta\mid \mathbf{x}_{\mathrm{obs}})$,
    where $q_\phi$ is the learned posterior approximation.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## SBI in [BayesFlow](https://bayesflow.org)

    BayesFlow is a flexible Python library for simulation-based inference with neural networks, including diffusion models.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.image(src="static/images/bf_landing_light.png", caption="BayesFlow: a Python library for simulation-based amortized Bayesian inference with neural networks. ")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    So let's train a neural posterior estimator! In BayesFlow this is just a few lines of code:


    ```python
    import bayesflow as bf

    # 1. Define prior and simulator
    def my_prior():
        return {'parameters': θ}

    def my_observation_model(parameters):
        return {'sim_data': x}

    simulator = bf.make_simulator([my_prior, my_observation_model])

    # 2. Create workflow
    workflow = bf.BasicWorkflow(
        simulator=simulator,
        inference_network=bf.networks.DiffusionModel(...)
    )

    # 3. Train
    workflow.fit_online(epochs=100)

    # 4. Infer (amortized)
    posterior_samples = workflow.sample(conditions=new_data)

    # 5. Diagnose
    ...
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How does this look for the [inverse kinematics](https://arxiv.org/pdf/2101.10763.pdf)?
    """)
    return


@app.cell
def _():
    import bayesflow as bf

    return (bf,)


@app.function
def prior() -> dict[str, np.ndarray]:
    """
    Generates a random draw from a 4-dimensional Gaussian prior distribution with a
    spherical covariance matrix. The parameters represent a robot's arm
    configuration, with the first parameter indicating the arm's height and the 
    remaining three are angles.

    Returns
    -------
    params : A single draw from the 4-dimensional Gaussian prior.
    """
    scales = np.array([0.25, 0.5, 0.5, 0.5])
    prior_samples = np.random.normal(loc=0, scale=scales)
    return dict(parameters=prior_samples)


@app.cell
def _():
    # Inverse Kinematics
    def observation_model(
        parameters
    ) -> dict[str, np.ndarray]:
        """
        Returns the 2D coordinates of a robot arm given parameter vector.
        The first parameter represents the arm's height and the remaining three
        correspond to angles.

        Reference: https://arxiv.org/pdf/2101.10763.pdf

        Parameters
        ----------
        parameters   : The four model parameters which will determine the coordinates

        Returns
        -------
        x : The 2D coordinates of the arm
        """
        height_arm, angle_1, angle_2, angle_3 = parameters
        # length of segments
        l1: float = 0.5
        l2: float = 0.5
        l3: float = 1.0

        # Determine 2D position
        x1 = l1 * np.sin(angle_1)
        x1 += l2 * np.sin(angle_1 + angle_2)
        x1 += l3 * np.sin(angle_1 + angle_2 + angle_3) + height_arm

        x2 = l1 * np.cos(angle_1)
        x2 += l2 * np.cos(angle_1 + angle_2)
        x2 += l3 * np.cos(angle_1 + angle_2 + angle_3)
        return dict(observables=np.array([x1, x2]))

    variable_names = ["height_arm", "angle_1", "angle_2", "angle_3"]
    variable_names_nice = [" ".join(v.title().split('_')) for v in variable_names]
    return observation_model, variable_names_nice


@app.cell
def _(bf, observation_model):
    # we merge prior and observation model into a simulator
    simulator = bf.make_simulator([prior, observation_model])

    # now we create the simulator and generate training data
    n_simulations = 10_000
    training_data = simulator.sample(n_simulations)

    print(f"Generated {n_simulations} simulations")
    print(f"Data shape: {training_data['observables'].shape}")
    print(f"Parameter shape: {training_data['parameters'].shape}")
    return simulator, training_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The dataset now contains:
    - `parameters`: samples from the prior (our *ground truth* parameters),
    - `observables`: corresponding simulated end positions.

    This is the only supervision signal used for training: there are no data beyond simulations.

    Next, we visualize the prior samples and simulations.
    """)
    return


@app.cell(hide_code=True)
def _(bf, simulator, training_data, variable_names_nice):
    def plot_params_kinematic(params=None, params2=None):
        _, _ax = plt.subplots(1, 4, sharex=True, sharey=True, 
                               layout='constrained', figsize=(10, 2))
        for a_i, (a, name) in enumerate(zip(_ax, variable_names_nice)):
            if params2 is not None:
                a.hist(params2[:, a_i], density=True, color='black', alpha=.5)
            if params is not None:
                a.hist(params[:, a_i], density=True, color='#E7298A')
            a.set_xlabel(name)
        _ax[0].set_ylabel("Density")
        _ax[0].set_ylim(0, 1.6)
        _ax[0].set_xlim(-4, 4)
        plt.show()
        return

    plot_params_kinematic(params2=training_data['parameters'])

    _adapter_plot = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .rename("parameters",  "inference_variables")
        .rename("observables", "inference_conditions")
    )
    _training_data = simulator.sample(3)
    _prior_samples = _adapter_plot.forward(_training_data)

    _, _ax = plt.subplots(1, 3, figsize=(10, 4),
                           subplot_kw=dict(box_aspect=0.9), squeeze=False,
                           layout='constrained')
    _ax = _ax.flatten()

    _m = InverseKinematicsModel(linecolors=['#E7298A']*3)
    _m.update_plot_ax(_ax[0], 
                     _prior_samples["inference_variables"][0:1], 
                     _prior_samples["inference_conditions"][0][::-1], 
                     exemplar_color="#e6e7eb")
    _m.update_plot_ax(_ax[1], 
                     _prior_samples["inference_variables"][1:2], 
                     _prior_samples["inference_conditions"][1][::-1], 
                     exemplar_color="#e6e7eb")
    _m.update_plot_ax(_ax[2], 
                     _prior_samples["inference_variables"][2:3], 
                     _prior_samples["inference_conditions"][2][::-1], 
                     exemplar_color="#e6e7eb")
    _ax[0].set_title('Random Simulation 1')
    _ax[1].set_title('Random Simulation 2')
    _ax[2].set_title('Random Simulation 3')
    plt.show()

    def plot_arm_posterior(posterior_samples, obs):
        _, _ax = plt.subplots(figsize=(3,3))

        _m = InverseKinematicsModel(
            linecolors=[['#E7298A'], ['#E7298A'], ['#E7298A']]
        )
        _m.update_plot_ax(_ax, 
            posterior_samples["parameters"][0], 
            obs['observables'][0, ::-1], 
            exemplar_color="#E7298A"
        )
        _ax.set_title('Arm Configurations')
        plt.show()

    return plot_arm_posterior, plot_params_kinematic


@app.cell
def _(bf, simulator, training_data):
    # we tell the workflow what the input to our neural network is and how the target is called
    adapter = (
        bf.adapters.Adapter()
        .to_array() # we could do more complex transformations here
        .convert_dtype("float64", "float32")
        .rename("parameters", "inference_variables")
        .rename("observables", "inference_conditions")
    )

    # Create a simple workflow
    workflow_kinematics_example = bf.BasicWorkflow(
        adapter=adapter,
        simulator=simulator,
        inference_network=bf.networks.DiffusionModel()
    )

    # let's only train for a short amount of time
    n_epochs_short = 10
    history_example = workflow_kinematics_example.fit_offline(
        training_data,
        epochs=10,
        batch_size=128
    )
    return adapter, n_epochs_short, workflow_kinematics_example


@app.cell
def _(plot_params_kinematic, training_data, workflow_kinematics_example):
    # Test amortized inference on new data
    obs = {"observables": np.array([[0, 1.5]])}  # could be real data

    # Fast inference
    _posterior_samples = workflow_kinematics_example.sample(
        conditions=obs,
        num_samples=1000
    )

    # plot marginals
    plot_params_kinematic(
        _posterior_samples['parameters'][0],
        training_data['parameters']
    )
    return (obs,)


@app.cell(hide_code=True)
def _(make_cell_depended, n_epochs_short):
    mo.md(rf"""
    The model above was trained only briefly ({n_epochs_short} epochs) to demonstrate the workflow mechanics.
    Posterior quality is still rough at this stage.
    So let's train a bit longer: 200 epochs! This will take around {make_cell_depended} minutes.
    """)
    return


@app.cell
def _(adapter, bf, simulator, training_data):
    workflow_kinematics_diffusion = bf.BasicWorkflow(
        adapter=adapter,
        simulator=simulator,
        inference_network=bf.networks.DiffusionModel(),
        standardize='inference_conditions'
    )

    history = workflow_kinematics_diffusion.fit_offline(
        training_data,
        epochs=200,
        batch_size=128,
        verbose=0
    )
    return (workflow_kinematics_diffusion,)


@app.cell(hide_code=True)
def _():
    make_cell_depended = 2
    mo.md(r"""
    ### Amortized Bayesian Inference

    SBI is typically used in an **amortized** way, meaning we train a neural inference model once
    and then reuse it for many observations.

    #### Why amortization matters

    Traditional Bayesian methods (e.g., MCMC) solve one inference problem at a time:
    - new observation $\mathbf{x}_{\mathrm{obs}}$  → rerun MCMC
    - computational cost grows linearly with the number of datasets.

    With amortized inference we instead learn a global conditional model:
    $q_\phi(\boldsymbol\theta\mid\mathbf{x}) \approx p(\boldsymbol\theta\mid\mathbf{x})$.


    After training, posterior sampling for a new observation is fast and convenient:
    We can **solve many inference problems** with only one neural network!

    #### A note on the “amortization gap”

    The network must generalize to new $\mathbf{x}$ values across the full simulated data space.
    If simulations do not cover the region of interest, the approximation may degrade.
    In practice this is addressed by better priors, more simulations, or robust inference methods.
    """)
    return (make_cell_depended,)


@app.cell
def _(
    obs,
    plot_arm_posterior,
    plot_params_kinematic,
    training_data,
    workflow_kinematics_diffusion,
):
    # Fast inference
    posterior_samples_single = workflow_kinematics_diffusion.sample(
        conditions=obs,
        num_samples=1000
    )

    plot_params_kinematic(
        posterior_samples_single['parameters'][0],
        training_data['parameters']
    )

    # simulate the posterior samples and plot them
    plot_arm_posterior(posterior_samples_single, obs)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    So something seems to have been learned. We get a nice bimodal posterior!

    But are we sure that this is the correct posterior?
    Let's do amortize inference on 100 datasets!
    """)
    return


@app.cell
def _(bf, simulator, variable_names_nice, workflow_kinematics_diffusion):
    test_data = simulator.sample(100)

    posterior_samples_test_data = workflow_kinematics_diffusion.sample(
        conditions=test_data,
        num_samples=100,
    )

    bf.diagnostics.plots.coverage(
        estimates=posterior_samples_test_data,
        targets=test_data,
        variable_names=variable_names_nice
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. What are Diffusion Models?

    Diffusion models are generative models that create samples by *iteratively denoising* random noise.
    """)
    return


@app.cell(hide_code=True)
def _():
    t_slider_backward = mo.ui.slider(
            start=0.0,
            stop=1.0,
            step=0.05,
            value=1,
            label="Diffusion time t:",
            show_value=True,
            debounce=True
        )

    t_slider_backward
    return (t_slider_backward,)


@app.cell(hide_code=True)
def _(
    obs,
    plot_params_kinematic,
    t_slider_backward,
    workflow_kinematics_diffusion,
):
    logging.getLogger("bayesflow").setLevel(logging.ERROR)

    estimated_parameters_t = workflow_kinematics_diffusion.sample(
        conditions=obs,
        num_samples=100,
        stop_time=t_slider_backward.value
    )

    plot_params_kinematic(
        estimated_parameters_t['parameters'][0]
    )

    print(f'Denoised Posterior at t={t_slider_backward.value}')
    return


@app.cell(hide_code=True)
def _():
    mo.image(src="static/images/diffusion_model_review.jpg", caption="Diffusion Models In Simulation-Based Inference: A Tutorial Review")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Forward process (Training)

    We start with a clean sample $\boldsymbol\theta_0$ (here: a parameter vector) and gradually add noise:

    $\boldsymbol\theta_t = \alpha_t\,\boldsymbol\theta_0 + \sigma_t\,\boldsymbol\epsilon, \quad \boldsymbol\epsilon \sim \mathcal{N}(0, I)$,

    where $t \in [0,1]$ is a continuous diffusion time.
    At $t \approx 1$, the distribution becomes close to pure noise.
    Different choices of the *noise schedule* $\alpha_t$ and $\sigma_t$ determine how fast noise is added over time and have an impact on the performance later ([Arruda et al. (2025)](https://arxiv.org/abs/2512.20685)).
    """)
    return


@app.cell(hide_code=True)
def _():
    t_slider_forward = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.0,
        label="Diffusion time t:",
        show_value=True,
        debounce=True
        )
    t_slider_forward
    return (t_slider_forward,)


@app.cell(hide_code=True)
def _(
    plot_params_kinematic,
    t_slider_backward,
    t_slider_forward,
    training_data,
    workflow_kinematics_diffusion,
):
    parameters_0 = training_data['parameters'][:1000]
    _t_current = t_slider_forward.value
    _log_snr = workflow_kinematics_diffusion.approximator.inference_network.noise_schedule.get_log_snr(_t_current, training=True)
    _alpha_t, _sigma_t = workflow_kinematics_diffusion.approximator.inference_network.noise_schedule.get_alpha_sigma(_log_snr)

    _noise = np.random.normal(size=parameters_0.shape)
    parameters_t = _alpha_t * parameters_0 + _sigma_t * _noise

    plot_params_kinematic(
        parameters_t
    )

    print(f'Parameters at t={t_slider_backward.value}')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    So what does the network actually learn?
    It gets a noisy parameter $\boldsymbol\theta_t$ and the corresponding simulation $\mathbf{x}$ and predicts the score $\nabla_{\boldsymbol\theta_t}\!\log p(\boldsymbol\theta_t\mid\mathbf{x})$.

    What is this score? It is the direction in which we need to move $\boldsymbol\theta_t$ to increase its probability under the posterior.
    And it can be analytically computed from the noise schedule!
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.image(src="static/images/score_visual.jpg", width=300, caption="Visualization of the score.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Reverse process (Inference)

    The diffusion model learns how to move from noisy samples back to clean samples.
    In practice, sampling is performed by solving a learned reverse-time stochastic differential equation (SDE) or an equivalent deterministic ODE.

    And how? The reverse SDE/ODE is purely defined in terms of the noise schedule and the learned score!
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Short Summary

    - Diffusion model learns a score function: $\nabla_{\boldsymbol\theta} \log p(\boldsymbol\theta\mid\mathbf{x})$, the "direction" it which we need to solve the reverse SDE
    - Sample by starting from noise and iteratively denoising

    This provides a highly expressive posterior approximation, especially useful for:
    - multimodal posteriors,
    - high-dimensional parameters,
    - post-hoc modifications during inference.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## More generative models
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.image(src="static/images/fm_cm_visual.png", caption="Flow matching and consistency models as alternative parameterizations of diffusion models.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Related generative models can be considered a different *parameterization* of a **diffusion model**:
    - **Flow matching**: we directly predict the vector field of the deterministic reverse path
    - **Consistency models**: designed for very fast sampling by learning to jump directly to clean samples

    All of these models are available in BayesFlow. So let's load some trained models and compare their performance on our inverse kinematics problem!
    """)
    return


@app.cell
def _(adapter, bf, simulator, training_data, workflow_kinematics_diffusion):
    workflows = {}

    workflow_kinematics_flow = bf.BasicWorkflow(
        adapter=adapter,
        simulator=simulator,
        inference_network=bf.networks.FlowMatching(),
        checkpoint_filepath='intro_example/models',
        checkpoint_name='tutorial_flow_matching_model',
        standardize='inference_conditions'
    )
    workflows['flow_matching_model'] = workflow_kinematics_flow

    workflow_kinematics_consistency = bf.BasicWorkflow(
        adapter=adapter,
        simulator=simulator,
        inference_network=bf.networks.StableConsistencyModel(),
        checkpoint_filepath='intro_example/models',
        checkpoint_name='tutorial_consistency_model',
        standardize='inference_conditions'
    )
    workflows['consistency_model'] = workflow_kinematics_consistency

    for _name in workflows.keys():
        _model_path = f'intro_example/models/tutorial_{_name}.keras'
        if os.path.exists(_model_path): # load if already trained
            workflows[_name].approximator = keras.models.load_model(
                f'intro_example/models/tutorial_{_name}.keras'
            )
        else:
            # train otherwise
            workflows[_name].fit_offline(
                training_data,
                epochs=200,
                batch_size=128,
                verbose=0
            )

    workflows['diffusion_model'] = workflow_kinematics_diffusion
    return (workflows,)


@app.cell(hide_code=True)
def _(obs, workflows):
    _, _ax = plt.subplots(1, 3, figsize=(10, 4),
                           subplot_kw=dict(box_aspect=0.9), squeeze=False,
                           layout='constrained')
    _ax = _ax.flatten()

    workflows_list = [workflows['diffusion_model'], workflows['flow_matching_model'], workflows['consistency_model']]

    for _i, (_a, _w) in enumerate(zip(_ax, workflows_list)):
        posterior_samples = _w.sample(
            conditions=obs,
            num_samples=300
        )
        _m = InverseKinematicsModel(
            linecolors=[['#E7298A'], ['#1B9E77'], ['#E6AB02']][_i]*3
        )
        _m.update_plot_ax(_a, 
                         posterior_samples["parameters"][0], 
                         obs['observables'][0, ::-1], 
                         exemplar_color="#e6e7eb"
                         )

    _ax[0].set_title('Diffusion Model')
    _ax[1].set_title('Flow Matching')
    _ax[2].set_title('Consistency Model')
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    There are lots of choices to make when designing a diffusion model, e.g.:
    - Which noise schedule?
    - Diffusion model vs. other generative models?

    We explained and benchmarked them extensively in [Arruda et al. (2025)](https://arxiv.org/abs/2512.20685):
    """)
    return


@app.cell(hide_code=True)
def _():
    from case_study1.helper_visualize import plot_benchmark_results_plotly

    plotly_df = pd.read_csv(BASE / 'case_study1' / 'plots' / f'plotly_df.csv')
    plot_benchmark_results_plotly(
        plotly_df
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Why are diffusion models so special for SBI?

    Diffusion-based SBI provides two particularly useful properties.
    We already have seen that they can tackle difficult and high-dimensional posteriors.

    The real game changer is: score-based structure enables “post-hoc control”

    Diffusion models learn a conditional vector field (a *score/velocity*-like object) that drives denoising.
    This makes it possible to modify inference after training, for example:
    - introducing additional constraints at sampling time,
    - composing information from different sources.

    This idea forms the basis of *compositional inference*, which is an active research direction for building scalable [hierarchical SBI methods](https://arxiv.org/abs/2505.14429).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.image(src="static/images/adaptive_inference.jpg")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Adaptation During Inference Time

    The inverse-kinematics posterior is typically multimodal: multiple arm configurations can match the same
    end-position. Here, we steer sampling *during reverse diffusion* by adding the gradient of a
    differentiable "preference" term to the learned reverse dynamics.

    We use the first angle of the "elbow" as a simple selector:
    - **Elbow-up**
    - **Elbow-down**
    """)
    return


@app.function
def elbow_up_down_constraint(workflow, target="elbow-up"):
    """
    Constraint for guided diffusion: pick "elbow-up" or "elbow-down".

    The rule is always:
        constraint is satisfied  <=>  c(zt) <= 0

    - If target="elbow-up":
          c(zt) =  -sin(a1)    -> wants sin(a1) >= 0
    - If target="elbow-down":
          c(zt) = sin(a1)    -> wants sin(a1) <= 0
    """
    sign = -1.0 if target == "elbow-up" else 1.0

    def c_elbow(z):
        if 'inference_variables' in workflow.approximator.standardize_layers:
            z = workflow.approximator.standardize_layers["inference_variables"](
            z, forward=False
        )
        a1 = z[..., 1]
        return sign * keras.ops.sin(a1)

    return c_elbow


@app.cell(hide_code=True)
def _():
    # UI controls
    mode = mo.ui.radio(
        options=["elbow-up", "elbow-down"], 
        value="elbow-up", 
        label="Steering target:"
    )
    strength = mo.ui.slider(
        start=0.0, 
        stop=1,
        step=0.01, 
        value=0,
        label="Guidance strength λ:",
        show_value=True,
        debounce=True
    )

    mo.hstack([mode, strength])
    return mode, strength


@app.cell(hide_code=True)
def _(mode, obs, strength, workflow_kinematics_diffusion):
    # Draw samples with and without guidance for side-by-side comparison
    constraints = [elbow_up_down_constraint(
        workflow_kinematics_diffusion, target=str(mode.value)
    )]

    theta_unguided = workflow_kinematics_diffusion.sample(
         conditions=obs,
         num_samples=300,
    )
    theta_unguided = theta_unguided['parameters'][0]

    theta_guided = workflow_kinematics_diffusion.sample(
         conditions=obs,
         num_samples=300,
         guidance_kwargs=dict(
             constraints=constraints, 
             guidance_strength=float(strength.value),
         )
    )
    theta_guided = theta_guided['parameters'][0]

    # Visualize effect on arm configurations 
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), subplot_kw=dict(box_aspect=1.0), layout="constrained")

    model_left = InverseKinematicsModel(linecolors=["#E7298A"] * 3)   # unguided
    model_right = InverseKinematicsModel(linecolors=["#E7298A"] * 3)  # guided

    model_left.update_plot_ax(
        ax[0],
        theta_unguided,
        obs["observables"][0, ::-1],
        exemplar_color="#E7298A",
    )
    model_right.update_plot_ax(
        ax[1],
        theta_guided,
        obs["observables"][0, ::-1],
        exemplar_color="#E7298A",
    )

    ax[0].set_title("Posterior samples")
    ax[1].set_title(f"Guided posterior samples ({mode.value}, λ={strength.value})")
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary

    In this tutorial we implemented simulation-based inference (SBI) using BayesFlow on an inverse kinematics problem.

    **Main takeaways:**

    1. **SBI is likelihood-free Bayesian inference**:
       We avoid evaluating $p(\mathbf{x}\mid\boldsymbol\theta)$ and train purely from simulations.

    2. **Amortization makes inference cheap at test time**:
       After training, we can sample approximate posteriors for new observations instantly. Very useful for diagnostics!

    3. **Diffusion models provide strong posterior expressiveness**:
       Iterative denoising can represent complex and multimodal posteriors more reliably than many single-pass density models.

    4. **Diffusion models allow post-hoc intervention**:
       We can compose information from multiple sources or apply constraints at inference time.


    **Recommended next steps:**

    Check out [BayesFlow](https://bayesflow.org) for
    - exploring learned summary networks for higher-dimensional observations,
    - evaluating calibration and coverage using diagnostic tools, like simulation-based calibration.

    **Further reading:**

    [Diffusion Models In Simulation-Based Inference: A Tutorial Review](https://bayesflow-org.github.io/diffusion-experiments/)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.image(src="static/images/mail_to_qr.png", width=200, 
             caption="Reach out to me with questions or feedback.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Tutorial written by Jonas Arruda, February 2026.
    """)
    return


if __name__ == "__main__":
    app.run()
