import marimo

__generated_with = "0.19.4"
app = marimo.App(
    app_title="Tutorial: Simulation-Based Inference using Diffusion Models",
    css_file="",
    auto_download=["html"],
)

with app.setup:
    import marimo as mo

    import os
    os.environ["KERAS_BACKEND"] = "jax"


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
    ## What you will build in this tutorial

    We will train neural posterior approximators for a simple **inverse kinematics** problem.
    The goal is: given a 2D end position $\mathbf{x}$, infer the unknown arm configuration $\boldsymbol{\theta}$.

    Concretely, you will see how to:

    - define a *prior* over parameters and a *simulator* producing synthetic observations,
    - generate a training dataset by forward simulation,
    - train an amortized inference model using **diffusion-based SBI** in [BayesFlow](https://bayesflow.org),
    - draw approximate posterior samples for a new observation,
    - compare diffusion models to *Flow Matching* and *Consistency Models*,
    - make *post-hoc modifications* to inference via guided sampling.
    """)
    return


@app.cell
def _():
    import logging

    import numpy as np
    import matplotlib.pyplot as plt
    import bayesflow as bf
    import keras
    from tqdm import tqdm

    from intro_example.inverse_kinematics import InverseKinematicsModel

    logging.getLogger("bayesflow").setLevel(logging.ERROR)
    np.random.seed(42)
    return InverseKinematicsModel, bf, keras, np, plt


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Simulation-Based Inference (SBI)

    In Bayesian inference we want the **posterior distribution**
    $p(\boldsymbol\theta \mid \mathbf{x})$: a probability distribution over parameters
    $\boldsymbol\theta$ given observed data $\mathbf{x}$.

    A standard Bayesian workflow needs the likelihood $p(\mathbf{x}\mid \boldsymbol\theta)$:
    $p(\boldsymbol\theta\mid \mathbf{x}) \propto p(\mathbf{x}\mid\boldsymbol\theta)\,p(\boldsymbol\theta)$.

    However, in many scientific applications the likelihood is **intractable**:
    we cannot evaluate it pointwise, but we *can* simulate realistic data from the model.
    This is the core setting for **simulation-based inference (SBI)**.
    But also if the likelihood is tractable, SBI can be applied and be useful.

    ### SBI in one sentence

    If we can sample from $p(\mathbf{x}\mid \boldsymbol\theta) p(\boldsymbol\theta)$, we can train a neural network
    to approximate $p(\boldsymbol\theta \mid \mathbf{x})$ directly.

    ### The learning problem behind SBI

    We generate many simulated training pairs:
    $(\boldsymbol\theta_i, \mathbf{x}_i) \sim p(\boldsymbol\theta)p(\mathbf{x}\mid\boldsymbol\theta)$,
    and train a conditional model to learn the mapping
    $\mathbf{x} \mapsto p(\boldsymbol\theta\mid \mathbf{x})$ (or something related to that mapping).

    After training, inference for a new observation $\mathbf{x}_{\mathrm{obs}}$
    becomes a fast forward-pass procedure:
    $\boldsymbol\theta^{(1)},\dots,\boldsymbol\theta^{(N)} \sim q_\phi(\boldsymbol\theta\mid \mathbf{x}_{\mathrm{obs}})$,
    where $q_\phi$ is the learned posterior approximation.

    In the next cells we define a simple simulator (inverse kinematics) and build a [BayesFlow](https://bayesflow.org) workflow around it.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example model: [inverse kinematics](https://arxiv.org/pdf/2101.10763.pdf)

    We consider a simple 3-segment planar robot arm with unknown configuration:
    - one scalar **height offset**,
    - three **joint angles**.

    The simulator maps parameters $\boldsymbol\theta= (h, \alpha_1, \alpha_2, \alpha_3)$ to a 2D position $\mathbf{x}=(x_1, x_2)$.

    **Important:** this is a deliberately *non-identifiable* setting:
    different angle combinations can yield very similar end positions.
    That makes the posterior naturally **multimodal** and therefore a good stress test
    for flexible inference methods such as diffusion models.
    """)
    return


@app.cell
def _(np):
    # Inverse Kinematics
    def observation_model(
        parameters
    ) -> np.ndarray:
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
def _(np):
    def prior():
        """
        Generates a random draw from a 4-dimensional Gaussian prior distribution with a
        spherical convariance matrix. The parameters represent a robot's arm 
        configuration, with the first parameter indicating the arm's height and the 
        remaining three are angles.

        Returns
        -------
        params : A single draw from the 4-dimensional Gaussian prior.
        """
        scales = np.array([0.25, 0.5, 0.5, 0.5])
        prior_samples = np.random.normal(loc=0, scale=scales)
        return dict(parameters=prior_samples)
    return (prior,)


@app.cell
def _(bf, observation_model, prior):
    # we merge prior and observation model into a simulator
    simulator = bf.make_simulator([prior, observation_model])

    # now we create the simulator and generate training data
    n_simulations = 10000
    training_data = simulator.sample(n_simulations)
    prior_samples = simulator.sample(1000)['parameters']

    print(f"Generated {n_simulations} simulations")
    print(f"Data shape: {training_data['observables'].shape}")
    print(f"Parameter shape: {training_data['parameters'].shape}")
    return prior_samples, simulator, training_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The dataset now contains:
    - `parameters`: samples from the prior (our *ground truth* parameters),
    - `observables`: corresponding simulated end positions.

    This is the only supervision signal used for training: there are no data beyond simulations.

    Next we visualize the prior samples and simulations to get an intuition about the scale and variability of each parameter.
    """)
    return


@app.cell(hide_code=True)
def _(plt, training_data, variable_names_nice):
    def plot_params_kinematic(params, params2=None):
        _, _ax = plt.subplots(1, 4, sharex=True, sharey=True, 
                               layout='constrained', figsize=(10, 2))
        for a_i, (a, name) in enumerate(zip(_ax, variable_names_nice)):
            if params2 is not None:
                a.hist(params2[:, a_i], density=True, color='black', alpha=.5)
            a.hist(params[:, a_i], density=True, color='teal')
            a.set_xlabel(name)
        _ax[0].set_ylabel("Density")
        _ax[0].set_ylim(0, 1.6)
        _ax[0].set_xlim(-4, 4)
        plt.show()
        return

    plot_params_kinematic(training_data['parameters'])
    return


@app.cell(hide_code=True)
def _(InverseKinematicsModel, bf, np, plt, simulator):
    np.random.seed(26)
    adapter_plot = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .rename("parameters",  "inference_variables")
        .rename("observables", "inference_conditions")
    )
    _training_data = simulator.sample(3)
    _prior_samples = adapter_plot.forward(_training_data)

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
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Amortized Bayesian Inference

    SBI is typically used in an **amortized** way, meaning we train a neural inference model once
    and then reuse it for many observations.

    ### Why amortization matters

    Traditional Bayesian methods (e.g. MCMC) solve one inference problem at a time:
    - new observation $\mathbf{x}_{\mathrm{obs}}$  → rerun MCMC
    - computational cost grows linearly with the number of datasets.

    With amortized inference we instead learn a global conditional model:
    $q_\phi(\boldsymbol\theta\mid\mathbf{x}) \approx p(\boldsymbol\theta\mid\mathbf{x})$.


    After training, posterior sampling for a new observation is fast and convenient:
    - no tuning of samplers,
    - no burn-in,
    - no repeated optimization.

    ### A note on the “amortization gap”

    The network must generalize to new $\mathbf{x}$ values across the full simulated data space.
    If simulations do not cover the region of interest, the approximation may degrade.
    In practice this is addressed by better priors, more simulations, or robust inference methods.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### BayesFlow workflow structure

    BayesFlow organizes SBI pipelines into modular components:

    1. **Simulator**: produces synthetic pairs $(\boldsymbol\theta, \mathbf{x})$
    2. **Adapter**: converts simulator dictionaries into a consistent training format for neural networks
    3. **Inference network**: learns the posterior approximation
    4. **Workflow**: combines everything into `fit(...)` and `sample(...)` calls

    In this tutorial we keep the *summary network* set to `None`, because our observation is already low-dimensional (2D).
    For more complex data (time series, images, sets), BayesFlow can learn summaries automatically via neural encoders.
    """)
    return


@app.cell
def _(bf, simulator, training_data):
    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .rename("parameters", "inference_variables")
        .rename("observables", "inference_conditions")
    )

    # Create a simple workflow
    workflow_kinematics_example = bf.BasicWorkflow(
        adapter=adapter,
        simulator=simulator,
        summary_network=None,
        inference_network=bf.networks.DiffusionModel()
    )

    # let's only train for a short amount of time
    n_epochs_short = 10
    workflow_kinematics_example.fit_offline(
        training_data,
        epochs=n_epochs_short,
        batch_size=128
    )

    print("✓ Amortized inference network trained!")
    return adapter, n_epochs_short, workflow_kinematics_example


@app.cell
def _(bf, np, prior_samples, variable_names_nice, workflow_kinematics_example):
    # Test amortized inference on new data
    obs = {"observables": np.array([[0, 1.5]])}

    # Fast inference
    _posterior_samples = workflow_kinematics_example.sample(
        conditions=obs,
        num_samples=1000
    )

    bf.diagnostics.pairs_posterior(
        estimates=_posterior_samples,
        priors=prior_samples,
        variable_names=variable_names_nice,
        height=1.75
    )
    return (obs,)


@app.cell(hide_code=True)
def _(n_epochs_short):
    mo.md(rf"""
    The model above was trained only briefly ({n_epochs_short} epochs) to demonstrate the workflow mechanics.
    Posterior quality may be rough at this stage.

    Next, we train longer and also compare different inference backends (which we will explain soon):
    - **Diffusion model** (high flexibility, iterative sampling)
    - **Flow matching** (simpler training scheme, iterative sampling)
    - **Consistency model** (designed for very fast sampling)
    """)
    return


@app.cell(hide_code=True)
def _(adapter, bf, keras, simulator, training_data):
    workflows = []

    workflow_kinematics_diffusion = bf.BasicWorkflow(
        adapter=adapter,
        simulator=simulator,
        inference_network=bf.networks.DiffusionModel(),
        checkpoint_filepath='intro_example/models',
        checkpoint_name='tutorial_diffusion',
        standardize='all'
    )
    workflows.append(workflow_kinematics_diffusion)

    workflow_kinematics_flow = bf.BasicWorkflow(
        adapter=adapter,
        simulator=simulator,
        inference_network=bf.networks.FlowMatching(),
        checkpoint_filepath='intro_example/models',
        checkpoint_name='tutorial_flow',
        standardize='all'
    )
    workflows.append(workflow_kinematics_flow)

    workflow_kinematics_consistency = bf.BasicWorkflow(
        adapter=adapter,
        simulator=simulator,
        inference_network=bf.experimental.StableConsistencyModel(),
        checkpoint_filepath='intro_example/models',
        checkpoint_name='tutorial_consistency',
        standardize='all'
    )
    workflows.append(workflow_kinematics_consistency)

    for _w, _name in zip(workflows, ['diffusion', 'flow', 'consistency']):
        _model_path = f'intro_example/models/tutorial_{_name}.keras'
        if os.path.exists(_model_path): # load if already trained
            _w.approximator = keras.models.load_model(
                f'intro_example/models/tutorial_{_name}.keras'
            )
        else:
            # train otherwise
            _w.fit_offline(
                training_data,
                epochs=100,
                batch_size=128
            )
    return workflow_kinematics_diffusion, workflows


@app.cell
def _(
    bf,
    obs,
    prior_samples,
    variable_names_nice,
    workflow_kinematics_diffusion,
):
    _posterior_samples = workflow_kinematics_diffusion.sample(
        conditions=obs,
        num_samples=1000
    )

    bf.diagnostics.pairs_posterior(
        estimates=_posterior_samples,
        priors=prior_samples,
        variable_names=variable_names_nice,
        height=1.75
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's apply the model not only to a single observation, but to multiple data set since we are in an amortized setting!

    This way, we can also assess the model's performance across a range of observations.
    BayesFlow provides default diagnostic plots to assess whether posterior samples are well-calibrated.
    Here we run quick checks on newly simulated test data.
    """)
    return


@app.cell
def _(bf, variable_names_nice, workflow_kinematics_diffusion):
    test_data = workflow_kinematics_diffusion.simulator.sample(100)

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
    mo.image(src="static/images/diffusion_model_review.jpg")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Forward process (noising)

    We start with a clean sample $\boldsymbol\theta_0$ (here: a parameter vector) and gradually add noise:

    $\boldsymbol\theta_t = \alpha_t\,\boldsymbol\theta_0 + \sigma_t\,\boldsymbol\epsilon, \quad \boldsymbol\epsilon \sim \mathcal{N}(0, I)$,

    where $t \in [0,1]$ is a continuous diffusion time.
    At $t \approx 1$, the distribution becomes close to pure noise.
    """)
    return


@app.cell(hide_code=True)
def _():
    t_slider_forward = mo.ui.slider(
            start=0.0,
            stop=1.0,
            step=0.05,
            value=0.8,
            label="Diffusion time t:",
            show_value=True
        )
    t_slider_forward
    return (t_slider_forward,)


@app.cell(hide_code=True)
def _(
    bf,
    keras,
    np,
    t_slider_forward,
    training_data,
    variable_names_nice,
    workflow_kinematics_diffusion,
):
    parameters_0 = training_data['parameters'][:1000]
    _t_current = t_slider_forward.value
    _log_snr = workflow_kinematics_diffusion.approximator.inference_network.noise_schedule.get_log_snr(_t_current, training=True)
    _alpha_t, _sigma_t = workflow_kinematics_diffusion.approximator.inference_network.noise_schedule.get_alpha_sigma(_log_snr)

    _noise = np.random.normal(size=parameters_0.shape)
    parameters_t = _alpha_t * parameters_0 + _sigma_t * _noise

    # Create visualization
    bf.diagnostics.pairs_posterior(
        estimates=keras.ops.convert_to_numpy(parameters_t),
        priors=parameters_0,
        variable_names=variable_names_nice,
        height=1.75,
        label=f'Parameters at t={_t_current}',
        show_single_legend=True,
        post_color='teal'
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Reverse process (denoising / sampling)

    The diffusion model learns how to move from noisy states back to clean samples.
    In practice, sampling is performed by solving a learned reverse-time stochastic differential equation (SDE)
    or an equivalent deterministic ODE.

    ### Diffusion models for SBI

    In simulation-based inference we want:
    $p(\boldsymbol\theta \mid \mathbf{x})$.

    A *conditional diffusion model* learns a denoising direction that depends on the observation $\mathbf{x}$.
    This provides a highly expressive posterior approximation, especially useful for:
    - multimodal posteriors,
    - high-dimensional parameters,
    - post-hoc modifications during inference.


    Below we use a SDE solver which integrates the learned reverse dynamics
    from $t=1$ (noise) down to $t=0$ (posterior samples).

    Important: Sampling is *iterative denoising*, not a single network forward pass.
    """)
    return


@app.cell(hide_code=True)
def _(keras, workflow_kinematics_diffusion):
    t_slider_backward = mo.ui.slider(
            start=0.0,
            stop=1.0,
            step=0.05,
            value=1.0,
            label="Diffusion time t:",
            show_value=True
        )

    priors = workflow_kinematics_diffusion.approximator.inference_network.base_distribution.sample(100)
    priors = workflow_kinematics_diffusion.approximator.standardize_layers["inference_variables"](priors, forward=False)  # to original space
    priors = keras.ops.convert_to_numpy(priors)

    t_slider_backward
    return priors, t_slider_backward


@app.cell(hide_code=True)
def _(
    bf,
    keras,
    obs,
    priors,
    t_slider_backward,
    variable_names_nice,
    workflow_kinematics_diffusion,
):
    estimated_parameters_t = workflow_kinematics_diffusion.sample(
        conditions=obs,
        num_samples=100,
        stop_time=t_slider_backward.value
    )
    estimated_parameters_t = keras.ops.convert_to_numpy(estimated_parameters_t['parameters'][0])

    bf.diagnostics.pairs_posterior(
        estimates=estimated_parameters_t,
        priors=priors,
        variable_names=variable_names_nice,
        height=1.75,
        label=f'Denoised Posterior at t={t_slider_backward.value}'
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Short Summary

    - Diffusion model learns score function: $\nabla_{\boldsymbol\theta} \log p(\boldsymbol\theta\mid\mathbf{x})$, the "direction" it which we need to solve the reverse SDE
    - Sample by starting from noise and iteratively denoising

    Related generative models can be considered a different *parameterization* of a **diffusion model**:
    - **flow matching**: we directly predict the vector field of the deterministic reverse path
    - **consistency models**: designed for very fast sampling by learning to jump directly to clean samples
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.image(src="static/images/fm_cm_visual.pdf")
    return


@app.cell(hide_code=True)
def _(InverseKinematicsModel, obs, plt, workflows):
    _, _ax = plt.subplots(1, 3, figsize=(10, 4),
                           subplot_kw=dict(box_aspect=0.9), squeeze=False,
                           layout='constrained')
    _ax = _ax.flatten()

    for _i, (_a, _w) in enumerate(zip(_ax, workflows)):
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
    There are lots of choices to make when designing a diffusion model, we explained and benchmarked them extensively in [Arruda et al. (2025)](https://arxiv.org/abs/2512.20685):
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.image(src="static/images/c2st_benchmark_boxplot_best.jpg")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Why are diffusion models so special for SBI?

    Diffusion-based SBI provides two particularly useful properties.

    ### 1) Flexibility for difficult posteriors

    Many SBI problems produce posteriors that are:
    - strongly non-Gaussian,
    - multimodal (multiple valid explanations),
    - highly correlated.

    Conditional diffusion models are well-suited to represent such distributions without strong parametric assumptions.

    ### 2) Score-based structure enables “post-hoc control”

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
    ### Adaption During Inference Time

    The inverse-kinematics posterior is typically multimodal: multiple arm configurations can match the same
    end-effector position. Here we steer sampling *during reverse diffusion* by adding the gradient of a
    differentiable "preference" term to the learned reverse dynamics.

    We use the $x$-coordinate of the "elbow" (after the first two segments) as a simple mode selector:
    - **Elbow-up**: prefer larger elbow y
    - **Elbow-down**: prefer smaller elbow y
    """)
    return


@app.cell
def _(keras):
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
            theta = workflow.approximator.standardize_layers["inference_variables"](z, forward=False)
            a1 = theta[..., 1]
            return sign * keras.ops.sin(a1)

        return c_elbow
    return (elbow_up_down_constraint,)


@app.cell(hide_code=True)
def _():
    # UI controls
    mode = mo.ui.radio(options=["elbow-up", "elbow-down"], value="elbow-up", label="Steering target:")
    strength = mo.ui.slider(start=0.0, stop=1, step=0.01, value=0, label="Guidance strength λ:", show_value=True)

    mo.hstack([mode, strength])
    return mode, strength


@app.cell(hide_code=True)
def _(
    InverseKinematicsModel,
    alpha,
    elbow_up_down_constraint,
    mode,
    obs,
    ops,
    plt,
    strength,
    workflow_kinematics_diffusion,
):
    # Draw samples with and without guidance for side-by-side comparison
    constraints = [elbow_up_down_constraint(
        workflow_kinematics_diffusion, target=str(mode.value)
    )]

    def scaling_function(t):
        log_snr = workflow_kinematics_diffusion.approximator.inference_network.noise_schedule.get_log_snr(t, training=False)
        _, sigma_t = workflow_kinematics_diffusion.approximator.inference_network.noise_schedule.get_alpha_sigma(log_snr)
        return ops.square(alpha) / ops.square(sigma_t)

    theta_unguided = workflow_kinematics_diffusion.sample(
         conditions=obs,
         num_samples=500,
    )
    theta_unguided = theta_unguided['parameters'][0]

    theta_guided = workflow_kinematics_diffusion.sample(
         conditions=obs,
         num_samples=500,
         constraint_guidance=dict(
             constraints=constraints, 
             guidance_strength=float(strength.value),
         )
    )
    theta_guided = theta_guided['parameters'][0]

    # Visualize effect on arm configurations 
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), subplot_kw=dict(box_aspect=1.0), layout="constrained")

    model_left = InverseKinematicsModel(linecolors=["teal"] * 3)   # unguided
    model_right = InverseKinematicsModel(linecolors=["teal"] * 3)  # guided

    model_left.update_plot_ax(
        ax[0],
        theta_unguided,
        obs["observables"][0, ::-1],
        exemplar_color="#e6e7eb",
    )
    model_right.update_plot_ax(
        ax[1],
        theta_guided,
        obs["observables"][0, ::-1],
        exemplar_color="#e6e7eb",
    )

    ax[0].set_title("Posterior samples")
    ax[1].set_title(f"Guided posterior samples ({mode.value}, λ={strength.value})")
    plt.show()

    # Also show parameter-pair plots for guided vs unguided
    #bf.diagnostics.pairs_posterior(
    #    estimates=theta_guided_np,
    #    priors=theta_unguided_np,
    #    variable_names=variable_names_nice,
    #    height=1.75,
    #    label='Guided Posterior'
    #)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Summary

    In this tutorial we implemented simulation-based inference (SBI) using BayesFlow on an inverse kinematics problem.

    **Main takeaways:**

    1. **SBI is likelihood-free Bayesian inference**:
       we avoid evaluating $p(\mathbf{x}\mid\boldsymbol\theta)$ and train purely from simulations.

    2. **Amortization makes inference cheap at test time**:
       after training, we can sample approximate posteriors for new observations instantly.

    3. **Diffusion models provide strong posterior expressiveness**:
       iterative denoising can represent complex and multimodal posteriors more reliably than many single-pass density models.

    4. **Diffusion models allow post-hoc modifications**:
       we can adapt sampling strategies or compose information from multiple sources at inference time.

    5. **BayesFlow workflows standardize SBI experiments**:
       simulator → adapter → inference network → training → posterior sampling.



    ### BayesFlow Workflow

    ```python
    # 1. Define simulator
    def simulator(batch_size):
        return {'parameters': θ, 'sim_data': x}

    # 2. Create workflow
    workflow = bf.BasicWorkflow(
        simulator=simulator,
        summary_network=bf.networks.SetTransformer(...),
        inference_network=bf.networks.DiffusionModel(...)
    )

    # 3. Train
    workflow.fit_online(epochs=100)

    # 4. Infer (amortized)
    posterior_samples = workflow.sample(conditions=new_data)
    ```

    **Recommended next steps:**
    - explore learned summary networks for higher-dimensional observations,
    - evaluate calibration and coverage using diagnostic tools,
    - experiment with faster samplers (consistency models) when runtime matters.

    **Further reading:**
    - Arruda et al. (2025): [Diffusion Models In Simulation-Based Inference: A Tutorial Review](https://bayesflow-org.github.io/diffusion-experiments/)
    - [BayesFlow Documentation](https://bayesflow.org)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Tutorial written by Jonas Arruda, January 2026.
    """)
    return


if __name__ == "__main__":
    app.run()
