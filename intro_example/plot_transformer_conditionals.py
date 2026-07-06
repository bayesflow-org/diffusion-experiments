# %%
# Second figure for the intro example: a single diffusion model with a
# transformer backbone, trained with masking, that answers several
# conditional/marginal queries after training. We visualize four scenarios:
#   1. guided posterior         sampling steered toward the "upper arm" (elbow-up)
#   2. masked conditional       one observation coordinate marked missing
#   3. masked target            the height parameter marginalized out
#   4. fixed target             the height parameter fixed to a known value
# See intro_example/Diffusion_Models.ipynb (cells 46-52) for the walkthrough.
import os
import pickle
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"

import matplotlib.pyplot as plt
import numpy as np
import bayesflow as bf
import keras

from inverse_kinematics import InverseKinematicsModel

BASE = Path(__file__).resolve().parent

EPOCHS = 200
BATCH_SIZE = 128
NUM_SAMPLES = 1000

# Transformer-backbone color
TRANSFORMER_COLOR = "#F768A1"

# Observation for position (0, 1.5) from https://arxiv.org/abs/2101.10763
obs = {"observables": np.array([[0.0, 1.5]], dtype=np.float32)}

# %%
# Same adapter as the rest of the intro example (see Diffusion_Models.ipynb).
adapter = (
    bf.adapters.Adapter()
    .to_array()
    .convert_dtype("float64", "float32")
    .rename("parameters", "inference_variables")
    .rename("observables", "inference_conditions")
)

# Same diffusion model as before, but with a transformer backbone and masking
# enabled during training.
workflow = bf.BasicWorkflow(
    adapter=adapter,
    inference_network=bf.networks.DiffusionModel(
        subnet="diffusion_transformer",
        drop_target_prob=0.3,   # sometimes condition on a parameter instead of inferring it
        drop_missing_prob=0.3,  # sometimes mark a parameter/observation as missing
    ),
    standardize="inference_conditions",
)

model_path = BASE / "models" / "diffusion_transformer_inverse_kinematics.keras"

if not model_path.exists():
    training_data = pickle.load(open(BASE / "models" / "inverse_kinematics_train_data.pkl", "rb"))
    workflow.fit_offline(training_data, epochs=EPOCHS, batch_size=BATCH_SIZE)
    workflow.approximator.save(model_path)
else:
    workflow.approximator = keras.models.load_model(model_path)

# %%
# 1. Guided posterior: steer sampling toward the "elbow-up" (upper arm raised)
def elbow_up_constraint(z):
    return -keras.ops.sin(z[..., 1])

posterior_full = workflow.sample(
    conditions=obs,
    num_samples=NUM_SAMPLES,
    guidance_kwargs=dict(constraints=[elbow_up_constraint]),
)["parameters"][0]

# 2. Masked conditional: drop the x-coordinate of the observation and mark it as
#    missing. condition_mask has one entry per observation dimension; 0 = missing.
condition_mask = np.array([[0.0, 1.0]])
posterior_missing_obs = workflow.sample(
    conditions=obs,
    num_samples=NUM_SAMPLES,
    condition_mask=condition_mask,
)["parameters"][0]

# 3. Masked target: mark the arm's height offset h (parameter index 0) as
#    missing via target_condition_mask, so the network marginalizes it out.
target_condition_mask = np.array([[0.0, 1.0, 1.0, 1.0]])  # drop height, keep three angles
posterior_drop_height = workflow.sample(
    conditions=obs,
    num_samples=NUM_SAMPLES,
    target_condition_mask=target_condition_mask,
)["parameters"][0]
# The marginalized height is pure latent noise. Draw it from the prior instead
height_prior_sigma = InverseKinematicsModel().sigmas[0]
posterior_drop_height[:, 0] = np.random.default_rng(42).normal(
    0.0, height_prior_sigma, size=NUM_SAMPLES
)

# 4. Fixed target: fix the height to a known value via target_inference_mask
#    (1 = inferred, 0 = fixed) together with targets_fixed.
targets_fixed = np.array([[0.0, 0.0, 0.0, 0.0]])
posterior_fix_height = workflow.sample(
    conditions=obs,
    num_samples=NUM_SAMPLES,
    target_inference_mask=target_condition_mask,
    targets_fixed=targets_fixed,
)["parameters"][0]

# %%
YOBS = r"\mathbf{y}_\mathrm{obs}"
THETA = r"\boldsymbol{\theta}"
EMPTY = r"\emptyset"   # masked: missing observation / marginalized parameter


panels = [
    (f"Guided \n${YOBS} = (1.5,\\,0)$\n$(\\theta_1,\\,\\theta_2,\\,\\theta_3,\\,\\theta_4), \\sin(\\theta_2) \\geq 0$", posterior_full),
    (f"Masked condition\n${YOBS} = (1.5,\\,{EMPTY})$\n${THETA} = (\\theta_1,\\,\\theta_2,\\,\\theta_3,\\,\\theta_4)$", posterior_missing_obs),
    (f"Marginalized target\n${YOBS} = (1.5,\\,0)$\n${THETA} = ({EMPTY},\\,\\theta_2,\\,\\theta_3,\\,\\theta_4)$", posterior_drop_height),
    (f"Fixed target\n${YOBS} = (1.5,\\,0)$\n${THETA} = (\\theta_1=0,\\,\\theta_2,\\,\\theta_3,\\,\\theta_4)$", posterior_fix_height),
]

fig, axarr = plt.subplots(1, 4, figsize=(10, 3.4),
                          subplot_kw=dict(box_aspect=1), squeeze=False, layout="constrained")

for (title, samples), ax in zip(panels, axarr.flat):
    m = InverseKinematicsModel(linecolors=[TRANSFORMER_COLOR] * 3)
    m.update_plot_ax(ax, samples, obs["observables"][0, ::-1], exemplar_color="#e6e7eb")

    ax.grid(False)
    ax.patch.set_facecolor("#FFE5CC")
    ax.patch.set_alpha(0.75)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.5)

fig.savefig(BASE / "inv_kinematics_transformer.pdf", bbox_inches="tight")
plt.show()
