# %%
import os
from pathlib import Path

os.environ["KERAS_BACKEND"] = "jax"

import matplotlib.pyplot as plt
import numpy as np
from bayesflow.diagnostics.metrics import classifier_two_sample_test
from bayesflow.metrics.maximum_mean_discrepancy import maximum_mean_discrepancy

from inverse_kinematics import InverseKinematicsModel

BASE = Path(__file__).resolve().parent

# %%
# Observation from https://arxiv.org/abs/2101.10763
obs = {"observables": np.array([0, 1.5])}

titles = {
    "abc": "ABC-SMC",
    "flow_matching": "Flow Matching",
    "cot_flow_matching": "Flow Matching (COT)",
    "consistency_model": "Discrete Consistency",
    "stable_consistency_model": "Continuous Consistency",
    "diffusion_edm_vp": "VP EDM",
    "diffusion_edm_ve": "VE EDM",
    "diffusion_cosine_F": r"Cosine $\mathbf{F}$-pred.",
    "diffusion_cosine_v": r"Cosine $\boldsymbol{v}$-pred.",
    "diffusion_cosine_noise": r"Cosine $\boldsymbol{\epsilon}$-pred.",
}

colors = {
    "abc": "maroon",
    "diffusion_edm_ve": "#FB9A99",
    "diffusion_edm_vp": "#E7298A",
    "diffusion_cosine_noise": "#54278F",
    "diffusion_cosine_v": "#9E9AC8",
    "diffusion_cosine_F": "#7570B3",
    "flow_matching": "#1B9E77",
    "cot_flow_matching": "#4CAF50",
    "consistency_model": "#D95F02",
    "stable_consistency_model": "#E6AB02"
}

files = os.listdir("models")
kinematics_samples = {
    f.replace("_inverse_kinematics.npy", ""): np.load(os.path.join("models", f)) for f in files if "kinematics.npy" in f
}
# %%
# Compute C2ST scores and MMD scores
approx_ground_truth = kinematics_samples["abc"]
models = [m for m in list(titles.keys()) if m != "abc"]

c2st_results = {k: None for k in models}
mmd_results = {k: None for k in models}
for m in models:
    c2st_results[m] = classifier_two_sample_test(kinematics_samples[m], approx_ground_truth)
    mmd_results[m] = maximum_mean_discrepancy(kinematics_samples[m], approx_ground_truth)#.detach().cpu().item()

# %%
fig, axarr = plt.subplots(2, len(kinematics_samples) // 2, figsize=(10, 4),
                          subplot_kw=dict(box_aspect=1), squeeze=False, layout='constrained')

for name, ax in zip(titles, axarr.flat):
     m = InverseKinematicsModel(linecolors=[colors[name]] * 3)
     m.update_plot_ax(ax, kinematics_samples[name][:1000], obs["observables"][::-1], exemplar_color="#e6e7eb")

for title, ax in zip(titles, axarr.flat):
    ax.grid(False)
    ax.patch.set_facecolor('#FFE5CC')
    ax.patch.set_alpha(0.75)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines["bottom"].set_alpha(0.0)
    ax.spines["top"].set_alpha(0.0)
    ax.spines["right"].set_alpha(0.0)
    ax.spines["left"].set_alpha(0.0)
    ax.set_aspect('equal')
    ax.set_title(titles[title], fontsize=10.5)

    if title != "abc":
        mmd = mmd_results[title]
        c2st = c2st_results[title]

        ax.text(
            0.99, 0.01,
            f"MMD={mmd:.2g}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8.5,
            color="black"
        )

        ax.text(
            0.99, 0.9,
            f"C2ST={c2st:.2g}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8.5,
            color="black"
        )

fig.savefig(BASE / "inv_kinematics_samples.pdf", bbox_inches="tight")
plt.show()
