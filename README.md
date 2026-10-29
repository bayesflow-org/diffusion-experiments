# Diffusion Models In Simulation-Based Inference: A Tutorial Review

This repository hosts the experimental results for [Diffusion Models In Simulation-Based Inference: A Tutorial Review](https://openreview.net/forum?id=86Q261z005) by Arruda et al. (2026).

## Contents
Several case studies illustrating the application of diffusion models in simulation-based inference (SBI):
- intro_example: Inverse kinematics.
- case_study1: Low-dimensional benchmarks in SBI.
- case_study2: High-dimensional ODE task.
- case_study3: Gaussian-Random-Field task.
- case_study4: Compositional score matching example.

Moreover, a tutorial on diffusion models for SBI is provided as marimo-notebook: [tutorial.py](https://bayesflow-org.github.io/diffusion-experiments/).

----

A post-publication software update identified two points at which the implemented methods differed from their intended formulations. 
1. Loss weights $\omega(t)$ were applied only as a batch-level rescaling, so all diffusion models were effectively trained without per-sample weighting; the reported cosine results thus correspond to a cosine schedule without weighting, which we recommend. Sigmoid weighting was not implemented as intended, and after testing, it is not recommended anymore.
2. The continuous consistency model was trained with a slightly modified objective (one term of the tangent was scaled twice and used an adaptive weighting that could not vary with $t$); we expect improved performance relative to the results reported here.

In BayesFlow v.2.1.0, the above differences have been corrected.

----

## Requirements
- Python 3.11
- [uv](https://github.com/astral-sh/uv) package manager

## Installation
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
# Create virtual environment and install dependencies
uv sync
```

## Running Experiments
- Launch training and evaluation, e.g., for case study 1:
```bash
uv run -m case_study1.run_benchmark
```
