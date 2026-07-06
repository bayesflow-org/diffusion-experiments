# Diffusion Models In Simulation-Based Inference: A Tutorial Review

This repository hosts the experimental results for a [review](https://arxiv.org/abs/2512.20685) on diffusion models in Simulation-Based Inference (SBI).

## Contents
Several case studies illustrating the application of diffusion models in SBI:
- intro_example: Inverse kinematics.
- case_study1: Low-dimensional benchmarks in SBI.
- case_study2: High-dimensional ODE task.
- case_study3: Gaussian-Random-Field task.
- case_study4: Compositional score matching example.

Moreover, a tutorial on diffusion models for SBI is provided as marimo-notebook: [tutorial.py](https://bayesflow-org.github.io/diffusion-experiments/).

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
