import numpy as np
from scipy.stats import beta as beta_dist, norm as norm_dist

import bayesflow as bf


def beta_from_normal(z, a=50, b=50):
    u = norm_dist.cdf(z)
    x = beta_dist.ppf(u, a, b)  # Beta inverse CDF
    return x


def simulate_ddm_trial(nu, alpha, t0, beta, dt=1e-3, scale=1.0, max_time=10.0):
    """
    Simulates one realization of the diffusion process given
    a set of parameters and a step size `dt`.

    Parameters:
    -----------
    nu         : float
        The drift rate (rate of information uptake)
    alpha         : float
        The boundary separation (decision threshold).
    t0       : float
        Non-decision time (additive constant)
    beta      : float in [0, 1]
        Relative starting point (prior option preferences)
    dt        : float, optional (default: 1e-3 = 0.001)
        The step size for the Euler algorithm.
    scale     : float, optional (default: 1.0)
        The scale (sqrt(var)) of the Wiener process. Not considered
        a parameter and typically fixed to either 1.0 or 0.1.
    max_time  : float, optional (default: .10)
        The maximum number of seconds before forced termination.

    Returns:
    --------
    (x, c) - a tuple of response time (y - float) and a
        binary decision (c - int)
    """

    # Inits (process starts at relative starting point)
    y = beta * alpha
    rt = t0
    const = scale * np.sqrt(dt)

    # Loop through process and check boundary conditions
    while (alpha >= y >= 0) and rt <= max_time:
        # Perform diffusion equation
        z = np.random.randn()
        y += nu * dt + const * z

        # Increment step counter
        rt += dt

    if y >= alpha:
        c = 1.0
    else:
        c = 0.0
    return c, rt


def simulate_ddm(nu, alpha, t0, beta, n_subjects=1, n_trials=1):
    if isinstance(nu, (float, int)):
        nu = np.ones((n_subjects,)) * nu
        alpha = np.ones((n_subjects,)) * alpha
        t0 = np.ones((n_subjects,)) * t0
    data = np.zeros((n_subjects, n_trials, 2))
    for j_subject in range(n_subjects):
        for i_trial in range(n_trials):
            data[j_subject, i_trial] = simulate_ddm_trial(nu[j_subject], alpha[j_subject], t0[j_subject], beta)
    if n_subjects == 1 and n_trials == 1:
        data = data[0, 0]
    elif n_subjects == 1:
        data = data[0]
    elif n_trials == 1:
        data = data[:, 0]
    return dict(sim_data=data)


def score_log_norm(x, m, s):
    return -(x-m) / s**2


# ---------------------------
# Priors
# ---------------------------
def sample_hierarchical_priors(n_subjects=1):
    """
    Hierarchical draws as in your specification.
    Returns a dict with group params and per subject params.
    """
    # Group level
    mu_nu = np.random.normal(0.5, 0.3)
    mu_log_alpha = np.random.normal(0.0, 0.05)
    mu_log_t0 = np.random.normal(-1.0, 0.3)

    log_sigma_nu = np.random.normal(-1.0, 1.0)
    log_sigma_log_alpha = np.random.normal(-3.0, 1.0)
    log_sigma_log_t0 = np.random.normal(-1.0, 0.3)

    sigma_nu = np.exp(log_sigma_nu)
    sigma_log_alpha = np.exp(log_sigma_log_alpha)
    sigma_log_t0 = np.exp(log_sigma_log_t0)

    beta_raw = np.random.normal(0.0, 1.0)
    beta = beta_from_normal(beta_raw, a=50, b=50)

    # Subject level
    nu = np.random.normal(mu_nu, sigma_nu, size=n_subjects)
    log_alpha = np.random.normal(mu_log_alpha, sigma_log_alpha, size=n_subjects)
    log_t0 = np.random.normal(mu_log_t0, sigma_log_t0, size=n_subjects)
    alpha = np.exp(log_alpha)
    t0 = np.exp(log_t0)

    return {
        # group
        "mu_nu": mu_nu,
        "mu_log_alpha": mu_log_alpha,
        "mu_log_t0": mu_log_t0,
        "log_sigma_nu": log_sigma_nu,
        "log_sigma_log_alpha": log_sigma_log_alpha,
        "log_sigma_log_t0": log_sigma_log_t0,
        "beta_raw": beta_raw,
        "beta": beta,
        # subjects
        "nu": nu,
        "alpha": alpha,
        "t0": t0,
    }


def prior_global_score(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mu_nu = x["mu_nu"]
    mu_log_alpha = x["mu_log_alpha"]
    mu_log_t0 = x["mu_log_t0"]
    log_sigma_nu = x["log_sigma_nu"]
    log_sigma_log_alpha = x["log_sigma_log_alpha"]
    log_sigma_log_t0 = x["log_sigma_log_t0"]
    beta_raw = x["beta_raw"]

    parts = {
        "mu_nu": score_log_norm(mu_nu, m=0.5, s=0.3),
        "mu_log_alpha": score_log_norm(mu_log_alpha, m=0.0, s=0.05),
        "mu_log_t0": score_log_norm(mu_log_t0, m=-1.0, s=0.3),
        "log_sigma_nu": score_log_norm(log_sigma_nu, m=-1.0, s=1.0),
        "log_sigma_log_alpha": score_log_norm(log_sigma_log_alpha, m=-3.0, s=1.0),
        "log_sigma_log_t0": score_log_norm(log_sigma_log_t0, m=-1.0, s=0.3),
        "beta_raw": score_log_norm(beta_raw, m=0.0, s=1.0),
    }
    return parts


def sample_flat_priors():
    # Subject level
    nu = np.random.normal(0.5, np.exp(-1.0))
    log_alpha = np.random.normal(0.0, np.exp(-3.0))
    log_t0 = np.random.normal(-1.0, np.exp(-1.0))

    #beta = np.random.beta(a=50, b=50)
    beta_raw = np.random.normal(0.0, 1.0)
    beta = beta_from_normal(beta_raw, a=50, b=50)

    return {
        "nu": nu,
        "log_alpha": log_alpha,
        "alpha": np.exp(log_alpha),
        "log_t0": log_t0,
        "t0": np.exp(log_t0),
        "beta_raw": beta_raw,
        "beta": beta,
    }


def prior_flat_score(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    nu = x["nu"]
    log_alpha = x["log_alpha"]
    log_t0 = x["log_t0"]
    beta_raw = x["beta_raw"]

    parts = {
        "nu": score_log_norm(nu, m=0.5, s=np.exp(-1.0)),
        "log_alpha": score_log_norm(log_alpha, m=0.0, s=np.exp(-3.0)),
        "log_t0": score_log_norm(log_t0, m=-1.0, s=np.exp(-1.0)),
        "beta_raw":  score_log_norm(beta_raw, m=0.0, s=1.0)
    }
    return parts


simulator_flat = bf.make_simulator([sample_flat_priors, simulate_ddm])
simulator_hierarchical = bf.make_simulator([sample_hierarchical_priors, simulate_ddm])
