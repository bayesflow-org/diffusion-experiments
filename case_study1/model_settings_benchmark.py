#%%
import os

import bayesflow as bf
import keras
from bayesflow.networks.inference.diffusion import CosineNoiseSchedule

from case_study1.ema_callback import EMA, save_ema_models


EPOCHS = 1_000
BATCH_SIZE = 128
NUM_BATCHES_PER_EPOCH = 32_000 // BATCH_SIZE  # for online training

SUBNET_KWARGS = {
    "widths": (256, 256, 256, 256, 256),
}

MODELS = {
        "flow_matching": (bf.networks.FlowMatching, {"subnet_kwargs": SUBNET_KWARGS}),
        "flow_matching_edm": (bf.networks.FlowMatching, {'time_power_law_alpha': -0.6, "subnet_kwargs": SUBNET_KWARGS}),
        "ot_flow_matching": (bf.networks.FlowMatching, {"use_optimal_transport": True, "subnet_kwargs": SUBNET_KWARGS,
                                                        "optimal_transport_kwargs": {"condition_ratio": 0.5}}),
        "ot_partial_flow_matching": (bf.networks.FlowMatching, {"use_optimal_transport": True, "subnet_kwargs": SUBNET_KWARGS,
                                                                "optimal_transport_kwargs": {"partial_factor": 0.8,
                                                                                             "condition_ratio": 0.5}}),
        "cot_flow_matching": (bf.networks.FlowMatching,
                              {"use_optimal_transport": True, "subnet_kwargs": SUBNET_KWARGS,
                               "optimal_transport_kwargs": {"condition_ratio": 0.01}}),
        "cot_0_02_flow_matching": (bf.networks.FlowMatching,
                              {"use_optimal_transport": True, "subnet_kwargs": SUBNET_KWARGS,
                               "optimal_transport_kwargs": {"condition_ratio": 0.02}}),
        "cot_0_05_flow_matching": (bf.networks.FlowMatching,
                              {"use_optimal_transport": True, "subnet_kwargs": SUBNET_KWARGS,
                               "optimal_transport_kwargs": {"condition_ratio": 0.05}}),
        "cot_0_1_flow_matching": (bf.networks.FlowMatching,
                              {"use_optimal_transport": True, "subnet_kwargs": SUBNET_KWARGS,
                               "optimal_transport_kwargs": {"condition_ratio": 0.1}}),
        "cot_partial_flow_matching": (bf.networks.FlowMatching,
                                  {"use_optimal_transport": True, "subnet_kwargs": SUBNET_KWARGS,
                                   "optimal_transport_kwargs": {"condition_ratio": 0.01, "partial_factor": 0.9}}),
        "consistency_model": (bf.networks.ConsistencyModel, {"total_steps": EPOCHS*BATCH_SIZE, "subnet_kwargs": SUBNET_KWARGS}),
        "stable_consistency_model": (bf.networks.StableConsistencyModel, {"subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_edm_vp": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "preserving"},
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_edm_vp_ema": (bf.networks.DiffusionModel, {
                    "noise_schedule": "edm",
                    "prediction_type": "F",
                    "schedule_kwargs": {"variance_type": "preserving"},
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_edm_vp_transformer": (bf.networks.DiffusionModel, {
                            "noise_schedule": "edm",
                            "prediction_type": "F",
                            "schedule_kwargs": {"variance_type": "preserving"},
                    "subnet": "time_transformer",
                    "subnet_kwargs": dict(widths=(128, 128, 128, 128, 128))}),
        "diffusion_edm_ve": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "exploding"},
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_cosine_F": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "F", }),
        "diffusion_cosine_v": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "velocity",
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_cosine_v_lw": (bf.networks.DiffusionModel, {
            "noise_schedule": CosineNoiseSchedule(weighting="likelihood_weighting"),
            "prediction_type": "velocity",
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_cosine_noise": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "noise",
            "subnet_kwargs": SUBNET_KWARGS}),
    }

NUM_STEPS_SAMPLER = 500
MIN_STEPS = 50
MAX_STEPS = 1_000
SAMPLER_SETTINGS = {
    'ode-euler': {
        'method': 'euler',
        'steps': NUM_STEPS_SAMPLER,
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'ode-euler-mini': {
        'method': 'euler',
        'steps': 10,
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'ode-euler-small': {
        'method': 'euler',
        'steps': 50,
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'ode-euler-scheduled': {
        'method': 'euler',
        'steps': NUM_STEPS_SAMPLER,
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'ode-rk45': {
        'method': 'rk45',
        'steps': NUM_STEPS_SAMPLER,
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'ode-tsit5': {
        'method': 'tsit5',
        'steps': NUM_STEPS_SAMPLER,
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'ode-rk45-adaptive': {
        'method': 'rk45',
        'steps': "adaptive",
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'ode-rk45-adaptive-joint': {
        'method': 'rk45',
        'steps': "adaptive",
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'ode-tsit5-adaptive': {
        'method': 'tsit5',
        'steps': "adaptive",
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'ode-tsit5-adaptive-joint': {
        'method': 'tsit5',
        'steps': "adaptive",
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'sde-euler': {
        'method': 'euler_maruyama',
        'steps': NUM_STEPS_SAMPLER,
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'sde-euler-adaptive': {
        'method': 'euler_maruyama',
        'steps': "adaptive",
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'sde-euler-adaptive-joint': {
        'method': 'euler_maruyama',
        'steps': "adaptive",
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'sde-euler-pc': {
        'method': 'euler_maruyama',
        'steps': NUM_STEPS_SAMPLER,
        'corrector_steps': 1,
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'sde-sea': {
        'method': 'sea',
        'steps': NUM_STEPS_SAMPLER,
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'sde-shark': {
        'method': 'shark',
        'steps': NUM_STEPS_SAMPLER,
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'sde-two_step-adaptive': {
        'method': 'two_step_adaptive',
        'steps': "adaptive",
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'sde-two_step-adaptive-joint': {
        'method': 'two_step_adaptive',
        'steps': "adaptive",
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'sde-langevin': {
        'method': 'langevin',
        'steps': NUM_STEPS_SAMPLER * 4,
        'min_steps': MIN_STEPS
    },
    'sde-langevin-pc': {
        'method': 'langevin',
        'steps': NUM_STEPS_SAMPLER * 4,
        'corrector_steps': 5,
        'min_steps': MIN_STEPS
    },
}
ODE_METHODS = [k for k in SAMPLER_SETTINGS.keys() if k.startswith('ode')]
SDE_METHODS = [k for k in SAMPLER_SETTINGS.keys() if k.startswith('sde') and not 'langevin' in k and not 'pc' in k]
LANGEVIN_METHODS = [k for k in SAMPLER_SETTINGS.keys() if 'langevin' in k or 'pc' in k]


def is_compatible(_model: str, _sampler: str) -> bool:
    if not _model.startswith("diffusion") and _sampler.startswith("sde"):
        return False
    if not _model.startswith("diffusion") and (_sampler == "sde" or _sampler == "langevin" or _sampler == "ode-euler-scheduled"):
        return False
    if "consistency" in _model:
        if _sampler == 'ode':
            return True
        elif _sampler != "ode-euler":
            return False
    if (not "flow_matching" in _model) and _sampler in ['ode-euler-mini', 'ode-euler-small']:
        return False
    return True


def create_adapter(task_name):
    if task_name in ['bernoulli_glm', 'gaussian_linear', 'bernoulli_glm_raw']:
        return (
            bf.adapters.Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .rename('parameters', 'inference_variables')
            .rename('observables', 'inference_conditions')
        )
    elif task_name in ['lotka_volterra', 'sir']:
        return (
            bf.adapters.Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .log("parameters")
            .log("observables", p1=True)
            .rename('parameters', 'inference_variables')
            .rename('observables', 'inference_conditions')
        )
    elif task_name in ['gaussian_mixture', 'gaussian_linear_uniform', 'two_moons']:
        config = {  # parameters
            'gaussian_mixture': (-10, 10),
            'gaussian_linear_uniform': (-1, 1),
            'two_moons': (-1, 1),
        }[task_name]
        return (
            bf.adapters.Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .constrain("parameters", lower=config[0], upper=config[1], epsilon=1e-6, inclusive='both')
            .rename('parameters', 'inference_variables')
            .rename('observables', 'inference_conditions')
        )
    elif task_name in ['slcp', 'slcp_distractors']:
        config = {  # parameters, observables
            'slcp': ((-3, 3), (-30, 30)),
            'slcp_distractors': ((-3, 3), (-3000, 3000)),
        }[task_name]
        return (
            bf.adapters.Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .constrain("parameters", lower=config[0][0], upper=config[0][1], epsilon=1e-6, inclusive='both')
            .constrain("observables", lower=config[1][0], upper=config[1][1], epsilon=1e-6, inclusive='both')
            .nan_to_num("observables", default_value=-5)  # observables can be nan due to out of bounds
            .rename('parameters', 'inference_variables')
            .rename('observables', 'inference_conditions')
        )
    else:
        raise ValueError(f"Unknown adapter for {task_name}")


def load_model(conf_tuple, simulator, training_data, storage, problem_name, model_name, sum_dim=0, use_ema=True):
    if sum_dim > 0:
        summary_network = bf.networks.TimeSeriesNetwork(summary_dim=sum_dim)
    else:
        summary_network = None

    adapter = create_adapter(problem_name)

    workflow = bf.BasicWorkflow(
        adapter=adapter,
        simulator=simulator,
        summary_network=summary_network,
        inference_network=conf_tuple[0](**conf_tuple[1]),
        standardize='all'
    )

    model_path = storage / f'benchmark_{problem_name}_{model_name}.keras'
    model_path_ema = storage / f'benchmark_{problem_name}_{model_name}_ema.keras'
    if 'ema' in model_name:
        cbs = [EMA()]
    else:
        cbs = None

    if not os.path.exists(model_path):
        if training_data is None:
            workflow.fit_online(
                num_batches_per_epoch=NUM_BATCHES_PER_EPOCH,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=2,
                callbacks=cbs
            )
        else:
            workflow.fit_offline(
                training_data,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=2,
                callbacks=cbs
            )
        workflow.approximator.save(model_path)

        if 'ema' in model_name:
            save_ema_models(workflow.approximator, cbs[0], path_ema=model_path_ema, path_noema=model_path)
            workflow.approximator = keras.models.load_model(model_path_ema)
    else:
        if use_ema and 'ema' in model_name:
            workflow.approximator = keras.models.load_model(model_path_ema)
        else:
            workflow.approximator = keras.models.load_model(model_path)
    return workflow
