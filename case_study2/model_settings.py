import os

import bayesflow as bf
import keras

from case_study2.ema_callback import EMA, save_ema_models


EPOCHS = 1000
BATCH_SIZE = 64
NUM_SAMPLES_INFERENCE = 1000

SUBNET_KWARGS = {
    "widths": (256, 256, 256, 256, 256),
}

MODELS = {
        ## TimeSeriesNetwork summary network
        "flow_matching": (bf.networks.FlowMatching, {"subnet_kwargs": SUBNET_KWARGS}),
        "flow_matching_edm": (bf.networks.FlowMatching, {'time_power_law_alpha': -0.6, "subnet_kwargs": SUBNET_KWARGS}),
        "ot_flow_matching": (bf.networks.FlowMatching, {"use_optimal_transport": True, "subnet_kwargs": SUBNET_KWARGS}),
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
        "diffusion_cosine_noise": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "noise",
            "subnet_kwargs": SUBNET_KWARGS}),
        ## FusionTransformer summary network
        "flow_matching_ft": (bf.networks.FlowMatching, {"subnet_kwargs": SUBNET_KWARGS}),
        "flow_matching_edm_ft": (bf.networks.FlowMatching, {'time_power_law_alpha': -0.6, "subnet_kwargs": SUBNET_KWARGS}),
        "ot_flow_matching_ft": (bf.networks.FlowMatching, {"use_optimal_transport": True, "subnet_kwargs": SUBNET_KWARGS}),
        "consistency_model_ft": (bf.networks.ConsistencyModel, {"total_steps": EPOCHS * BATCH_SIZE, "subnet_kwargs": SUBNET_KWARGS}),
        "stable_consistency_model_ft": (bf.experimental.StableConsistencyModel, {"subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_edm_vp_ft": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "preserving"},
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_edm_vp_ft_ema": (bf.networks.DiffusionModel, {
                            "noise_schedule": "edm",
                            "prediction_type": "F",
                            "schedule_kwargs": {"variance_type": "preserving"},
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_edm_ve_ft": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "exploding"},
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_cosine_F_ft": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "F",
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_cosine_v_ft": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "velocity",
            "subnet_kwargs": SUBNET_KWARGS}),
        "diffusion_cosine_noise_ft": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "noise",
            "subnet_kwargs": SUBNET_KWARGS}),
    }

MIN_STEPS = 50
MAX_STEPS = 1_000
SAMPLER_SETTINGS = {
    'ode': {
        'method': 'tsit5',
        'steps': "adaptive",
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
    'sde': {
        'method': 'two_step_adaptive',
        'steps': "adaptive",
        'min_steps': MIN_STEPS,
        'max_steps': MAX_STEPS
    },
}

def load_model(adapter, conf_tuple, param_names, training_data, validation_data, storage, problem_name, model_name,
               use_ema=True):
    if 'ft' in model_name:
        summary_network = bf.networks.FusionTransformer(summary_dim=len(param_names) * 2)
    else:
        summary_network = bf.networks.TimeSeriesNetwork(summary_dim=len(param_names) * 2,
                                                        recurrent_dim=256, recurrent_type='LSTM')

    workflow = bf.BasicWorkflow(
        adapter=adapter,
        training_data=training_data,
        validation_data=validation_data,
        summary_network=summary_network,
        inference_network=conf_tuple[0](**conf_tuple[1]),
        standardize='all'
    )

    model_path = storage / f'petab_benchmark_diffusion_model_{problem_name}_{model_name}.keras'
    model_path_ema = storage / f'petab_benchmark_diffusion_model_{problem_name}_{model_name}_ema.keras'
    if 'ema' in model_name:
        cbs = [EMA()]
    else:
        cbs = None

    if not os.path.exists(model_path):
        workflow.fit_offline(
            training_data,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=validation_data,
            verbose=2,
            callbacks=cbs
        )
        workflow.approximator.save(model_path)

        if 'ema' in model_name:
            save_ema_models(workflow.approximator, cbs[0], path_ema=model_path_ema, path_noema=model_path)
    else:
        if use_ema and 'ema' in model_name:
            workflow.approximator = keras.models.load_model(model_path_ema)
        else:
            workflow.approximator = keras.models.load_model(model_path)
    return workflow
