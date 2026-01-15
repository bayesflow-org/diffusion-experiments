import os
import pickle
import numpy as np
from pathlib import Path
import logging
os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import keras

from keras.utils import clear_session

BASE = Path(__file__).resolve().parent
EPOCHS = 1000
BATCH_SIZE = 128
NUM_SAMPLES_INFERENCE = 1000
MODELS = {
        "flow_matching": (bf.networks.FlowMatching, {}),
        "cot_flow_matching": (bf.networks.FlowMatching, {"use_optimal_transport": True}),
        "consistency_model": (bf.networks.ConsistencyModel, {"total_steps": EPOCHS*BATCH_SIZE}),
        "stable_consistency_model": (bf.experimental.StableConsistencyModel, {}),
        "diffusion_edm_vp": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F", 
            "schedule_kwargs": {"variance_type": "preserving"}}),
        "diffusion_edm_ve": (bf.networks.DiffusionModel, {
            "noise_schedule": "edm",
            "prediction_type": "F", 
            "schedule_kwargs": {"variance_type": "exploding"}}),
        "diffusion_cosine_F": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "F", }),
        "diffusion_cosine_v": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "velocity"}),   
        "diffusion_cosine_noise": (bf.networks.DiffusionModel, {
            "noise_schedule": "cosine",
            "prediction_type": "noise"}),
    }

DATASETS = ["inverse_kinematics"]

def train_model(model_name, dataset_name, conf_tuple, data):

    adapter = (
        bf.adapters.Adapter()
        .rename("parameters", "inference_variables")
        .rename("observables", "inference_conditions")
    )

    inference_net = conf_tuple[0](**conf_tuple[1])

    model_path = BASE / 'models' / f"{model_name}_{dataset_name}.keras"
    workflow = bf.workflows.BasicWorkflow(
        adapter=adapter,
        inference_network=inference_net,
        standardize='all',
    )

    if not os.path.exists(model_path):
        _ = workflow.fit_offline(
            data=data["train"],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=data["validation"]
        )
        workflow.approximator.save(model_path)
    else:
        workflow.approximator = keras.models.load_model(model_path)

    if dataset_name == "two_moons":
        # Posterior samples on, as typical in SBI (e.g., https://arxiv.org/abs/1905.07488)
        obs = np.zeros((1, 2), dtype=np.float32)
        
    elif dataset_name == "inverse_kinematics":
        # Posterior samples for position (0, 1.5) from https://arxiv.org/abs/2101.10763
        obs = np.array([[0, 1.5]], dtype=np.float32)
    else:
        raise NotImplementedError("Dataset should be in ['two_moons', 'inverse_kinematics']")

    bf.utils.logging.info(f"Sampling {NUM_SAMPLES_INFERENCE} samples using {model_name} on {dataset}...")
    samples = workflow.sample(conditions={"observables": obs}, num_samples=NUM_SAMPLES_INFERENCE)
    np.save(BASE / "models" / f"{model_name}_{dataset_name}.npy", samples["parameters"].squeeze())


if __name__ == "__main__":
    
    for dataset in DATASETS:

        data_dict = {
            "train": pickle.load(open(BASE / "models" / f"{dataset}_train_data.pkl", "rb")),
            "validation": pickle.load(open(BASE / "models" / f"{dataset}_validation_data.pkl", "rb")),
        }

        for model_name, conf_tuple in MODELS.items():
            bf.utils.logging.info(f"Training {model_name} on {dataset}...")

            train_model(model_name, dataset, conf_tuple, data_dict)
            clear_session()
