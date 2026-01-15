#%%
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import pickle
import itertools
import time
from pathlib import Path
import logging

import numpy as np
import torch
import sbibm
from sbibm.metrics import c2st
import bayesflow as bf
import keras
from keras.utils import clear_session

from case_study1.model_settings_benchmark import load_model, NUM_BATCHES_PER_EPOCH, BATCH_SIZE, is_compatible
from case_study1.model_settings_benchmark import MODELS, SAMPLER_SETTINGS, ODE_METHODS, SDE_METHODS, LANGEVIN_METHODS, NUM_STEPS_SAMPLER


logging.getLogger("bayesflow").setLevel(logging.DEBUG)

job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
partition = os.environ.get("SLURM_JOB_PARTITION", "unknown")
benchmarks = list(
    (m, mlp, t, s) for s, m, mlp, t in itertools.product(
    ['ode', 'sde', 'langevin'], MODELS.keys(), ['mlp', 'time_mlp'], sbibm.get_available_tasks()
) if is_compatible(m, s)
) # 640 jobs, first 360 are ODE

model_name, subnet, task_name, sampler_family = benchmarks[job_id]
BASE = Path(__file__).resolve().parent
metrics_dir = BASE / 'metrics'
#metrics_dir = Path('/lustre/scratch/data/jarruda_hpc-diffusion_experiments/case_study1/metrics')

logging.info(f"Running job {job_id} with model {model_name}, subnet {subnet}, task {task_name}, sampler {sampler_family}.")
task = sbibm.get_task(task_name)
conf_tuple = MODELS[model_name]
if sampler_family == 'ode':
    sampling_methods = [m for m in ODE_METHODS if is_compatible(model_name, m)]
elif sampler_family == 'sde':
    sampling_methods = [m for m in SDE_METHODS if is_compatible(model_name, m)]
else:
    sampling_methods = [m for m in LANGEVIN_METHODS if is_compatible(model_name, m)]

sim_budget = 'online'
if 'offline' in model_name:
    sim_budget = NUM_BATCHES_PER_EPOCH*BATCH_SIZE

if subnet == 'time_mlp':
    # load_model will set the subnet accordingly
    model_name += '_time_mlp'

#%%
if task_name == 'lotka_volterra':
    # sbibm requires julia for this task
    simulator = bf.simulators.LotkaVolterra(subsample='original')

elif task_name == 'sir':
    # sbibm requires julia for this task
    simulator = bf.simulators.SIR(subsample='original', scale_by_total=True)  # sbibm uses scale_by_total=False
else:
    prior = task.get_prior()
    sbibm_simulator = task.get_simulator()

    def sbibm_simulator_bf():
        thetas = prior(num_samples=1)
        xs = sbibm_simulator(thetas)
        return dict(parameters=thetas.numpy()[0], observables=xs.numpy()[0])

    simulator = bf.make_simulator(sbibm_simulator_bf)

if isinstance(sim_budget, int):
    training_data = simulator.sample((sim_budget,))
    logging.info(f"Simulated {sim_budget} parameter-observation pairs.")
else:
    training_data = None
    logging.info(f"Using online simulation.")

#%%
workflow = load_model(conf_tuple=conf_tuple,
                      training_data=training_data,
                      simulator=simulator,
                      storage=BASE / "models",
                      problem_name=task_name, model_name=model_name)

#%%
observations = []
for num_observation in range(1, 11):
    logging.info(f"Running {num_observation} observations.")
    observation = task.get_observation(num_observation=num_observation).numpy()
    if task_name == 'sir':
        observation = observation / simulator.total_count  # sbibm SIR uses scale_by_total=False
    reference_samples = task.get_reference_posterior_samples(num_observation=num_observation)
    num_samples = reference_samples.shape[0]
    observations.append(observation)

    for sampler_name in sampling_methods:
        if 'joint' in sampler_name:
            continue  # joint sampling done later
        file_name = f'{model_name}_{task_name}_{sampler_name}_{num_observation}'

        if not os.path.exists(metrics_dir / f'samples_{file_name}.pkl'):
            start_time = time.perf_counter()
            if 'consistency' in model_name:
                posterior_samples_dict = workflow.sample(conditions={'observables': observation}, num_samples=num_samples)
            else:
                if 'scheduled' in SAMPLER_SETTINGS[sampler_name]:
                    min_snr = workflow.inference_network.noise_schedule.log_snr_min
                    max_snr = workflow.inference_network.noise_schedule.log_snr_max
                    snr_schedule = keras.ops.linspace(min_snr, max_snr, num=NUM_STEPS_SAMPLER)
                    SAMPLER_SETTINGS[sampler_name]['steps'] = workflow.inference_network.noise_schedule.get_t_from_log_snr(
                        snr_schedule, training=False
                    )
                posterior_samples_dict = workflow.sample(conditions={'observables': observation}, num_samples=num_samples,
                                                         **SAMPLER_SETTINGS[sampler_name])
            end_time = time.perf_counter()
            logging.info(f"Sampling time: {end_time - start_time} seconds.")

            posterior_samples = posterior_samples_dict['parameters'][0]
            with open(metrics_dir / f'samples_{file_name}.pkl', 'wb') as f:
                pickle.dump(posterior_samples, f)
            with open(metrics_dir / f'time_{file_name}.txt', 'w') as f:
                f.write(str(end_time - start_time))

observations = np.concatenate(observations, axis=0)

#%%
for sampler_name in sampling_methods:
    if 'joint' in sampler_name:
        file_name = f'{model_name}_{task_name}_{sampler_name}'

        if not os.path.exists(metrics_dir / f'samples_{file_name}.pkl'):
            start_time = time.perf_counter()
            posterior_samples_dict = workflow.sample(conditions={'observables': observations}, num_samples=num_samples,
                                                     **SAMPLER_SETTINGS[sampler_name])
            end_time = time.perf_counter()
            logging.info(f"Joint sampling time: {end_time - start_time} seconds.")

            posterior_samples = posterior_samples_dict['parameters']
            with open(metrics_dir / f'samples_{file_name}.pkl', 'wb') as f:
                pickle.dump(posterior_samples, f)
            with open(metrics_dir / f'time_{file_name}.txt', 'w') as f:
                f.write(str(end_time - start_time))

clear_session()

#%%
for sampler_name in sampling_methods:
    results_file = metrics_dir / f'c2st_results_{model_name}_{task_name}_{sampler_name}.pkl'
    if os.path.exists(results_file):
        logging.info(f"C2ST results for {sampler_name} already exist, skipping.")
        continue

    logging.info(f"C2ST for Sampler: {sampler_name}")
    if 'joint' in sampler_name:
        with open(metrics_dir / f'samples_{model_name}_{task_name}_{sampler_name}.pkl', 'rb') as f:
            posterior_samples_list = pickle.load(f)
        with open(metrics_dir / f'time_{model_name}_{task_name}_{sampler_name}.txt', 'r') as f:
            sampling_time = float(f.read())
        logging.info(f"Joint sampling time: {sampling_time} seconds.")

    c2st_results = []
    for num_observation in range(1, 11):
        if 'joint' in sampler_name:
            posterior_samples = posterior_samples_list[num_observation - 1]
        else:
            file_name = f'{model_name}_{task_name}_{sampler_name}_{num_observation}'
            with open(metrics_dir / f'samples_{file_name}.pkl', 'rb') as f:
                posterior_samples = pickle.load(f)
            with open(metrics_dir / f'time_{file_name}.txt', 'r') as f:
                sampling_time = float(f.read())
            logging.info(f"Sampling time: {sampling_time} seconds.")

        reference_samples = task.get_reference_posterior_samples(num_observation=num_observation)
        posterior_samples = torch.as_tensor(posterior_samples)
        posterior_samples = posterior_samples[torch.isfinite(posterior_samples).all(dim=-1)]
        c2st_accuracy = c2st(reference_samples, posterior_samples).numpy().item()
        logging.info(f"{num_observation} C2ST accuracy: {c2st_accuracy}")

        c2st_results.append({
            'c2st': c2st_accuracy,
            'time': sampling_time,
        })

    with open(results_file, 'wb') as f:
        pickle.dump(c2st_results, f)
logging.info('Done.')
