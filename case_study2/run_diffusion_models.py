#%%
# # PEtab benchmark model with BayesFlow

import os
import sys
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "tensorflow"
else:
    print(f"Using '{os.environ['KERAS_BACKEND']}' backend")

import numpy as np
import pickle
import logging
from pathlib import Path

import bayesflow as bf

from case_study2.model_settings import MODELS, NUM_SAMPLES_INFERENCE, SAMPLER_SETTINGS, load_model
from case_study2.helper_pypesto import load_problem, simulate_parallel, compute_metrics

# generate petab problem
job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 10))
BASE = Path(__file__).resolve().parent
problem_name = sys.argv[1] if len(sys.argv) > 1 else "Beer_MolBioSystems2014"
num_training_sets = 512*64
num_validation_sets = 1000
COMPUTE_METRICS = False
logging.info(problem_name)

model_name = list(MODELS.keys())[job_id]
conf_tuple = MODELS[model_name]
logging.info(model_name)

logging.getLogger("pypesto").setLevel(logging.ERROR)
logging.getLogger("petab").setLevel(logging.ERROR)
logging.getLogger("bayesflow").setLevel(logging.ERROR)
models_dir = BASE / "models"
metrics_dir = BASE / 'metrics'
#metrics_dir = Path('/lustre/scratch/data/jarruda_hpc-diffusion_experiments/case_study2/metrics')
#models_dir = Path('/lustre/scratch/data/jarruda_hpc-diffusion_experiments/case_study2/models')

#%%
if __name__ == "__main__":
    pypesto_problem, petab_problem, factory, amici_predictor = load_problem(problem_name, create_amici_model=COMPUTE_METRICS)
    param_names = [name for i, name in enumerate(pypesto_problem.x_names) if i in pypesto_problem.x_free_indices]
    lbs = np.array([lb for i, lb in enumerate(petab_problem.lb_scaled) if i in pypesto_problem.x_free_indices])
    ubs = np.array([ub for i, ub in enumerate(petab_problem.ub_scaled) if i in pypesto_problem.x_free_indices])

    if os.path.exists(models_dir / f"validation_data_petab_{problem_name}.pkl"):
        logging.info('Loading data')
        with open(models_dir / f'validation_data_petab_{problem_name}.pkl', 'rb') as f:
            validation_data = pickle.load(f)
        try:
            with open(models_dir / f'training_data_petab_{problem_name}.pkl', 'rb') as f:
                training_data = pickle.load(f)
        except FileNotFoundError:
            training_data = None
            logging.info("Training data not found")
    else:
        logging.info('Generate data')
        training_data = simulate_parallel(num_training_sets, amici_predictor, factory, petab_problem, pypesto_problem, n_cpus=n_cpus)
        validation_data = simulate_parallel(num_validation_sets, amici_predictor, factory, petab_problem, pypesto_problem, return_df=True, n_cpus=n_cpus)

        with open(models_dir / f'training_data_petab_{problem_name}.pkl', 'wb') as f:
            pickle.dump(training_data, f)
        with open(models_dir / f'validation_data_petab_{problem_name}.pkl', 'wb') as f:
            pickle.dump(validation_data, f)

    adapter = (
        bf.adapters.Adapter()
        .drop('amici_params')  # only used for simulation
        .drop('sim_data_df')
        .to_array()
        .convert_dtype("float64", "float32")
        .concatenate(param_names, into="inference_variables")
        .constrain("inference_variables", lower=lbs, upper=ubs, inclusive='both')  # after concatenate such that we can apply an array as constraint
        .as_time_series("sim_data")
        .log("sim_data", p1=True)
        .rename("sim_data", "summary_variables")
    )

    #%%
    workflow = load_model(adapter=adapter, conf_tuple=conf_tuple, param_names=param_names, training_data=training_data,
                              validation_data=validation_data, storage=models_dir, problem_name=problem_name,
                              model_name=model_name)
    if not COMPUTE_METRICS:
        diagnostics_plots = workflow.plot_default_diagnostics(test_data=validation_data, num_samples=NUM_SAMPLES_INFERENCE,
                                                              calibration_ecdf_kwargs={"difference": True, 'stacked': True})
        for k in diagnostics_plots.keys():
            diagnostics_plots[k].savefig(BASE / "plots" / f"{problem_name}_{model_name}_{k}.png")

    #%%
    test_data = {}
    for key, values in validation_data.items():
        test_data[key] = values

    if os.path.exists(metrics_dir / f'{problem_name}_metrics_{model_name}.pkl'):
        with open(metrics_dir / f'{problem_name}_metrics_{model_name}.pkl', 'rb') as f:
            metrics = pickle.load(f)
        logging.info(f'Metrics for model {model_name} already exist')
    elif COMPUTE_METRICS:
        metrics = compute_metrics(model_name=model_name, workflow=workflow, test_data=test_data,
                                  petab_problem=petab_problem, pypesto_problem=pypesto_problem,
                                  num_samples_inference=NUM_SAMPLES_INFERENCE, sampler_settings=SAMPLER_SETTINGS,
                                  n_jobs=n_cpus)

        with open(metrics_dir / f'{problem_name}_metrics_{model_name}.pkl', 'wb') as f:
            pickle.dump(metrics, f)

    logging.info('Done')
