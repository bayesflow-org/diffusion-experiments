import logging
from collections import defaultdict
from copy import deepcopy
from functools import partial

import benchmark_models_petab as benchmark_models
import numpy as np
import pandas as pd
import pypesto
import pypesto.petab
from bayesflow.diagnostics.metrics import (root_mean_squared_error,
                                           posterior_contraction,
                                           calibration_error,
                                           classifier_two_sample_test)
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import median_abs_deviation

mad = partial(median_abs_deviation, nan_policy='omit')

@delayed
def sample_and_simulate(amici_predictor, factory, petab_problem, pypesto_problem, return_df=False):
    """Single iteration of sampling and simulation"""
    prior_sample = sample_from_prior(petab_problem, pypesto_problem)
    test = simulator_amici(prior_sample['amici_params'], amici_predictor, factory, petab_problem, pypesto_problem,
                           return_df=return_df)

    # Combine both dictionaries
    result = {**prior_sample, **test}
    return result


def simulate_parallel(n_samples, amici_predictor, factory, petab_problem, pypesto_problem, return_df=False, n_cpus=-1):
    """Parallel sampling and simulation"""
    results = Parallel(n_jobs=n_cpus, verbose=1)(
        sample_and_simulate(amici_predictor, factory, petab_problem, pypesto_problem, return_df) for _ in range(n_samples)
    )
    results_dict = defaultdict(list)

    for r in results:
        for key, value in r.items():
            results_dict[key].append(value)
    for key, value_list in results_dict.items():
        if isinstance(value_list[0], pd.DataFrame):
            pass
        else:
           results_dict[key] = np.array(value_list)
    return results_dict



def load_problem(problem_name = "Beer_MolBioSystems2014", create_amici_model=True):
    if create_amici_model:
        import amici
        amici.swig_wrappers.logger.setLevel(logging.CRITICAL)
        logging.getLogger("pypesto").setLevel(logging.ERROR)
        logging.getLogger("petab").setLevel(logging.ERROR)
        logging.getLogger("bayesflow").setLevel(logging.ERROR)

    petab_problem = benchmark_models.get_problem(problem_name)

    # decrease upper bounds for offset, scaling and noise parameters
    scale_params_id = [name for name in petab_problem.parameter_df.index.values if name[:6] == 'offset' or name[:5] == 'scale']
    petab_problem.parameter_df.loc[scale_params_id, 'upperBound'] = 100
    sd_params_id = [name for name in petab_problem.parameter_df.index.values if name[:3] == 'sd_']
    petab_problem.parameter_df.loc[sd_params_id, 'upperBound'] = 10

    # add normal prior (on scale) around real parameters values
    real_data_params = petab_problem.parameter_df.nominalValue
    std = 0.5
    for i in real_data_params.index:
        if petab_problem.parameter_df.loc[i, 'estimate'] == 0:
            continue
        # set prior mean depending on scale
        mean = scale_values(real_data_params.loc[i], petab_problem.parameter_df.loc[i, 'parameterScale'])
        if not 'objectivePriorType' in petab_problem.parameter_df or pd.isna(petab_problem.parameter_df.loc[i, 'objectivePriorType']):
            petab_problem.parameter_df.loc[i, 'objectivePriorType'] = "parameterScaleNormal"
            petab_problem.parameter_df.loc[i, 'objectivePriorParameters'] = f"{mean};{std}"

    for i, row in petab_problem.parameter_df.iterrows():
        if 'objectivePriorType' in row and not pd.isna(row['objectivePriorType']):
            if row['estimate'] == 0:
                logging.info(f"Parameter {i} has a {row['objectivePriorType']} prior but is not estimated, setting to nan")
                petab_problem.parameter_df.loc[i, 'objectivePriorType'] = np.nan
            # validate petab problem, if scale for parameter is defined, prior must be on the same scale
            if row['parameterScale'] != 'lin' and not row['objectivePriorType'].startswith('parameterScale'):
                raise ValueError(f"Parameter {i} has parameterScale {row['parameterScale']} but {row['objectivePriorType']} prior")

    amici_predictor = None
    factory = None
    if create_amici_model:
        importer = pypesto.petab.PetabImporter(petab_problem, simulator_type="amici")
        factory = importer.create_objective_creator()

        model = factory.create_model(verbose=False)
        amici_predictor = factory.create_predictor()
        amici_predictor.amici_objective.amici_solver.setAbsoluteTolerance(1e-8)
    else:
        # load problem
        class DummySimulator:
            def __init__(self):
                self.petab_problem = petab_problem

        importer = pypesto.petab.PetabImporter(petab_problem, simulator_type='petab', simulator=DummySimulator())

    # Creating the pypesto problem from PEtab
    pypesto_problem = importer.create_problem()
    if create_amici_model:
        pypesto_problem.print_parameter_summary()
    return pypesto_problem, petab_problem, factory, amici_predictor

#%%
def get_samples_from_dict(samples_dict, pypesto_problem):
    samples = np.stack([samples_dict[name][..., 0] for name in pypesto_problem.x_names], axis=-1)
    return samples


def sample_from_prior(petab_problem, pypesto_problem):
    lb = petab_problem.parameter_df['lowerBound'].values
    ub = petab_problem.parameter_df['upperBound'].values
    param_names_id = petab_problem.parameter_df.index.values
    param_scale = petab_problem.parameter_df['parameterScale'].values
    if 'objectivePriorType' in petab_problem.parameter_df.columns:
        prior_type = petab_problem.parameter_df['objectivePriorType'].values
    else:
        prior_type = [np.nan] * len(param_names_id)
    estimate_param = petab_problem.parameter_df['estimate'].values

    prior_dict = {}
    for i, name in enumerate(param_names_id):
        if estimate_param[i] == 0:
            prior_dict[name] = petab_problem.parameter_df['nominalValue'].values[i]  # linear space
        elif prior_type[i] == 'uniform':  # linear space
            prior_dict[name] = np.random.uniform(low=lb[i], high=ub[i])
        elif prior_type[i] == 'parameterScaleUniform' or pd.isna(prior_type[i]):
            # scale bounds to scaled space
            lb_scaled_i = scale_values(lb[i], param_scale[i])
            ub_scaled_i = scale_values(ub[i], param_scale[i])
            val = np.random.uniform(low=lb_scaled_i, high=ub_scaled_i)
            # scale to linear space
            prior_dict[name] = values_to_linear_scale(val, param_scale[i])
        elif prior_type[i] == 'parameterScaleNormal':
            mean, std = petab_problem.parameter_df['objectivePriorParameters'].values[i].split(';')
            lb_scaled_i = scale_values(lb[i], param_scale[i])
            ub_scaled_i = scale_values(ub[i], param_scale[i])
            a, b = (lb_scaled_i - float(mean)) / float(std), (ub_scaled_i - float(mean)) / float(std)
            rv = stats.truncnorm.rvs(loc=float(mean), scale=float(std), a=a, b=b)
            # scale to linear space
            prior_dict[name] = values_to_linear_scale(rv, param_scale[i])
        elif prior_type[i] == 'normal':
            mean, std = petab_problem.parameter_df['objectivePriorParameters'].values[i].split(';')
            a, b = (lb[i] - float(mean)) / float(std), (ub[i] - float(mean)) / float(std)
            rv = stats.truncnorm.rvs(loc=float(mean), scale=float(std), a=a, b=b)
            prior_dict[name] = rv
        elif prior_type[i] == 'laplace':
            loc, scale = petab_problem.parameter_df['objectivePriorParameters'].values[i].split(';')
            for t in range(10):
                rv = np.random.laplace(loc=float(loc), scale=float(scale))
                if lb[i] <= rv <= ub[i]:  # sample from truncated laplace
                    break
            prior_dict[name] = rv
        else:
            raise ValueError("Unknown prior type:", prior_type[i])
        # scale params and make list
        prior_dict[name] = np.array([scale_values(prior_dict[name], param_scale[i])])

    # prepare variables for simulation
    x = get_samples_from_dict(prior_dict, pypesto_problem=pypesto_problem)
    prior_dict['amici_params'] = x  # scaled parameters for amici
    return prior_dict


def simulator_amici(amici_params, amici_predictor, factory, petab_problem, pypesto_problem, return_df=False):
    pred = amici_predictor(amici_params)  # expect amici_params to be scaled
    sim_df, failed = amici_pred_to_df(pred, amici_params,
                                      factory=factory, petab_problem=petab_problem, pypesto_problem=pypesto_problem)
    sim = amici_df_to_array(sim_df)
    if failed:
        sim = sim * np.nan  # set all to nan if simulation failed
    if return_df:
        return dict(sim_data=sim, sim_failed=failed, sim_data_df=sim_df)
    return dict(sim_data=sim, sim_failed=failed)


def create_pypesto_problem(petab_problem, measurement_df=None):
    _petab_problem = deepcopy(petab_problem)
    if not measurement_df is None:
        _petab_problem.measurement_df = measurement_df
    _importer = pypesto.petab.PetabImporter(_petab_problem, simulator_type="amici", validate_petab=False)
    _factory = _importer.create_objective_creator()
    _model = _factory.create_model(verbose=False)

    _pypesto_problem = _importer.create_problem(
        startpoint_kwargs={"check_fval": True, "check_grad": True}
    )
    _pypesto_problem.startpoint_method = pypesto.startpoint.PriorStartpoints(check_fval=True, check_grad=True)

    if isinstance(_pypesto_problem.objective, pypesto.objective.AggregatedObjective):
        _pypesto_problem.objective._objectives[0].amici_solver.setAbsoluteTolerance(1e-8)
    else:
        _pypesto_problem.objective.amici_solver.setAbsoluteTolerance(1e-8)

    if not measurement_df is None:
        return _pypesto_problem, _petab_problem
    return _pypesto_problem


def scale_values(x, scale):
    if np.isscalar(x):
        return_scalar = True
    else:
        return_scalar = False
    x = np.atleast_1d(x)
    scale = np.atleast_1d(scale)

    if scale.size == 1:
        scale = np.repeat(scale.item(), x.size)

    out = np.empty_like(x, dtype=float)
    for i, (val, sc) in enumerate(zip(x, scale)):
        if sc == 'log10':
            out[i] = np.log10(val)
        elif sc == 'log':
            out[i] = np.log(val)
        elif sc == 'lin':
            out[i] = val
        else:
            raise ValueError(f"Unknown scale: {sc}")
    if return_scalar:
        return out.item()
    return out


def values_to_linear_scale(x, scale):
    if np.isscalar(x):
        return_scalar = True
    else:
        return_scalar = False
    x = np.atleast_1d(x)
    scale = np.atleast_1d(scale)

    if scale.size == 1:
        scale = np.repeat(scale.item(), x.size)

    out = np.empty_like(x, dtype=float)
    for i, (val, sc) in enumerate(zip(x, scale)):
        if sc == 'log10':
            out[i] = np.power(10, val)
        elif sc == 'log':
            out[i] = np.exp(val)
        elif sc == 'lin':
            out[i] = val
        else:
            raise ValueError(f"Unknown scale: {sc}")
    if return_scalar:
        return out.item()
    return out


def apply_noise_to_data(sim_df, params, field, pypesto_problem, petab_problem):
    # apply noise parameters to simulation
    for obs_id in sim_df['observableId'].unique():
        for cond_id in sim_df['simulationConditionId'].unique():
            obs_cond_index = (sim_df['observableId']==obs_id) & (sim_df['simulationConditionId']==cond_id)
            if obs_cond_index.sum() == 0:
                continue
            # get noise parameter for this observable
            obs_noise_param = sim_df.loc[obs_cond_index, 'noiseParameters'].unique()
            if len(obs_noise_param) > 1:
                logging.info(obs_noise_param)
                raise ValueError(f"Multiple noise parameters for observable {obs_id}")

            # scale simulation according to observable transformation
            obs_transformation = petab_problem.observable_df.loc[obs_id, 'observableTransformation']
            scaled_sim = scale_values(sim_df.loc[obs_cond_index, field].values, obs_transformation)

            # scale parameters according to their transformation
            params_scaled = values_to_linear_scale(params, petab_problem.parameter_df['parameterScale'].values)

            # get noise parameter value
            obs_noise_param = obs_noise_param[0]
            obs_distribution = petab_problem.observable_df.loc[obs_id, 'noiseDistribution']
            obs_formula = petab_problem.observable_df.loc[obs_id, 'noiseFormula']
            if len(obs_formula.split('+')) == 2:
                obs_noise_params = obs_noise_param.split(';')
                if obs_noise_params[1] == '0':
                    noise_param_index = pypesto_problem.x_names.index(obs_noise_params[0])
                    noise_param_value = params_scaled[noise_param_index]
                else:
                    noise_param_index1 = pypesto_problem.x_names.index(obs_noise_params[0])
                    noise_param_index2 = pypesto_problem.x_names.index(obs_noise_params[1])
                    noise_param_value =  params_scaled[noise_param_index1] + params_scaled[noise_param_index2]
            elif isinstance(obs_noise_param, str):  # is a parameter
                noise_param_index = pypesto_problem.x_names.index(obs_noise_param)
                noise_param_value = params_scaled[noise_param_index]
            else:
                noise_param_value = float(obs_noise_param)  # is a nominal value

            # apply noise model
            if obs_distribution == 'normal':
                sim_df.loc[obs_cond_index, field] = scaled_sim + np.random.normal(0, 1, size=scaled_sim.shape) * noise_param_value
                sim_df.loc[obs_cond_index, field] = np.maximum(sim_df.loc[obs_cond_index, field], 0)  # avoid negative values in biological quantities
            else:
                raise ValueError(f"Unknown observable distribution: {obs_distribution}")
            sim_df.loc[obs_cond_index, field] = values_to_linear_scale(sim_df.loc[obs_cond_index, field].values,
                                                                       obs_transformation)
    return sim_df


def amici_pred_to_df(pred, params, factory, pypesto_problem, petab_problem, field='simulation'):
    """
        Convert amici prediction results to a dataframe with noise applied.

        Parameters
        ----------
        pred : pypesto.predict.PredictionResult
            The predictor object containing simulation results.
        params: array-like
            The parameter values used for the simulation.
        factory:
            pypesto factory object
        pypesto_problem:
            pypesto problem object
        petab_problem:
            petab problem object
        field : str, optional
            The field to extract from the prediction results. Default is 'simulation'.
        Returns
        -------
        array: a 2D numpy array of shape (n_timepoints, n_series) with NaNs for missing values.
              Each column corresponds to a unique (simulationConditionId, observableId) pair.
    """
    failed = False
    try:
        if field == 'simulation':
            sim_df = factory.prediction_to_petab_simulation_df(pred)
        else:
            sim_df = factory.prediction_to_petab_measurement_df(pred)
        sim_df = apply_noise_to_data(sim_df, params, field=field,
                                     pypesto_problem=pypesto_problem, petab_problem=petab_problem)
    except ValueError as e:
        logging.warning("Simulation failed:", e)
        sim_df = petab_problem.measurement_df.copy()
        sim_df[field] = 0  # will be set to nan later
        failed = True
    return sim_df, failed


def amici_df_to_array(sim_df, field='simulation'):
    """
    Convert amici simulation dataframe to a 2D numpy array.

    Parameters
    ----------
    sim_df : pandas.DataFrame
        The simulation dataframe in PEtab format.
    field: str
        The field in the simulation dataframe to extract (default is 'simulation').
    Returns
    -------
    array: a 2D numpy array of shape (n_timepoints, n_series) with NaNs for missing values.
          Each column corresponds to a unique (simulationConditionId, observableId) pair.
    """
    # all unique time points
    timepoints = np.sort(sim_df["time"].unique())

    # pivot to wide format, index=series, columns=time
    wide = sim_df.pivot_table(
        index=['simulationConditionId', 'observableId'],
        columns="time",
        values=field,
        aggfunc="first",  # just in case duplicates exist
        sort=True
    )

    # reindex columns to include all time points
    wide = wide.reindex(columns=timepoints)

    arr = wide.to_numpy(dtype=float).T  # shape (n_timepoints, n_series)

    # inf to nan
    arr[np.isinf(arr)] = np.nan
    return arr


def compute_likelihood(petab_problem, measurement_df, eval_params) -> float:
    if not 'measurement' in measurement_df.columns:
        measurement_df['measurement'] = measurement_df['simulation']  # pypesto expects measurement column
    _pypesto_problem, _ = create_pypesto_problem(petab_problem, measurement_df)
    return _pypesto_problem.objective._objectives[0](eval_params)


def compute_likelihood_parallel(petab_problem, workflow_samples, test_data, n_jobs: int) -> np.ndarray:
    import amici
    import logging
    amici.swig_wrappers.logger.setLevel(logging.CRITICAL)
    logging.getLogger("pypesto").setLevel(logging.ERROR)
    logging.getLogger("pypesto.petab").setLevel(logging.ERROR)
    logging.getLogger("petab").setLevel(logging.ERROR)
    logging.getLogger("bayesflow").setLevel(logging.ERROR)

    sim_list = test_data['sim_data_df']
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_likelihood)(petab_problem, sim_list[i], workflow_samples[i])
        for i in range(len(sim_list))
    )
    obj_val = np.asarray(results, dtype=float).reshape(len(sim_list), 1)
    return np.concatenate((workflow_samples, obj_val), axis=-1)


def sample_in_batches(data, workflow, num_samples, batch_size=64, sampler_settings=None) -> dict:
    posterior_samples = None
    for i in range(0, len(data['sim_data']), batch_size):
        batch_data = {k: v[i:i + batch_size] for k, v in data.items() if k != 'sim_data_df'}
        if sampler_settings is None:
            batch_samples = workflow.sample(conditions=batch_data, num_samples=num_samples)
        else:
            batch_samples = workflow.sample(conditions=batch_data, num_samples=num_samples,
                                            **sampler_settings)
        if i == 0:
            posterior_samples = batch_samples
        else:
            for key in posterior_samples.keys():
                posterior_samples[key] = np.vstack([posterior_samples[key], batch_samples[key]])
    return posterior_samples


def compute_metrics(model_name, workflow, test_data, sampler_settings, petab_problem, pypesto_problem,
                    num_samples_inference, n_jobs=1):
    metrics = []

    # augment test data
    test_data_aug = compute_likelihood_parallel(petab_problem, test_data['amici_params'], test_data, n_jobs=n_jobs)

    for solver_name in sampler_settings:
        if solver_name.startswith('sde') and not model_name.startswith('diffusion'):
            continue
        if 'consistency' in model_name:
            workflow_samples_dict = sample_in_batches(test_data, workflow, num_samples_inference)
        else:
            workflow_samples_dict = sample_in_batches(test_data, workflow, num_samples_inference,
                                                      sampler_settings=sampler_settings[solver_name])

        metrics.append({
            'model': model_name,
            'sampler': solver_name,
            'nrmse': root_mean_squared_error(workflow_samples_dict, test_data, aggregation=np.nanmedian)['values'].mean(),
            'nrmse_mad': root_mean_squared_error(workflow_samples_dict, test_data, aggregation=mad)['values'].mean(),
            'posterior_contraction': posterior_contraction(workflow_samples_dict, test_data, aggregation=np.nanmedian)['values'].mean(),
            'posterior_contraction_mad': posterior_contraction(workflow_samples_dict, test_data, aggregation=mad)['values'].mean(),
            'posterior_calibration_error': calibration_error(workflow_samples_dict, test_data, aggregation=np.nanmedian)['values'].mean(),
            'posterior_calibration_error_mad': calibration_error(workflow_samples_dict, test_data, aggregation=mad)['values'].mean(),
            'count_nan_data': np.sum(np.isnan(get_samples_from_dict(workflow_samples_dict, pypesto_problem)).any(axis=(1, 2)))
        })

        del workflow_samples_dict  # free memory

        # compute C2ST with augmented data (objective value)
        if 'consistency' in model_name:
            workflow_samples_dict = sample_in_batches(test_data, workflow, num_samples=1)
        else:
            workflow_samples_dict = sample_in_batches(test_data, workflow, num_samples=1,
                                                      sampler_settings=sampler_settings[solver_name])
        workflow_samples = get_samples_from_dict(workflow_samples_dict, pypesto_problem)[:, 0]
        workflow_samples_aug = compute_likelihood_parallel(petab_problem, workflow_samples, test_data, n_jobs=n_jobs)

        # compute C2ST
        workflow_samples_aug = workflow_samples_aug[~np.isnan(workflow_samples_aug).any(axis=1)]
        test_data_aug = test_data_aug[~np.isnan(test_data_aug).any(axis=1)]
        logging.info(f"{workflow_samples_aug.shape[0]} workflow samples and {test_data_aug.shape[0]} test data samples.")
        metrics[-1]['c2st'] = classifier_two_sample_test(workflow_samples_aug, test_data_aug)
    return metrics
