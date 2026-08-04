import os
import numpy as np

import bayesflow as bf
from bayesflow.diagnostics.metrics import root_mean_squared_error
from network_helpers.resnet_summary import ResNetSummary

from keras.utils import clear_session
import matplotlib.pyplot as plt


def train_model_online(config, data_dict, conf_tuple, simulator):
    summary_network = ResNetSummary(**config["model"]["summary_kwargs"])
    inference_net_kwargs = conf_tuple[1] | {"subnet_kwargs": config["model"]["subnet_kwargs"]}
    inference_network = eval(conf_tuple[0])(**inference_net_kwargs)

    workflow = bf.workflows.BasicWorkflow(
        simulator=simulator,
        summary_network=summary_network,
        inference_network=inference_network,
        inference_variables=["log_std", "alpha"],
        summary_variables=["field"],
        standardize="summary_variables",
        checkpoint_filepath=config["ckpt_dir"],
        checkpoint_name=config["wandb_run_name"],
    )
    history = workflow.fit_online(
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        num_batches_per_epoch=int(config["training"]["num_samples_inference"] // config["training"]["batch_size"]),
        validation_data=data_dict["validation"],
    )

    return workflow, history

def train_model_offline(config, data_dict, conf_tuple):
    summary_network = ResNetSummary(**config["model"]["summary_kwargs"])
    inference_net_kwargs = conf_tuple[1] | {"subnet_kwargs": config["model"]["subnet_kwargs"]}
    inference_network = eval(conf_tuple[0])(**inference_net_kwargs)
    workflow = bf.workflows.BasicWorkflow(
        summary_network=summary_network,
        inference_network=inference_network,
        inference_variables=["log_std", "alpha"],
        summary_variables=["field"],
        standardize="summary_variables",
        checkpoint_filepath=config["ckpt_dir"],
        checkpoint_name=config["wandb_run_name"],
    )
    history = workflow.fit_offline(
        data=data_dict["train"],
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        validation_data=data_dict["validation"],
    )

    return workflow, history


def evaluate_model_online(workflow, history, data_dict, config, wandb_run):
    f = bf.diagnostics.plots.loss(history)
    f.savefig(os.path.join(config["figure_dir"], f"post_loss.pdf"))
    wandb_run.log({"train/post_loss": wandb.Image(f)}, commit=False)
    plt.close(f)
    for data_name, test_data  in zip(["validation"], [data_dict["validation"]]):
        plot_fns = {
            "recovery": bf.diagnostics.recovery,
            "calibration": bf.diagnostics.calibration_ecdf,
        }
        fs = workflow.plot_custom_diagnostics(
            test_data=test_data,
            plot_fns=plot_fns,
        )
        for k, f in fs.items():
            f.savefig(os.path.join(config["figure_dir"], f"{data_name}_{k}.pdf"))
            wandb_run.log({f"{data_name}/{k}": wandb.Image(f)}, commit=False)
            wandb_run.log({f"{k}": wandb.Image(f)}, commit=False)
            plt.close(f)

        targets = {"alpha": test_data["alpha"], "log_std": test_data["log_std"]}
        # Numbers
        wf_samples = workflow.sample(num_samples=1000, conditions={"field": test_data["field"]})
        nrmse = root_mean_squared_error(targets=targets, estimates=wf_samples)
        ce = bf.diagnostics.metrics.calibration_error(targets=targets, estimates=wf_samples)
        clg = bf.diagnostics.metrics.calibration_log_gamma(targets=targets, estimates=wf_samples)
        validation_dict = {
            "nrmse": nrmse,
            "ce": ce,
            "clg": clg,
        }
        validation_path = os.path.join(config["proj_dir"], f"{data_name}_numbers.npz")
        kv = {}
        for k, m in validation_dict.items():
            v = np.asarray(m["values"]).astype(float)
            v[~np.isfinite(v)] = np.nan
            kv[f"{k}_values"] = v
            kv[f"{k}_names"] = np.asarray(m["variable_names"], dtype=object)
            kv[f"{k}_metric_name"] = np.asarray(m["metric_name"], dtype=object)
            wandb_run.log({
                f"{data_name}/{k}_logstd": round(v[0], 4),
                f"{data_name}/{k}_alpha": round(v[1], 4),

            }, commit=False)
            wandb_run.log({
                f"{k}_logstd": round(v[0], 4),
                f"{k}_alpha": round(v[1], 4),

            }, commit=True)
            wandb_run.summary[f"{k}"] = v
        np.savez_compressed(validation_path, **kv)

def evaluate_model_offline(workflow, history, data_dict, config, wandb_run):
    f = bf.diagnostics.plots.loss(history)
    f.savefig(os.path.join(config["figure_dir"], f"post_loss.pdf"))
    wandb_run.log({"train/post_loss": wandb.Image(f)}, commit=False)
    plt.close(f)
    small_training_data = {k: v[:len(data_dict["validation"]["field"])] for k, v in data_dict["train"].items()}
    for data_name, test_data  in zip(["train", "validation"], [small_training_data, data_dict["validation"]]):
        plot_fns = {
            "recovery": bf.diagnostics.recovery,
            "calibration": bf.diagnostics.calibration_ecdf,
        }
        fs = workflow.plot_custom_diagnostics(
            test_data=test_data,
            plot_fns=plot_fns,
        )
        for k, f in fs.items():
            f.savefig(os.path.join(config["figure_dir"], f"{data_name}_{k}.pdf"))
            wandb_run.log({f"{data_name}/{k}": wandb.Image(f)}, commit=False)
            wandb_run.log({f"{k}": wandb.Image(f)}, commit=False)
            plt.close(f)

        targets = {"alpha": test_data["alpha"], "log_std": test_data["log_std"]}
        # Numbers
        wf_samples = workflow.sample(num_samples=1000, conditions={"field": test_data["field"]})
        nrmse = root_mean_squared_error(targets=targets, estimates=wf_samples)
        ce = bf.diagnostics.metrics.calibration_error(targets=targets, estimates=wf_samples)
        clg = bf.diagnostics.metrics.calibration_log_gamma(targets=targets, estimates=wf_samples)
        validation_dict = {
            "nrmse": nrmse,
            "ce": ce,
            "clg": clg,
        }
        validation_path = os.path.join(config["proj_dir"], f"{data_name}_numbers.npz")
        kv = {}
        for k, m in validation_dict.items():
            v = np.asarray(m["values"]).astype(float)
            v[~np.isfinite(v)] = np.nan  # keep file valid
            kv[f"{k}_values"] = v
            kv[f"{k}_names"] = np.asarray(m["variable_names"], dtype=object)
            kv[f"{k}_metric_name"] = np.asarray(m["metric_name"], dtype=object)
            wandb_run.log({
                f"{data_name}/{k}_logstd": round(v[0], 4),
                f"{data_name}/{k}_alpha": round(v[1], 4),

            }, commit=False)
            wandb_run.log({
                f"{k}_logstd": round(v[0], 4),
                f"{k}_alpha": round(v[1], 4),

            }, commit=True)
            wandb_run.summary[f"{k}"] = v
        np.savez_compressed(validation_path, **kv)


def get_data_dict(dataset_kwargs, training_kwargs):
    from FyeldGenerator import generate_field

    def generate_power_spectrum(alpha, scale):
        def power_spectrum(k):
            base = np.power(k, -alpha) * scale ** 2
            return base

        return power_spectrum

    rng = np.random.default_rng(seed=dataset_kwargs["seed"])
    shape=dataset_kwargs["shape"]
    def distribution(shape):
        a = rng.normal(loc=0, scale=1., size=shape)
        b = rng.normal(loc=0, scale=1., size=shape)
        return a + 1j * b

    def prior():
        return {"log_std": rng.normal(scale=0.3), "alpha": rng.normal(loc=3, scale=0.5)}

    def likelihood(log_std, alpha):
        field = generate_field(
            distribution, generate_power_spectrum(alpha, np.exp(log_std)), shape, unit_length=1/(np.abs(alpha) + 1e-7)
        )
        return {"field": field[..., None]}

    simulator = bf.make_simulator([prior, likelihood])
    data_dict = {
        "train": simulator.sample(training_kwargs["num_samples_inference"]),
        "validation": simulator.sample(int(training_kwargs["num_samples_inference"]*0.05)),
    }
    return data_dict, simulator


if __name__ == "__main__":
    import wandb
    training_kwargs = {
        "epochs": 500,
        "batch_size": 32,
        "num_samples_inference": 5000,
        "total_samples": int(5000) // 32 * 500,
    }
    integrate_kwargs = {"method": "tsit5", "steps": "adaptive"}
    shapes = [(2 ** n, 2 ** n) for n in range(3, 9)] # From 8x8 to 256x256
    modes = ["online", "offline"]
    dataset_kwargs = {
        "seed": 42,
        "name": "grf_post",
    }

    MODELS = {
        "consistency_model": ("bf.networks.ConsistencyModel", {
            "total_steps": training_kwargs["epochs"] * training_kwargs["batch_size"]}),
        "flow_matching": ("bf.networks.FlowMatching", {
            "integrate_kwargs": integrate_kwargs,
        }),
        "diffusion_edm_vp": ("bf.networks.DiffusionModel", {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "preserving"},
            "integrate_kwargs": integrate_kwargs,
        }),
    }

    model_kwargs = {
        "shape_config_8": {
            "summary_kwargs": {
                "summary_dim": 8,
                "widths": 1 * (16,),
                "use_batchnorm": False,
                "dropout": 0.0,
            },
            "subnet_kwargs": {
                "widths": 3 * (64,),
                "dropout": 0.0,
            },
        },
        "shape_config_16": {
            "summary_kwargs": {
                "summary_dim": 8,
                "widths": 1 * (16,),
                "use_batchnorm": False,
                "dropout": 0.0,
            },
            "subnet_kwargs": {
                "widths": 3 * (64,),
                "dropout": 0.0,
            },
        },
        "shape_config_32": {
            "summary_kwargs": {
                "summary_dim": 8,
                "widths": 2*[16,],
                "use_batchnorm": False,
                "dropout": 0.0,
            },
            "subnet_kwargs": {
                "widths": 3 * (64,),
                "dropout": 0.0,
            },
        },
        "shape_config_64": {
            "summary_kwargs": {
                "summary_dim": 8,
                "widths": 2*[16,],
                "use_batchnorm": False,
                "dropout": 0.0,
            },
            "subnet_kwargs": {
                "widths": 3 * (64,),
                "dropout": 0.0,
            },
        },
        "shape_config_128": {
            "summary_kwargs": {
                "summary_dim": 8,
                "widths": 4*[16,],
                "use_batchnorm": False,
                "dropout": 0.0,
            },
            "subnet_kwargs": {
                "widths": 3 * (64,),
                "dropout": 0.0,
            },
        },
        "shape_config_256": {
            "summary_kwargs": {
                "summary_dim": 8,
                "widths": 4*[16,],
                "use_batchnorm": False,
                "dropout": 0.0,
            },
            "subnet_kwargs": {
                "widths": 3 * (64,),
                "dropout": 0.0,
            },
        },

    }

    for shape in shapes:
        dataset_kwargs = dataset_kwargs | {
            "shape": shape,
        }
        data_dict, simulator = get_data_dict(dataset_kwargs, training_kwargs)
        for mode in modes:
            training_kwargs = training_kwargs | {"mode": mode}
            for model_name, conf_tuple in MODELS.items():
                current_model_kwargs = {"model_name": model_name} | model_kwargs[f"shape_config_{shape[0]}"] | {"model_name_kwargs": conf_tuple[1]}
                bf.utils.logging.info(f"Training {model_name} on {dataset_kwargs['name']} with shape {shape} in {mode} mode")
                config = {
                    "training": training_kwargs,
                    "model": current_model_kwargs,
                    "dataset": dataset_kwargs,
                }
                wandb_run = wandb.init(
                    project="case3-diffusion-review-grf-post",
                    entity="your_wandb_entity",
                    dir="wandb_results",
                    config=config,
                )
                proj_dir = os.path.join(f"{model_name}", "NPE", f"{shape[0]}", wandb_run.name)
                ckpt_dir = os.path.join(proj_dir, "checkpoints")
                figure_dir = os.path.join(proj_dir, "figures")
                os.makedirs(proj_dir, exist_ok=True)
                os.makedirs(ckpt_dir, exist_ok=True)
                os.makedirs(figure_dir, exist_ok=True)
                config["proj_dir"] = proj_dir
                config["ckpt_dir"] = ckpt_dir
                config["figure_dir"] = figure_dir
                config["wandb_run_name"] = wandb_run.name
                if mode == "online":
                    model, history = train_model_online(config, data_dict, conf_tuple, simulator)
                    evaluate_model_online(model, history, data_dict, config, wandb_run)
                else:
                    model, history = train_model_offline(config, data_dict, conf_tuple)
                    evaluate_model_offline(model, history, data_dict, config, wandb_run)
                clear_session()
                wandb_run.finish()
