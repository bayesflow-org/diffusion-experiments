import os
import numpy as np
import bayesflow as bf
from keras.utils import clear_session
import matplotlib.pyplot as plt

from network_helpers.padded_unet import PaddedUNetSubnet


def train_model(config, conf_tuple, data_dict, simulator, wandb_run):
    adapter = (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .rename("params_expanded", "inference_conditions")
        .rename("field", "inference_variables")
    )

    inference_net_kwargs = conf_tuple[1] | {
        "subnet": eval(config["model"]["subnet"]),
        "subnet_kwargs": config["model"][config["model"]["subnet"]]["subnet_kwargs"],
    }
    inference_network = eval(conf_tuple[0])(**inference_net_kwargs)
    if config["training"]["mode"] == "offline":
        workflow = bf.workflows.BasicWorkflow(
            inference_network=inference_network,
            adapter=adapter,
            standardize=None,
            checkpoint_filepath=config["ckpt_dir"],
            checkpoint_name=config["wandb_run_name"],
            initial_learning_rate=1e-4,
        )
        history = workflow.fit_offline(
            data=data_dict["train"],
            epochs=config["training"]["epochs"],
            validation_data=data_dict["validation"],
            batch_size=config["training"]["batch_size"],
        )
    else:
        workflow = bf.workflows.BasicWorkflow(
            simulator=simulator,
            inference_network=inference_network,
            adapter=adapter,
            standardize=None,
            checkpoint_filepath=config["ckpt_dir"],
            checkpoint_name=config["wandb_run_name"]
        )
        history = workflow.fit_online(
            epochs=config["training"]["epochs"],
            batch_size=config["training"]["batch_size"],
            num_batches_per_epoch=int(config["training"]["num_samples_inference"] // config["training"]["batch_size"]),
            validation_data=data_dict["validation"],
        )

    f = bf.diagnostics.plots.loss(history)
    f.savefig(os.path.join(config["figure_dir"], f"like_loss.pdf"))
    wandb_run.log({"train/like_loss": wandb.Image(f)}, commit=True)
    plt.close(f)

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
        log_std = rng.normal(scale=0.3)
        alpha = rng.normal(loc=3, scale=0.5)
        params_expanded = np.array([log_std, alpha])
        params_expanded = np.ones(shape + (2,)) * params_expanded[None, None, :]
        return {
            "log_std": log_std,
            "alpha": alpha,
            "params_expanded": params_expanded
        }

    def likelihood(log_std, alpha):
        field = generate_field(
            distribution, generate_power_spectrum(alpha, np.exp(log_std)), shape, unit_length=1/(np.abs(alpha) + 1e-7)
        )
        return {"field": field[..., None]}

    simulator = bf.make_simulator([prior, likelihood])
    data_dict = {
        "train": simulator.sample(training_kwargs["num_samples_inference"]),
        "validation": simulator.sample(int(training_kwargs["num_samples_validation"]*0.1)),
    }
    return data_dict, simulator


if __name__ == "__main__":
    import wandb
    training_kwargs = {
        "epochs": 500,
        "batch_size": 32,
        "num_samples_inference": 5000,
        "num_samples_validation": 500,
        "total_samples": int(5000) // 32 * 500,
    }
    modes = ["online", "offline"]
    shapes = [(2**n, 2**n) for n in range(3, 8)] # From 8x8 to 128x128 field sizes
    seed = 42
    dataset_kwargs = {
        "seed": seed,
        "name": "grf_like",
    }

    integrate_kwargs = {"method": "tsit5", "steps": "adaptive"}
    MODELS = {
        "flow_matching": ("bf.networks.FlowMatching", {
            "integrate_kwargs": integrate_kwargs,
        }),
        "consistency_model": ("bf.networks.ConsistencyModel", {
            "total_steps": training_kwargs["epochs"] * training_kwargs["batch_size"]}),
        "diffusion_edm_vp": ("bf.networks.DiffusionModel", {
            "noise_schedule": "edm",
            "prediction_type": "F",
            "schedule_kwargs": {"variance_type": "preserving"},
            "integrate_kwargs": integrate_kwargs,
        }),
    }

    model_kwargs = {
        "shape_config_8": {
            "subnet": "PaddedUNetSubnet",
            "PaddedUNetSubnet": {
                "subnet_kwargs": {
                    "widths": 2*[32,],
                    "activation": "mish",
                    "use_batchnorm": False,
                    "dropout": 0.0,
                    "num_res_blocks": 2*[2,],
                    "pad_size": 0,
                },
            },
        },
        "shape_config_16": {
            "subnet": "PaddedUNetSubnet",
            "PaddedUNetSubnet": {
                "subnet_kwargs": {
                    "widths": 3*[32,],
                    "activation": "mish",
                    "use_batchnorm": False,
                    "dropout": 0.0,
                    "num_res_blocks": 3*[2,],
                    "pad_size": 3,
                },
            },
        },
        "shape_config_32": {
            "subnet": "PaddedUNetSubnet",
            "PaddedUNetSubnet": {
                "subnet_kwargs": {
                    "widths": 3 * [32, ],
                    "activation": "mish",
                    "use_batchnorm": False,
                    "dropout": 0.0,
                    "num_res_blocks": 3 * [2, ],
                    "pad_size": 5,
                },
            },
        },
        "shape_config_64": {
            "subnet": "PaddedUNetSubnet",
            "PaddedUNetSubnet": {
                "subnet_kwargs": {
                    "widths": 4*[32,],
                    "activation": "mish",
                    "use_batchnorm": False,
                    "dropout": 0.0,
                    "num_res_blocks": 4*[2,],
                    "pad_size": 5,
                },
            },
        },
        "shape_config_128": {
            "subnet": "PaddedUNetSubnet",
            "PaddedUNetSubnet": {
                "subnet_kwargs": {
                    "widths": 5*[32,],
                    "activation": "mish",
                    "use_batchnorm": False,
                    "dropout": 0.0,
                    "num_res_blocks": 5 * [2, ],
                    "pad_size": 5,
                },
            },
        },
    }

    for shape in shapes:
        dataset_kwargs = dataset_kwargs | {"shape": shape}
        for mode in modes:
            training_kwargs = training_kwargs | {"mode": mode}
            for model_name, conf_tuple in MODELS.items():
                data_dict, simulator = get_data_dict(dataset_kwargs, training_kwargs)
                current_model_kwargs = {"model_name": model_name} | model_kwargs[f"shape_config_{shape[0]}"] | {
                    "model_name_kwargs": conf_tuple[1]}
                bf.utils.logging.info(
                    f"Training {model_name} on {dataset_kwargs['name']} with shape {shape} in {training_kwargs['mode']} mode")
                config = {
                    "training": training_kwargs,
                    "model": current_model_kwargs,
                    "dataset": dataset_kwargs,
                }
                wandb_run = wandb.init(
                    project="case3-diffusion-review-grf-like",
                    entity="your_wandb_entity",
                    dir="wandb_results",
                    config=config,
                )
                proj_dir = os.path.join(f"{model_name}", "NLE", f"{shape[0]}", wandb_run.name)
                ckpt_dir = os.path.join(proj_dir, "checkpoints")
                figure_dir = os.path.join(proj_dir, "figures")
                os.makedirs(proj_dir, exist_ok=True)
                os.makedirs(ckpt_dir, exist_ok=True)
                os.makedirs(figure_dir, exist_ok=True)
                config["proj_dir"] = proj_dir
                config["ckpt_dir"] = ckpt_dir
                config["figure_dir"] = figure_dir
                config["wandb_run_name"] = wandb_run.name
                train_model(config, conf_tuple, data_dict, simulator, wandb_run)
                clear_session()
                del data_dict, simulator
                wandb_run.finish()
