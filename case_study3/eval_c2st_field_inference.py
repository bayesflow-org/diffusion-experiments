import numpy as np
import pandas as pd
from tqdm import tqdm

import bayesflow as bf
import keras


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
        "validation": simulator.sample(int(training_kwargs["num_samples_inference"]*0.1)),
    }
    return data_dict, simulator


def get_summary_df(mode, shape, run_summary_path):
    df = pd.read_csv(run_summary_path)
    df["dataset.shape"] = df["dataset.shape"].apply(lambda x: int(x[1:-1].split(",")[0]))
    df = df.rename(columns={
        "Name": "wandb_name",
        "dataset.shape": "shape",
        "model.model_name": "model_type",
        "training.mode": "mode",
    })
    df = df[(df["mode"] == mode) & (df["shape"] == shape[0])]
    return df


def get_classifier_training_data(data_dict, approximator, run, key="train"):
    current_shape = run["shape"]
    conditions = keras.ops.convert_to_tensor(data_dict[key]["params_expanded"], dtype="float32")
    n_samples = conditions.shape[0]
    batch_size = 100
    # batching only possible if batch_size divides n_samples
    assert n_samples % batch_size == 0, "Batch size must divide the number of samples"
    for b in tqdm(range(n_samples // batch_size)):
        conditions_batch = conditions[b*batch_size:(b+1)*batch_size]
        trial = 0
        while trial < 10:
            z_batch = keras.random.normal((batch_size, current_shape, current_shape, 1))
            samples_batch = approximator.inference_network(
                z_batch,
                conditions=conditions_batch,
                inverse=True,
            )
            if not (
                    np.any(np.isnan(keras.ops.convert_to_numpy(samples_batch))) or
                    np.any(np.isinf(keras.ops.convert_to_numpy(samples_batch))) or
                    np.any(np.isneginf(keras.ops.convert_to_numpy(samples_batch))) or
                    np.any(np.isposinf(keras.ops.convert_to_numpy(samples_batch))) or
                    np.any(np.abs(keras.ops.convert_to_numpy(samples_batch)) > 100)
            ):
                break
            else:
                print(f"NaN values found in samples, retrying... (trial {trial + 1}/10)")
                trial += 1
        if trial == 10:
            raise ValueError("NaN values found in samples after 10 trials, aborting.")
        if b == 0:
            samples = keras.ops.convert_to_numpy(samples_batch)
        else:
            samples = np.concatenate([samples, keras.ops.convert_to_numpy(samples_batch)], axis=0)
    x = np.concatenate([
        np.concatenate([
            data_dict[key]["field"],
            data_dict[key]["params_expanded"],
        ], axis=-1),
        np.concatenate([
            samples,
            data_dict[key]["params_expanded"],
        ], axis=-1),
    ], axis=0)
    y = np.concatenate([
        np.ones((data_dict[key]["field"].shape[0], 1)),
        np.zeros((samples.shape[0], 1)),
    ], axis=0)
    del samples, conditions, conditions_batch, z_batch, samples_batch
    return x, y



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json
    import os
    from keras.utils import clear_session

    from network_helpers.resnet_summary import ResNetSummary
    #from network_helpers import padded_unet  # needed for loading approximator from ckpt

    #modes = ["online", "offline"]
    modes = ["online"]
    overwrite = False

    dataset_kwargs = {
        "seed": 84,
        "name": "grf_like",
    }
    training_kwargs = {
        "epochs": 500,
        "batch_size": 16,
        "num_samples_inference": 5000,
    }

    model_kwargs = {
        "shape_config_8": {
            "summary_dim": 1,
            "widths": 1 * (16,),
            "use_batchnorm": False,
            "dropout": 0.0,
        },
        "shape_config_16": {
            "summary_dim": 1,
            "widths": 1 * (16,),
            "use_batchnorm": False,
            "dropout": 0.0,
        },
        "shape_config_32": {
            "summary_dim": 1,
            "widths": 2 * [16, ],
            "use_batchnorm": False,
            "dropout": 0.0,
        },
        "shape_config_64": {
            "summary_dim": 1,
            "widths": 2 * [16, ],
            "use_batchnorm": False,
            "dropout": 0.0,
        },
        "shape_config_128": {
            "summary_dim": 1,
            "widths": 4 * [16, ],
            "use_batchnorm": False,
            "dropout": 0.0,
        },
        "shape_config_256": {
            "summary_dim": 1,
            "widths": 4 * [16, ],
            "use_batchnorm": False,
            "dropout": 0.0,
        },
    }

    parameter = "field"
    #shapes = [(2**n, 2**n) for n in range(7,8)]#range(3, 8)] # From 8x8 to 128x128 field sizes
    shapes = [(128, 128)]
    run_summary_path = "new_residualuvit.csv"
    for shape in shapes:
        for mode in modes:
            df = get_summary_df(mode, shape, run_summary_path)
            for runidx, run in tqdm(df.iterrows(), desc=f"models left at {mode} shape {shape}"):
                print(f"{mode} {run['wandb_name']}: shape {run['shape']}, model {run['model_type']}")
                proj_dir = os.path.join(f"{run['model_type']}", "NLE", f"{run['shape']}", run["wandb_name"])
                ckpt_dir = os.path.join(proj_dir, "checkpoints")
                if os.path.exists(os.path.join(ckpt_dir, "classifier.keras")) and not overwrite:
                    print("Classifier already exists, skipping...")
                    continue
                else:
                    print("Classifier does not exist or overwrite is True, proceeding with training...")
                ckpt_approx_file_path = os.path.join(ckpt_dir, f"{run['wandb_name']}.keras")
                figure_dir = os.path.join(proj_dir, "figures")

                print("get classifier data")
                dataset_kwargs = dataset_kwargs | {
                    "shape": 2 * (run["shape"],),
                }
                data_dict, simulator = get_data_dict(dataset_kwargs, training_kwargs)
                # see if train data already exists:
                if os.path.exists(os.path.join(proj_dir, "train_data.npz")):
                    # save train & val samples
                    print("load train data")
                    # load from npz
                    with np.load(os.path.join(proj_dir, "train_data.npz"), allow_pickle=True) as data:
                        train_data = data["arr_0"].item()
                    x = train_data["x"]
                    y = train_data["y"]
                    del train_data
                else:
                    print("generate train data")
                    print("load approximator")
                    approximator = keras.saving.load_model(ckpt_approx_file_path)
                    x, y = get_classifier_training_data(data_dict, approximator, run, key="train")
                    # save train samples
                    np.savez_compressed(os.path.join(proj_dir, "train_data.npz"), {"x": x, "y": y})
                    del approximator
                if os.path.exists(os.path.join(proj_dir, "val_data.npz")):
                    print("load test data")
                    with np.load(os.path.join(proj_dir, "val_data.npz"), allow_pickle=True) as data:
                        val_data = data["arr_0"].item()
                    x_val = val_data["x"]
                    y_val = val_data["y"]
                    del val_data
                else:
                    print("generate test data")
                    print("load approximator")
                    approximator = keras.saving.load_model(ckpt_approx_file_path)
                    x_val, y_val = get_classifier_training_data(data_dict, approximator, run, key="validation")
                    # save val samples
                    np.savez_compressed(os.path.join(proj_dir, "val_data.npz"), {"x": x_val, "y": y_val})
                    del approximator
                print("load classifier")
                classifier_kwargs = model_kwargs[f"shape_config_{run['shape']}"]
                inputs = keras.Input((run["shape"], run["shape"], 3))
                outputs = ResNetSummary(**classifier_kwargs)(inputs)
                classifier = keras.Model(inputs=inputs, outputs=outputs)
                classifier.summary()
                classifier.compile(
                    optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4),
                    loss=keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
                )

                print("train classifier")
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=500,
                    restore_best_weights=True,
                )
                history = classifier.fit(
                    x=x,
                    y=y,
                    epochs=training_kwargs["epochs"],
                    batch_size=training_kwargs["batch_size"],
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping],
                )
                print(
                    f"Best val accuracy: {max(history.history['val_accuracy'])} at epoch {np.argmax(history.history['val_accuracy'])}")
                fig = plt.figure()
                plt.plot(history.history["loss"], label="train loss")
                plt.plot(history.history["val_loss"], label="val loss")
                plt.plot(history.history["accuracy"], label="train accuracy")
                plt.plot(history.history["val_accuracy"], label="val accuracy")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.title(
                    f"Best val acc: {round(max(history.history['val_accuracy']), 4)} at epoch {np.argmax(history.history['val_accuracy'])}")
                plt.savefig(os.path.join(figure_dir, f"classifier_loss_{parameter}.pdf"))
                plt.close(fig)

                save_kwargs = {
                    "run_name": run["wandb_name"],
                    "classifier_kwargs": classifier_kwargs,
                    "classifier-result": float(max(history.history['val_accuracy'])),
                    "at-epoch": int(np.argmax(history.history['val_accuracy'])),
                }
                with open(os.path.join(proj_dir, f"classifier_results_{parameter}.json"), "w") as f:
                    json.dump(save_kwargs, f)

                # save model & classifier
                keras.saving.save_model(classifier, os.path.join(ckpt_dir, "classifier.keras"))

                # save train history
                np.save(os.path.join(proj_dir, f"classifier_history_{parameter}.npy"), history.history)

                clear_session()
                del x, y, x_val, y_val, classifier, history, data_dict, simulator, early_stopping
                print("Done.")
