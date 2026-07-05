"""Print Case Study 1 model parameter counts for all SBIBM tasks."""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import os
import sys
import warnings
from contextlib import redirect_stderr
from pathlib import Path

warnings.filterwarnings("ignore", message="JULIA_SYSIMAGE_DIFFEQTORCH not set")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import keras
with open(os.devnull, "w") as _devnull, redirect_stderr(_devnull):
    import sbibm

    from case_study1.model_settings_benchmark import MODELS


def _format_int(value: int) -> str:
    return f"{value:,}"


def _markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "Task",
        "Dim theta",
        "Dim x",
        "Model",
        "Total params",
        "Trainable params",
        "Non-trainable params",
    ]
    keys = ["task", "dim_parameters", "dim_data", "model", "total", "trainable", "non_trainable"]

    rendered_rows = []
    for row in rows:
        rendered_rows.append(
            [
                str(row[key])
                if key not in {"total", "trainable", "non_trainable"}
                else _format_int(int(row[key]))
                for key in keys
            ]
        )

    widths = [
        max(len(header), *(len(row[i]) for row in rendered_rows))
        for i, header in enumerate(headers)
    ]
    lines = [
        "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    for row in rendered_rows:
        lines.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")

    return "\n".join(lines)


def _csv_table(rows: list[dict[str, object]]) -> str:
    headers = ["task", "dim_parameters", "dim_data", "model", "total", "trainable", "non_trainable"]
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row[key]) for key in headers))
    return "\n".join(lines)


def _count_variable_params(variables) -> int:
    return sum(int(variable.shape.num_elements()) for variable in variables)


def _model_kwargs_for_count(model_name: str) -> dict:
    kwargs = copy.deepcopy(MODELS[model_name][1])

    # The full benchmark value only controls the precomputed training schedule.
    # It does not change the learnable network weights, but makes build() slow.
    if model_name == "consistency_model":
        kwargs["total_steps"] = 10

    return kwargs


def count_parameters(model_name: str, task_name: str) -> dict[str, object]:
    keras.backend.clear_session()
    task = sbibm.get_task(task_name)
    model_cls = MODELS[model_name][0]
    network = model_cls(**_model_kwargs_for_count(model_name))
    network.build((None, task.dim_parameters), (None, task.dim_data))

    total = int(network.count_params())
    trainable = _count_variable_params(network.trainable_variables)
    del network
    keras.backend.clear_session()
    gc.collect()

    return {
        "task": task_name,
        "dim_parameters": task.dim_parameters,
        "dim_data": task.dim_data,
        "model": model_name,
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output table format.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=sbibm.get_available_tasks(),
        help="Task to include. May be passed multiple times. Defaults to all tasks.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODELS),
        help="Model to include. May be passed multiple times. Defaults to all models.",
    )
    return parser.parse_args()


def main() -> None:
    logging.getLogger("bayesflow").setLevel(logging.ERROR)

    args = parse_args()
    task_names = args.task or sbibm.get_available_tasks()
    model_names = args.model or ("diffusion_edm_vp", "diffusion_edm_vp_transformer")

    rows = [
        count_parameters(model_name=model_name, task_name=task_name)
        for task_name in task_names
        for model_name in model_names
    ]

    if args.format == "csv":
        print(_csv_table(rows))
    else:
        print(_markdown_table(rows))


if __name__ == "__main__":
    main()

# | Task                    | Dim theta | Dim x | Model                        | Total params | Trainable params | Non-trainable params |
# | ----------------------- | --------- | ----- | ---------------------------- | ------------ | ---------------- | -------------------- |
# | lotka_volterra          | 4         | 20    | diffusion_edm_vp             | 514,068      | 514,068          | 0                    |
# | lotka_volterra          | 4         | 20    | diffusion_edm_vp_transformer | 980,325      | 980,325          | 0                    |
# | gaussian_mixture        | 2         | 2     | diffusion_edm_vp             | 508,434      | 508,434          | 0                    |
# | gaussian_mixture        | 2         | 2     | diffusion_edm_vp_transformer | 977,751      | 977,751          | 0                    |
# | gaussian_linear_uniform | 10        | 10    | diffusion_edm_vp             | 514,586      | 514,586          | 0                    |
# | gaussian_linear_uniform | 10        | 10    | diffusion_edm_vp_transformer | 979,903      | 979,903          | 0                    |
# | two_moons               | 2         | 2     | diffusion_edm_vp             | 508,434      | 508,434          | 0                    |
# | two_moons               | 2         | 2     | diffusion_edm_vp_transformer | 977,751      | 977,751          | 0                    |
# | bernoulli_glm           | 10        | 10    | diffusion_edm_vp             | 514,586      | 514,586          | 0                    |
# | bernoulli_glm           | 10        | 10    | diffusion_edm_vp_transformer | 979,903      | 979,903          | 0                    |
# | sir                     | 2         | 10    | diffusion_edm_vp             | 510,482      | 510,482          | 0                    |
# | sir                     | 2         | 10    | diffusion_edm_vp_transformer | 978,775      | 978,775          | 0                    |
# | gaussian_linear         | 10        | 10    | diffusion_edm_vp             | 514,586      | 514,586          | 0                    |
# | gaussian_linear         | 10        | 10    | diffusion_edm_vp_transformer | 979,903      | 979,903          | 0                    |
# | slcp                    | 5         | 8     | diffusion_edm_vp             | 511,509      | 511,509          | 0                    |
# | slcp                    | 5         | 8     | diffusion_edm_vp_transformer | 978,927      | 978,927          | 0                    |
# | slcp_distractors        | 5         | 100   | diffusion_edm_vp             | 535,061      | 535,061          | 0                    |
# | slcp_distractors        | 5         | 100   | diffusion_edm_vp_transformer | 990,703      | 990,703          | 0                    |
# | bernoulli_glm_raw       | 10        | 100   | diffusion_edm_vp             | 537,626      | 537,626          | 0                    |
# | bernoulli_glm_raw       | 10        | 100   | diffusion_edm_vp_transformer | 991,423      | 991,423          | 0                    |
