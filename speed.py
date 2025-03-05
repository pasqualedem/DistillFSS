from datetime import datetime
import os
import uuid
import click
import einops
import torch
import torch.nn as nn
import yaml

from tqdm import tqdm
from PIL import Image

from fssweed.data import get_preprocessing
from fssweed.models import build_model
from fssweed.utils.grid import ParallelRun, create_experiment
from fssweed.utils.logger import get_logger
from fssweed.utils.tracker import WandBTracker, wandb_experiment
from fssweed.utils.utils import load_yaml, to_device
from fssweed.data.utils import BatchKeys

OUT_FOLDER = "out"


@click.group()
def cli():
    """Run a speed test or a grid"""
    pass


def measure_speed(model, params, logger, tracker: WandBTracker):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    batch_size = params["batch_size"]
    num_batches = params["num_batches"]
    image_size = params["image_size"]

    transforms = get_preprocessing(
        {
            "preprocess": {
                "image_size": image_size,
            }
        }
    )

    num_shots = params["num_shots"]
    num_ways = params["num_ways"]
    
    images = [
        [transforms(Image.fromarray((torch.randn(image_size, image_size, 3).numpy() * 255).astype('uint8'))) for _ in range(num_shots + 1)]
        for _ in range(batch_size)
    ]
    images = torch.stack([torch.stack(batch) for batch in images]).to(device)

    dummy_input = {
        BatchKeys.IMAGES: images,
        BatchKeys.PROMPT_MASKS: torch.randint(
            0, 2, (batch_size, num_shots, num_ways + 1, image_size, image_size)
        )
        .float()
        .to(device),
        BatchKeys.FLAG_EXAMPLES: torch.ones(batch_size, num_shots, num_ways + 1)
        .to(device)
        .bool(),
        BatchKeys.DIMS: einops.repeat(
            torch.tensor([image_size, image_size]),
            "d -> b n d",
            b=batch_size,
            n=num_shots + 1,
        ).to(device),
    }

    logger.info(
        f"Measuring speed with batch size: {batch_size}, num shots: {num_shots}, num ways: {num_ways}, size: {image_size}x{image_size}"
    )
    logger.info(f"Number of batches: {num_batches}")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Measuring Speed"):
            _ = model(dummy_input)
    end_event.record()
    torch.cuda.synchronize()
    total_time = (
        start_event.elapsed_time(end_event) / 1000
    )  # Convert milliseconds to seconds

    # Release memory
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy_input)
    required_memory = torch.cuda.max_memory_allocated(device) / (
        1024**2
    )  # Convert bytes to megabytes
    logger.info(f"Peak memory usage: {required_memory:.2f} MB")
    tracker.log_metric("peak_memory_usage", required_memory)

    avg_time_per_batch = total_time / num_batches
    logger.info(f"Total time: {total_time:.4f} seconds")
    logger.info(f"Average time per batch: {avg_time_per_batch:.4f} seconds")

    tracker.log_metric("total_time", total_time)
    tracker.log_metric("avg_time_per_batch", avg_time_per_batch)


def speed_test(
    parameters, run_name=None, log_params=True, log_model=True, log_on_file=True
):
    if run_name is None:
        run_name = str(uuid.uuid4())[:8] + ".log"
        run_name = os.path.join(OUT_FOLDER, run_name)
        os.makedirs(OUT_FOLDER, exist_ok=True)
    params_filename = run_name + ".yaml"
    if log_params:
        with open(params_filename, "w") as f:
            yaml.dump(parameters, f)

    log_filename = run_name + ".log" if log_on_file else None
    logger = get_logger("SpeedTest", log_filename)
    logger.info("parameters:")
    logger.info(parameters)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on {device}")

    model = build_model(parameters["model"])
    model.to(device)
    model.eval()

    tracker = wandb_experiment(parameters, logger=logger)

    measure_speed(model, parameters["speed_test"], logger, tracker)

    tracker.end()


@cli.command("grid")
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a grid search",
)
@click.option(
    "--parallel",
    default=False,
    is_flag=True,
    help="Run the grid in parallel",
)
@click.option(
    "--only_create",
    default=False,
    is_flag=True,
    help="Only create the slurm scripts",
)
def grid(parameters, parallel, only_create=False):
    parameters = load_yaml(parameters)
    grid_name = parameters.pop("grid")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    grid_name = f"{current_time}_{grid_name}"
    log_folder = os.path.join(OUT_FOLDER, grid_name)

    runs_parameters = create_experiment(parameters)

    os.makedirs(log_folder)
    with open(os.path.join(log_folder, "hyperparams.yaml"), "w") as f:
        yaml.dump(parameters, f)

    grid_logger = get_logger("Grid", f"{log_folder}/grid.log")
    grid_logger.info(f"Running {len(runs_parameters)} runs")
    for i, run_parameters in enumerate(runs_parameters):
        run_name = f"{log_folder}/run_{i}"
        if parallel:
            run = ParallelRun(
                run_parameters,
                multi_gpu=False,
                logger=grid_logger,
                run_name=run_name,
            )
            run.launch(
                only_create=only_create,
                script_args=[
                    "--disable_log_params",
                    "--disable_log_model",
                    "--disable_log_on_file",
                ],
            )
        else:
            grid_logger.info(f"Running run {i+1}/{len(runs_parameters)}")
            speed_test(run_parameters, run_name=run_name)


@cli.command("run")
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a single run",
)
@click.option("--run_name", default=None, help="Name of the run")
@click.option(
    "--disable_log_params",
    default=False,
    is_flag=True,
    help="Disable Log the parameters",
)
@click.option(
    "--disable_log_model", default=False, is_flag=True, help="Disable Log the model"
)
@click.option(
    "--disable_log_on_file", default=False, is_flag=True, help="Disable Log on file"
)
def run(
    parameters,
    run_name=None,
    disable_log_params=False,
    disable_log_model=False,
    disable_log_on_file=False,
):
    parameters = load_yaml(parameters)
    speed_test(
        parameters,
        run_name,
        not disable_log_params,
        not disable_log_model,
        not disable_log_on_file,
    )


if __name__ == "__main__":
    cli()
