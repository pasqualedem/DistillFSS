from datetime import datetime
import os
import uuid
import click
import torch
import torch.nn as nn

from torchvision.transforms.functional import resize
from torchmetrics import F1Score, MetricCollection, Precision, Recall
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
import yaml

from fssweed.data import get_testloaders
from fssweed.data.utils import BatchKeys
from fssweed.data.utils import get_support_batch
from fssweed.models import MODEL_REGISTRY, build_distiller, build_model
from fssweed.models.loss import get_loss
from fssweed.substitution import get_substitutor
from fssweed.test import test
from fssweed.utils.logger import get_logger
from fssweed.utils.tracker import WandBTracker, wandb_experiment
from fssweed.utils.utils import ResultDict, linearize_metrics, load_yaml, to_device
from fssweed.utils.grid import create_experiment, make_grid


OUT_FOLDER = "out"


@click.group()
def cli():
    """Run a refinement or a grid"""
    pass


def refine_model(model, support_set, tracker: WandBTracker, logger, params, metrics, id2class=None):
    lr = params["lr"]
    max_iterations = params["max_iterations"]
    subsample = params.get("subsample")
    substitutor_name = params.get("substitutor")
    hot_parameters = params["hot_parameters"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = get_loss(params["loss"])

    model.train()
    if hot_parameters:
        for name, param in model.named_parameters():
            if any([
                hot_parameter in name
                for hot_parameter in hot_parameters
            ]):
                param.requires_grad = True
            else:
                param.requires_grad = False
            
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Training {name}")
    
    support_batch, support_gt = get_support_batch(support_set)
    
    substitutor = get_substitutor(substitutor_name, substitute=True, subsample=subsample)
    substitutor.reset(batch=(support_batch, support_gt))
    support_set_len = support_set[BatchKeys.IMAGES].shape[1]
    metric_update = 10

    bar = tqdm(range(max_iterations), desc="Training Progress")

    sequence_name = "predictions"
    tracker.create_image_sequence(sequence_name)
    for step in bar:
        loss_total = 0
        substitutor.reset(batch=(support_batch, support_gt))
        metrics.reset()
        
        for substep, (batch, gt) in enumerate(substitutor):
            result = model(batch)
            logits = result[ResultDict.LOGITS]
            loss_value = loss_fn(result, gt) / support_set_len
            loss_value.backward()
            loss_total += loss_value.item()
            outputs = logits.argmax(dim=1)
            metrics.update(outputs, gt)
            tracker.log_batch(
                batch,
                gt,
                outputs,
                step,
                substep,
                id2class,
                phase="train",
                sequence_name=sequence_name
            )
        
        optimizer.step()
        optimizer.zero_grad()
        
        if step % metric_update == 0:
            metric_values = linearize_metrics(metrics.compute(), id2class=id2class)
            f1_score = metric_values.get("MulticlassF1Score", 0)
            current_lr = optimizer.param_groups[0]['lr']
            tracker.log_metrics(metric_values)

        tracker.log_metric("loss", loss_total)
        bar.set_postfix({"Loss": loss_total, "F1 Score": f1_score, "Learning Rate": current_lr})
    tracker.add_image_sequence(sequence_name)
        
    # Get the training scores
    substitutor = get_substitutor(substitutor_name, substitute=True)
    support_batch, support_gt = get_support_batch(support_set)
    support_set_len = support_batch[BatchKeys.IMAGES].shape[1]
    metrics.reset()

    logger.info("Finished Training, extracting metrics...")
    substitutor.reset(batch=(support_batch, support_gt))
    model.eval()
    for batch, gt in substitutor: 
        with torch.no_grad():
            result = model(batch)
        logits = result[ResultDict.LOGITS]
        metrics.update(logits.argmax(dim=1), gt)
    metric_values = linearize_metrics(metrics.compute(), id2class=id2class)
    tracker.log_metrics({f"final_{k}": v for k, v in metric_values.items()})
    
    for k, v in metric_values.items():
        logger.info(f"Training - {k}: {v}")
    

def refine_and_test(parameters, log_filename=None):
    if log_filename is None:
        log_filename = str(uuid.uuid4())[:8]
        log_filename = os.path.join(OUT_FOLDER, log_filename)
        os.makedirs(OUT_FOLDER)
    # model filename is log filename but with .pt instead of .log
    model_filename = log_filename.replace(".log", ".pt")
        
    logger = get_logger("Refine", log_filename) 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on {device}")
        
    test_loaders = get_testloaders(
        parameters["dataset"],
        parameters["dataloader"]
    )
    image_size = parameters["dataset"]["preprocess"]["image_size"]
    
    model = build_model(parameters["model"])
    model.to(device)
    model.eval()
    
    tracker = wandb_experiment(parameters)
    
    for dataset_name, dataloader in test_loaders.items():
        id2class = dataloader.dataset.id2class
        metrics = MetricCollection(
            metrics=[
                    F1Score(
                        task="multiclass",
                        num_classes=dataloader.dataset.num_classes,
                        average="none",
                    ),
                    Precision(
                        task="multiclass",
                        num_classes=dataloader.dataset.num_classes,
                        average="none",
                    ),
                    Recall(
                        task="multiclass",
                        num_classes=dataloader.dataset.num_classes,
                        average="none",
                    ),
                    MulticlassJaccardIndex(
                        num_classes=dataloader.dataset.num_classes,
                        average="none",
                    )
            ]
        ).to(device)
        examples = dataloader.dataset.extract_prompts()
        examples = to_device(examples, device)
        prompt_to_use = parameters["test"].get("prompt_to_use", None)
        if prompt_to_use is not None:
            examples = {k: v[:prompt_to_use] for k, v in examples.items()}
        
        if "refinement" in parameters:
            with tracker.train():
                refine_model(model, examples, tracker, logger, parameters["refinement"], metrics.clone(), id2class)
            torch.save(model.state_dict(), model_filename)
        
        test(model, dataloader, examples, tracker, logger, dataset_name, image_size, metrics)

    tracker.end()

@cli.command("grid")
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a grid search",
)
def grid(parameters):
    parameters = load_yaml(parameters)
    grid_name = parameters.pop("grid")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    grid_name = f"{current_time}_{grid_name}"
    log_folder = os.path.join(OUT_FOLDER, grid_name)
    os.makedirs(log_folder)
    
    with open(os.path.join(log_folder, "hyperparams.yaml"), "w") as f:
        yaml.dump(parameters, f)
        
    runs_parameters = create_experiment(parameters)
    
    grid_logger = get_logger("Grid", f"{log_folder}/grid.log")
    grid_logger.info(f"Running {len(runs_parameters)} runs")
    for i, run_parameters in enumerate(runs_parameters):
        grid_logger.info(f"Running run {i+1}/{len(runs_parameters)}")
        refine_and_test(run_parameters, log_filename=f"{log_folder}/run_{i}.log")


@cli.command("run")
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a single run",
)
def run(parameters):
    parameters = load_yaml(parameters)
    refine_and_test(parameters)


if __name__ == "__main__":
    cli()