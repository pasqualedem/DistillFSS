from datetime import datetime
import os
import uuid
import click
import einops
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
    """Run a variance or a grid"""
    pass
    
import torch
import torch.nn.functional as F
import einops
from tqdm import tqdm
import torchvision.transforms.functional as TF

def calculate_dataset_variance(model, dataloader, tracker, logger, dataset_name, device):
    """
    Calculates the variance of features extracted from a dataset using the given model,
    including variance per class and foreground variance.
    
    Args:
        model (torch.nn.Module): Feature extractor model (e.g., ViT).
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        tracker (object): Object to track experiment metrics (e.g., WandB, TensorBoard, etc.).
        logger (object): Logger to log progress and results.
        dataset_name (str): Name of the dataset for logging purposes.
        device (torch.device): Device to move data to.

    Returns:
        dict: Variance of extracted features across the dataset, per class, and foreground.
    """
    model.eval()
    features_per_class = {}
    features_foreground = []
    
    with torch.no_grad():
        for images, gt in tqdm(dataloader):  # Assuming dataset returns (image, label)
            images = images[BatchKeys.IMAGES][0].to(device)  # Move to model's device
            gt = gt.to(device)  # Move ground truth to device
            
            features = model(images)  # Extract features
            b, c, h, w = features.shape
            features = einops.rearrange(features, "b c h w -> (b h w) c")
            
            gt_resized = F.interpolate(gt.unsqueeze(1).float(), size=(h, w), mode='nearest').squeeze(1).long()
            gt_resized = einops.rearrange(gt_resized, "b h w -> (b h w)")
            
            for class_id, class_name in dataloader.dataset.id2class.items():
                mask = gt_resized == class_id
                if mask.any():
                    if class_id not in features_per_class:
                        features_per_class[class_id] = []
                    features_per_class[class_id].append(features[mask])
                    
            foreground_mask = gt_resized > 0
            if foreground_mask.any():
                features_foreground.append(features[foreground_mask])
    
    variance_results = {}
    for class_id, feature_list in features_per_class.items():
        all_features = torch.cat(feature_list, dim=0)
        variance = torch.var(all_features, dim=0)
        variance_results[f"{dataset_name}.variance_{dataloader.dataset.id2class[class_id]}"] = variance.mean().item()
    
    if features_foreground:
        all_foreground_features = torch.cat(features_foreground, dim=0)
        foreground_variance = torch.var(all_foreground_features, dim=0)
        variance_results[f"{dataset_name}.foreground_variance"] = foreground_variance.mean().item()
    
    # Log results
    tracker.log_metrics(variance_results)
    for key, value in variance_results.items():
        logger.info(f"{key}: {value}")
    
    return variance_results



def calculate_variance(parameters, log_filename=None):
    if log_filename is None:
        log_filename = str(uuid.uuid4())[:8]
        log_filename = os.path.join(OUT_FOLDER, log_filename)
        os.makedirs(OUT_FOLDER)
        
    logger = get_logger("Variance", log_filename) 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on {device}")
        
    test_loaders = get_testloaders(
        parameters["dataset"],
        parameters["dataloader"]
    )
    
    model = build_model(parameters["model"])
    model.to(device)
    model.eval()
    
    tracker = wandb_experiment(parameters)
    
    for dataset_name, dataloader in test_loaders.items():        
        calculate_dataset_variance(model, dataloader, tracker, logger, dataset_name, device)

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
        calculate_variance(run_parameters, log_filename=f"{log_folder}/run_{i}.log")


@cli.command("run")
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a single run",
)
def run(parameters):
    parameters = load_yaml(parameters)
    calculate_variance(parameters)


if __name__ == "__main__":
    cli()