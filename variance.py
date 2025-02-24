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
from fssweed.models import MODEL_REGISTRY, build_distillator, build_model
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

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_dataset_statistics(model, dataloader, tracker, logger, dataset_name, device):
    """
    Computes intra-class variance, foreground variance, and inter-class distances (centroid distances).

    Args:
        model (torch.nn.Module): Feature extractor model (e.g., ViT).
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        tracker (object): Tracker to log experiment metrics.
        logger (object): Logger to log progress and results.
        dataset_name (str): Name of the dataset.
        device (torch.device): Device to move data to.

    Returns:
        dict: Dictionary containing intra-class variance, foreground variance, 
              and inter-class distance metrics.
    """
    model.eval()
    features_per_class = {}

    with torch.no_grad():
        for images, gt in tqdm(dataloader):  
            images = images[BatchKeys.IMAGES][0].to(device)  
            gt = gt.to(device)  

            features = model(images)  
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

    variance_results = {}
    class_centroids = {}

    # Compute intra-class variance & centroids
    for class_id, feature_list in features_per_class.items():
        all_features = torch.cat(feature_list, dim=0)
        class_centroid = torch.mean(all_features, dim=0)  # Compute centroid (mean feature vector)
        class_centroids[class_id] = class_centroid

        variance = torch.var(all_features, dim=0).mean().item()
        variance_results[f"{dataset_name}.variance_{dataloader.dataset.id2class[class_id]}"] = variance

    # Compute foreground variance
    if features_per_class:
        all_foreground_features = torch.cat([torch.cat(f, dim=0) for f in features_per_class.values()], dim=0)
        foreground_variance = torch.var(all_foreground_features, dim=0).mean().item()
        variance_results[f"{dataset_name}.foreground_variance"] = foreground_variance

    # Compute inter-class centroid distances
    if len(class_centroids) > 1:
        class_ids = list(class_centroids.keys())
        class_names = [dataloader.dataset.id2class[cid] for cid in class_ids]

        centroid_matrix = torch.stack([class_centroids[cid] for cid in class_ids])  # (num_classes, feature_dim)
        num_classes = len(class_ids)
        distance_matrix = torch.zeros((num_classes, num_classes), device=device)

        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    distance_matrix[i, j] = torch.norm(centroid_matrix[i] - centroid_matrix[j], p=2)  # Euclidean distance

        # Convert to Pandas DataFrame
        distance_matrix_df = pd.DataFrame(distance_matrix.cpu().numpy(), index=class_names, columns=class_names)

        # Log distance matrix as a table
        tracker.add_table(
            tag=f"{dataset_name}.interclass_distance_matrix",
            data=distance_matrix.cpu(),
            columns=class_names,
            rows=class_names
        )

        # Create a figure for the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(distance_matrix_df, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5, ax=ax)
        ax.set_title(f"{dataset_name} Inter-Class Distance Matrix")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Classes")

        # Log the figure
        tracker.add_figure(f"{dataset_name}.interclass_distance_plot", fig)
        plt.close(fig)  # Close the figure to avoid memory issues

    # Log results
    tracker.add_summary(variance_results)
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
        calculate_dataset_statistics(model, dataloader, tracker, logger, dataset_name, device)

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