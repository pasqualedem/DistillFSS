from datetime import datetime
import os
import uuid
import click
import einops
import torch
import torch.nn as nn
import plotly.express as px
import plotly.figure_factory as ff

from torchvision.transforms.functional import resize
from torchmetrics import F1Score, MetricCollection, Precision, Recall
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
import yaml

from distillfss.data import get_testloaders
from distillfss.data.utils import BatchKeys
from distillfss.data.utils import get_support_batch
from distillfss.models import MODEL_REGISTRY, build_distillator, build_model
from distillfss.models.loss import get_loss
from distillfss.substitution import get_substitutor
from distillfss.test import test
from distillfss.utils.logger import get_logger
from distillfss.utils.tracker import WandBTracker, wandb_experiment
from distillfss.utils.utils import ResultDict, linearize_metrics, load_yaml, to_device
from distillfss.utils.grid import create_experiment, make_grid


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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F
import einops
from tqdm import tqdm
from sklearn.manifold import TSNE


def tsne_visualization(features_per_class, id2class, dataset_name):
    """
    Generates a t-SNE visualization of feature embeddings.
    
    Args:
        features_per_class (dict): Dictionary of features per class.
        id2class (dict): Mapping from class ID to class name.
        dataset_name (str): Name of the dataset.
    """
    # Prepare data for t-SNE
    all_features = []
    labels = []
    
    for class_id, features in features_per_class.items():
        features = features.cpu().numpy()  # Convert to NumPy
        # sample 50 items per class
        features = features[:250]
        all_features.append(features)
        labels.extend([id2class[class_id]] * features.shape[0])

    all_features = np.vstack(all_features)  # Shape (num_samples, feature_dim)
    
    # Reduce to 2D with t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedded_features = tsne.fit_transform(all_features)  # Shape (num_samples, 2)

    df = pd.DataFrame(embedded_features, columns=["x", "y"])
    df["label"] = labels

    # Create scatter plot using Plotly
    fig = px.scatter(
        df, x="x", y="y", color="label", title=f"t-SNE Visualization of {dataset_name} Features",
        opacity=0.6, color_discrete_sequence=px.colors.qualitative.T10
    )

    return fig 
    

def calculate_dataset_statistics(model, dataloader, tracker, logger, dataset_name, device, reference_features=None):
    """
    Computes intra-class variance, foreground variance, and inter-class distances (centroid distances)
    with global feature normalization.

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
            features = F.interpolate(features, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            features = einops.rearrange(features, "b c h w -> (b h w) c")

            gt_resized = einops.rearrange(gt, "b h w -> (b h w)")

            for class_id, class_name in dataloader.dataset.id2class.items():
                mask = gt_resized == class_id
                if mask.any():
                    if class_id not in features_per_class:
                        features_per_class[class_id] = []
                    features_per_class[class_id].append(features[mask].mean(dim=0, keepdim=True).cpu())

    # Global normalization
    if features_per_class:
        all_features = torch.cat([torch.cat(f, dim=0) for f in features_per_class.values()], dim=0)
        # all_norm_features = F.normalize(all_features, p=2, dim=-1)  # L2 normalize globally
        all_norm_features = all_features

        # Update per-class feature lists with normalized values
        index = 0
        for class_id in features_per_class.keys():
            feature_list = features_per_class[class_id]
            num_features = sum(f.shape[0] for f in feature_list)
            features_per_class[class_id] = all_norm_features[index : index + num_features]
            index += num_features

    variance_results = {}
    class_centroids = {}

    # Compute intra-class stdev & centroids
    for class_id, all_norm_features in features_per_class.items():
        class_centroid = torch.mean(all_norm_features, dim=0)  # Compute centroid (mean feature vector)
        class_centroids[class_id] = class_centroid

        std_dev = torch.std(all_norm_features, dim=0).mean().item()  # Compute standard deviation
        variance = std_dev ** 2  # Variance = std_dev^2
        variance_results[f"{dataset_name}.variance_{dataloader.dataset.id2class[class_id]}"] = variance
        variance_results[f"{dataset_name}.std_dev_{dataloader.dataset.id2class[class_id]}"] = std_dev

    # Compute foreground variance
    if features_per_class:
        all_foreground_features = torch.cat([f for f in features_per_class.values()], dim=0)
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
                    # distance_matrix[i, j] = torch.norm(centroid_matrix[i] - centroid_matrix[j], p=2)  # Euclidean distance
                    distance_matrix[i, j] = 1 - F.cosine_similarity(centroid_matrix[i], centroid_matrix[j], dim=0)

        # Convert to Pandas DataFrame
        distance_matrix_df = pd.DataFrame(distance_matrix.cpu().numpy(), index=class_names, columns=class_names)

        # Log distance matrix as a table
        tracker.add_table(
            tag=f"{dataset_name}.interclass_distance_matrix",
            data=distance_matrix.cpu(),
            columns=class_names,
            rows=class_names
        )

        fig_heatmap = ff.create_annotated_heatmap(
            z=distance_matrix_df.values,
            x=distance_matrix_df.columns.tolist(),
            y=distance_matrix_df.index.tolist(),
            colorscale="viridis",
            showscale=True
        )

        fig_heatmap.update_layout(
            title=f"{dataset_name} Inter-Class Distance Matrix",
            xaxis_title="Classes",
            yaxis_title="Classes"
        )

        # Log the figure
        tracker.add_figure(f"{dataset_name}.interclass_distance_plot", fig_heatmap)
        
        print("Generating t-SNE visualization...")
        fig = tsne_visualization(features_per_class, dataloader.dataset.id2class, dataset_name)
        tracker.add_figure(f"{dataset_name}.tsne_plot", fig)

    # Log results
    tracker.add_summary(variance_results)
    for key, value in variance_results.items():
        logger.info(f"{key}: {value}")
        
    if reference_features is not None:
        # Compute cosine similarity between reference and current dataset features
        concat_features = torch.cat([all_features, reference_features], dim=0)
        # concat_features = F.normalize(concat_features, p=2, dim=-1)  # L2 normalize globally
        reference_features = concat_features[-reference_features.shape[0]:]
        dataset_features = concat_features[:-reference_features.shape[0]]
        
        reference_centroid = reference_features.mean(dim=0)
        dataset_centroid = dataset_features.mean(dim=0)
        
        distance = 1 - F.cosine_similarity(reference_centroid, dataset_centroid, dim=0)
        tracker.add_scalar(f"{dataset_name}.reference_distance", distance.item())
        
        features_per_dataset = {
            0: reference_features,
            1: dataset_features
        }
        id2dataset = {
            0: "Reference",
            1: dataset_name
        }
        
        fig = tsne_visualization(features_per_dataset, id2dataset, dataset_name)
        tracker.add_figure(f"{dataset_name}.tsne_rerference_plot", fig)

    return variance_results, all_features


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
    
    reference_dataset = parameters["reference"]
    reference_loader = test_loaders.pop(reference_dataset)
    
    _, reference_features = calculate_dataset_statistics(model, reference_loader, tracker, logger, reference_dataset, device, reference_features=None)
    
    for dataset_name, dataloader in test_loaders.items():        
        calculate_dataset_statistics(model, dataloader, tracker, logger, dataset_name, device, reference_features=reference_features)

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