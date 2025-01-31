from copy import deepcopy
import click
import torch
import torch.nn as nn

from torchvision.transforms.functional import resize
from torchmetrics import F1Score, MetricCollection, Precision, Recall
from tqdm import tqdm

from fssweed.data import get_testloaders
from fssweed.data.utils import BatchKeys
from fssweed.models import MODEL_REGISTRY, build_model
from fssweed.models.loss import get_loss
from fssweed.substitution import Substitutor
from fssweed.utils.logger import get_logger
from fssweed.utils.tracker import WandBTracker, wandb_experiment
from fssweed.utils.utils import ResultDict, linearize_metrics, load_yaml, to_device
from fssweed.utils.grid import make_grid


@click.group()
def cli():
    """Run a refinement or a grid"""
    pass


logger = get_logger("Refinement")


def get_support_batch(examples):
    support_batch = {
        BatchKeys.IMAGES: examples[BatchKeys.IMAGES].unsqueeze(0).clone(),
        BatchKeys.PROMPT_MASKS: examples[BatchKeys.PROMPT_MASKS].unsqueeze(0).clone(),
        BatchKeys.FLAG_MASKS: examples[BatchKeys.FLAG_MASKS].unsqueeze(0).clone(),
        BatchKeys.FLAG_EXAMPLES: examples[BatchKeys.FLAG_EXAMPLES].unsqueeze(0).clone(),
        BatchKeys.DIMS: examples[BatchKeys.DIMS].unsqueeze(0).clone()
    }
    support_gt = examples[BatchKeys.PROMPT_MASKS].argmax(dim=1).unsqueeze(0)
    return support_batch, support_gt


def refine_model(model, support_set, tracker: WandBTracker, params, metrics, id2class=None):
    lr = params["lr"]
    max_iterations = params["max_iterations"]
    subsample = params.get("subsample")
    hot_parameters = params["hot_parameters"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = get_loss(params["loss"])

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
    
    substitutor = Substitutor(substitute=True, subsample=subsample)
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
            loss_value = loss_fn(logits, gt) / support_set_len
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
    substitutor = Substitutor(substitute=True)
    support_batch, support_gt = get_support_batch(support_set)
    support_set_len = support_batch[BatchKeys.IMAGES].shape[1]
    metrics.reset()

    logger.info("Finished Training, extracting metrics...")
    substitutor.reset(batch=(support_batch, support_gt))
    for batch, gt in substitutor: 
        with torch.no_grad():
            result = model(batch)
        logits = result[ResultDict.LOGITS]
        metrics.update(logits.argmax(dim=1), gt)
    metric_values = linearize_metrics(metrics.compute(), id2class=id2class)
    tracker.log_metrics({f"final_{k}": v for k, v in metric_values.items()})
    
    for k, v in metric_values.items():
        logger.info(f"Training - {k}: {v}")
    

def merge_dicts(prompts, imgs):
    device = imgs[BatchKeys.IMAGES].device
    merge_prompts = deepcopy(prompts)
    out = {}
    for k in set(list(imgs.keys()) + list(merge_prompts.keys())):
        if k in imgs and prompts:
            dim = 0
            if k == BatchKeys.IMAGES:
                merge_prompts[k] = merge_prompts[k].unsqueeze(dim=0)
                dim = 1
            out[k] = torch.cat([imgs[k].cpu(), merge_prompts[k].cpu()], dim=dim).to(
                device
            )
            if k == BatchKeys.DIMS:
                out[k] = out[k].unsqueeze(dim=0).to(device)
        elif k in imgs:
            out[k] = imgs[k].to(device)
        else:
            out[k] = merge_prompts[k].unsqueeze(dim=0).to(device)
    return out


def refine_and_test(parameters):
    
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
                    )
            ]
        ).to(device)
        examples = dataloader.dataset.extract_prompts()
        examples = to_device(examples, device)
        
        with tracker.train():
            refine_model(model, examples, tracker, parameters["refinement"], metrics.clone(), id2class)

        tracker.log_test_prompts(examples, dataloader.dataset.id2class, dataset_name)

        bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            postfix={"loss": 0},
            desc="Test: ",
        )
        update_frequency = 10
        tracker.create_image_sequence(dataset_name)
        with torch.no_grad():
            for batch_idx, batch_dict in bar:
                image_dict, gt = batch_dict
                input_dict = to_device(merge_dicts(prompts=examples, imgs=image_dict), device)
                gt = to_device(gt, device)
                outputs = model(input_dict)[ResultDict.LOGITS]
                tracker.log_test_prediction(
                    batch_idx=batch_idx,
                    input_dict=image_dict,
                    gt=gt,
                    pred=outputs,
                    input_shape=image_size,
                    id2classes=dataloader.dataset.id2class,
                    dataset_name=dataset_name,
                )
                outputs = torch.argmax(outputs, dim=1)
                dims = image_dict[BatchKeys.DIMS][0].tolist()
                outputs = outputs[:, : dims[0], : dims[1]]
                metrics.update(outputs, gt)
                if batch_idx % update_frequency == 0:
                    metrics_values = linearize_metrics(metrics.compute(), id2class=id2class)
                    f1_score = metrics_values.get("MulticlassF1Score", 0)
                    bar.set_postfix({"F1 Score": f1_score})
            metrics_values = linearize_metrics(metrics.compute(), id2class=id2class)

            tracker.log_metrics(metrics=metrics_values)
            for k, v in metrics_values.items():
                logger.info(f"Test - {k}: {v}")
            tracker.add_image_sequence(dataset_name)
            
    tracker.end()
            

@cli.command("grid")
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a grid search",
)
def grid(parameters):
    grid_logger = get_logger("Grid")
    parameters = load_yaml(parameters)
    runs_parameters = make_grid(parameters)
    grid_logger.info(f"Running {len(runs_parameters)} runs")
    for i, run_parameters in enumerate(runs_parameters):
        grid_logger.info(f"Running run {i+1}/{len(runs_parameters)}")
        refine_and_test(run_parameters)


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