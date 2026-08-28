# refine.py

from datetime import datetime
import os
import uuid
import click
from einops import einops
import math
import random as _random
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from distillfss.substitution import get_substitutor, recusrive_clone
from distillfss.test import test
from distillfss.utils.logger import get_logger
from distillfss.utils.tracker import WandBTracker, wandb_experiment
from distillfss.utils.utils import ResultDict, linearize_metrics, load_yaml, to_device
from distillfss.utils.grid import ParallelRun, create_experiment, find_grid_to_resume, make_grid


OUT_FOLDER = "out"


@click.group()
def cli():
    """Run a refinement or a grid"""
    pass


def stitch_support(support_batch, support_gt):
    """Stitch support images into MULTI-CLASS composites: each output image is a
    grid with one image per class, so every training image contains ALL classes.
    This converts one-class-per-image into multi-label (the WeedMap regime that
    works) and forces the per-class heads to DISCRIMINATE classes within a single
    image -- directly attacking the winner-take-all collapse where 2/3 classes die.
    Geometry only (downscale + tile). Test images stay un-stitched."""
    imgs = support_batch[BatchKeys.IMAGES]      # [1, N, 3, H, W]
    gt = support_gt                             # [1, N, H, W]
    device = imgs.device
    B, N, Cch, H, W = imgs.shape
    Ccls = support_batch[BatchKeys.PROMPT_MASKS].shape[2]   # bg + fg
    n_fg = Ccls - 1

    # group support indices by the fg classes present in each
    by_class = {c: [] for c in range(1, n_fg + 1)}
    for i in range(N):
        for c in gt[0, i].unique().tolist():
            if int(c) >= 1:
                by_class[int(c)].append(i)
    avail = [c for c, idxs in by_class.items() if idxs]
    if len(avail) <= 1:
        return support_batch, support_gt

    cols = math.ceil(math.sqrt(len(avail)))
    rows = math.ceil(len(avail) / cols)
    ch, cw = H // rows, W // cols
    M = max(N, 4)

    out_imgs = torch.zeros(B, M, Cch, H, W, device=device)
    out_gt = torch.zeros(B, M, H, W, dtype=torch.long, device=device)
    for m in range(M):
        _random.shuffle(avail)
        for j, c in enumerate(avail):
            r, col = divmod(j, cols)
            src = _random.choice(by_class[c])
            cell_img = F.interpolate(imgs[0, src:src + 1], size=(ch, cw),
                                     mode="bilinear", align_corners=False)[0]
            cell_gt = F.interpolate(gt[0, src:src + 1].unsqueeze(1).float(),
                                    size=(ch, cw), mode="nearest")[0, 0].long()
            cell_gt = torch.where(cell_gt == c, cell_gt, torch.zeros_like(cell_gt))
            y0, x0 = r * ch, col * cw
            out_imgs[0, m, :, y0:y0 + ch, x0:x0 + cw] = cell_img
            out_gt[0, m, y0:y0 + ch, x0:x0 + cw] = cell_gt

    masks = F.one_hot(out_gt.reshape(B * M, H, W), Ccls).permute(0, 3, 1, 2).float()
    masks[:, 0] = 0
    masks = masks.reshape(B, M, Ccls, H, W)
    flags = torch.zeros(B, M, Ccls, dtype=torch.bool, device=device)
    flags[:, :, 0] = True
    for m in range(M):
        for c in out_gt[0, m].unique().tolist():
            if int(c) >= 1:
                flags[0, m, int(c)] = True

    new_batch = {k: v for k, v in support_batch.items()}
    new_batch.pop(BatchKeys.EMBEDDINGS, None)
    new_batch[BatchKeys.IMAGES] = out_imgs
    new_batch[BatchKeys.PROMPT_MASKS] = masks
    new_batch[BatchKeys.FLAG_EXAMPLES] = flags
    new_batch[BatchKeys.DIMS] = torch.tensor([[H, W]] * M, device=device).unsqueeze(0)
    return new_batch, out_gt


def augment_support(support_batch, support_gt):
    """Random geometry augmentation (flips + 90-deg rotations) applied
    CONSISTENTLY to support images, prompt masks and GT so pixel correspondence
    is preserved. Geometry-only -> label-preserving. This multiplies the
    effective few-shot data seen during refinement, which is the main lever
    against student overfitting at low shot. Embeddings are dropped so models
    re-extract features from the augmented images (DMTNet re-extracts per
    forward; do not enable for models that consume precomputed embeddings)."""
    import random

    batch = {k: recusrive_clone(v) for k, v in support_batch.items()}
    gt = support_gt.clone()
    batch.pop(BatchKeys.EMBEDDINGS, None)

    imgs = batch[BatchKeys.IMAGES]                 # [B, N, C, H, W]
    masks = batch.get(BatchKeys.PROMPT_MASKS)      # [B, N, Ccls, H, W]

    flips = []
    if random.random() < 0.5:
        flips.append(-1)
    if random.random() < 0.5:
        flips.append(-2)
    if flips:
        imgs = torch.flip(imgs, dims=flips)
        gt = torch.flip(gt, dims=flips)
        if masks is not None:
            masks = torch.flip(masks, dims=flips)

    # 90-deg rotations only for square images (keeps DIMS / postprocess valid)
    if imgs.shape[-1] == imgs.shape[-2]:
        k = random.randint(0, 3)
        if k:
            imgs = torch.rot90(imgs, k, dims=(-2, -1))
            gt = torch.rot90(gt, k, dims=(-2, -1))
            if masks is not None:
                masks = torch.rot90(masks, k, dims=(-2, -1))

    batch[BatchKeys.IMAGES] = imgs
    if masks is not None:
        batch[BatchKeys.PROMPT_MASKS] = masks
    return batch, gt


def validate_support(model, support_batch, support_gt, substitutor, metrics, id2class):
    metrics.reset()
    stud_metrics = metrics.clone()
    substitutor.reset(batch=(support_batch, support_gt))
    model.eval()
    for batch, gt in substitutor:
        with torch.no_grad():
            result = model(batch)
        logits = result[ResultDict.LOGITS]
        metrics.update(logits.argmax(dim=1), gt)
    return linearize_metrics(metrics.compute(), id2class=id2class)

def refine_model(
    model, support_set, tracker: WandBTracker, logger, params, metrics, id2class=None
):
    lr = params["lr"]
    max_iterations = params["max_iterations"]
    subsample = params.get("subsample")
    substitutor_name = params.get("substitutor")
    iterations_is_num_classes = params.get("iterations_is_num_classes", False)
    hot_parameters = params["hot_parameters"]
    skip_final_metrics = params.get("skip_final_metrics", False)
    validate_every = params.get("validate_every", None)
    weight_decay = params.get("weight_decay", 0.0)
    augment = params.get("augment", False)
    grad_clip = params.get("grad_clip", None)
    stitch = params.get("stitch", False)
    # fraction of episodes that are stitched (rest use real/augmented images) --
    # stitch breaks the class collapse but creates a train/test domain gap; mixing
    # keeps the model calibrated to the real (un-stitched) test domain.
    stitch_prob = params.get("stitch_prob", 1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = get_loss(params["loss"])

    # Some teachers take a training-only code path that needs data unavailable at
    # refinement time (PAHNet/SCCANPlus indexes `cat_idx`, the base-class ids from its
    # pretraining set, under `if self.training`). In distillation the teacher was always
    # .eval(), so this never triggered; TransferFSS unfreezes it. eval_mode keeps the
    # module in eval() — gradients still flow, only BN/dropout/those branches change.
    if params.get("eval_mode"):
        model.eval()
    else:
        model.train()
    # Support-set refinement runs with batch size 1. Models with a globally-pooled
    # branch then feed BatchNorm a [1, C, 1, 1] tensor, which raises
    # "Expected more than 1 value per channel when training" (hit by PAHNet in the
    # TransferFSS campaign, where the teacher is unfrozen instead of frozen).
    # freeze_bn keeps BN in eval mode (running stats, no batch stats) — standard for
    # few-shot fine-tuning. Opt-in so existing campaigns are unaffected.
    if params.get("freeze_bn"):
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
    if hot_parameters:
        for name, param in model.named_parameters():
            if any([hot_parameter in name for hot_parameter in hot_parameters]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Training {name}")

    support_batch, support_gt = get_support_batch(support_set)
    
    if "extract_features" in model.__class__.__dict__:
        images = einops.rearrange(support_batch[BatchKeys.IMAGES], "b s c h w -> (b s) c h w")
        with torch.no_grad():
            embeddings = model.extract_features(images)
        if isinstance(embeddings, list) or isinstance(embeddings, tuple):
            # Turn from [(B, C1, H1, W1), (B, C2, H2, W2), ...] to [[(1, C1, H1, W1)], [(1, C2, H2, W2)], ...], [[(1, C1, H1, W1)], [(1, C2, H2, W2)], ...], ... B times]
            embeddings = [[emb[i].unsqueeze(0) for emb in embeddings] for i in range(embeddings[0].shape[0])]
            support_batch[BatchKeys.EMBEDDINGS] = [embeddings]
        else:
            support_batch[BatchKeys.EMBEDDINGS] = einops.rearrange(embeddings, "(b s) c h w -> b s c h w", b=support_batch[BatchKeys.IMAGES].shape[0])

    substitutor = get_substitutor(
        substitutor_name, substitute=True, subsample=subsample, iterations_is_num_classes=iterations_is_num_classes
    )
    substitutor.reset(batch=(support_batch, support_gt))
    support_set_len = support_set[BatchKeys.IMAGES].shape[1]
    metric_update = 10

    bar = tqdm(range(max_iterations), desc="Training Progress")
    best_validation_score = -1
    best_validation_ckpt = None
    val_metrics = metrics.clone()
    stud_metrics = metrics.clone()

    sequence_name = "predictions"
    tracker.create_image_sequence(sequence_name)
    for step in bar:
        loss_total = 0
        if stitch and _random.random() < stitch_prob:
            substitutor.reset(batch=stitch_support(support_batch, support_gt))
        elif augment:
            substitutor.reset(batch=augment_support(support_batch, support_gt))
        else:
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
            if ResultDict.DISTILLED_LOGITS in result:
                stud_metrics.update(result[ResultDict.DISTILLED_LOGITS].argmax(dim=1), gt)
            tracker.log_batch(
                batch,
                gt,
                outputs,
                step,
                substep,
                id2class,
                phase="train",
                sequence_name=sequence_name,
            )

        if grad_clip:
            # Caps the occasional huge gradient step (early loss can spike to
            # ~thousands with augmentation) that otherwise nukes the student's
            # foreground pathway into an all-background collapse (test -> 0.0).
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if step % metric_update == 0:
            metric_values = linearize_metrics(metrics.compute(), id2class=id2class)
            stud_metrics_values = linearize_metrics(stud_metrics.compute(), id2class=id2class) if ResultDict.DISTILLED_LOGITS in result else {}
            jaccard = metric_values.get("MulticlassJaccardIndex_fg", 0)
            jaccard_stud = stud_metrics_values.get("MulticlassJaccardIndex_fg", None)
            current_lr = optimizer.param_groups[0]["lr"]
            tracker.log_metrics(metric_values if jaccard_stud is None else stud_metrics_values)
            tracker.log_metric("step", step // metric_update)
            logger.info(f"Step {step}: Loss {loss_total}, Jaccard {jaccard}, Learning Rate {current_lr}" + (f", DJaccard {jaccard_stud}" if jaccard_stud is not None else ""))
        if validate_every and step % validate_every == 0:
            with tracker.validate():
                metric_values = validate_support(model, support_batch, support_gt, substitutor, val_metrics.clone(), id2class)
                tracker.log_metrics(metric_values)
                tracker.log_metric("step", step // validate_every)
            if metric_values.get("MulticlassJaccardIndex_fg", 0) > best_validation_score:
                logger.info(f"New best validation Jaccard {metric_values.get('MulticlassJaccardIndex_fg', 0)} at step {step}")
                best_validation_score = metric_values.get("MulticlassJaccardIndex_fg", 0)
                best_validation_ckpt = model.state_dict()
            else:
                logger.info(f"Validation Jaccard {metric_values.get('MulticlassJaccardIndex_fg', 0)} at step {step}")
            model.eval() if params.get("eval_mode") else model.train()
            if params.get("freeze_bn"):  # re-apply: .train() re-enables BN batch stats
                for m in model.modules():
                    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                        m.eval()

        tracker.log_metric("loss", loss_total)
        postfix = {"Loss": loss_total, "Jaccard": jaccard, "Learning Rate": current_lr}
        if jaccard_stud is not None:
            postfix["DJaccard"] = jaccard_stud
        bar.set_postfix(postfix)
    tracker.add_image_sequence(sequence_name)

    # Get the training scores
    substitutor = get_substitutor(substitutor_name, substitute=True)
    support_batch, support_gt = get_support_batch(support_set)
    support_set_len = support_batch[BatchKeys.IMAGES].shape[1]
    metrics.reset()
    
    if best_validation_ckpt is not None:
        model.load_state_dict(best_validation_ckpt)
        logger.info(f"Loaded best validation checkpoint with Jaccard {best_validation_score}")

    if not skip_final_metrics:
        logger.info("Finished Training, extracting metrics...")
        metric_values = validate_support(model, support_batch, support_gt, substitutor, metrics, id2class)
        tracker.log_metrics({f"final_{k}": v for k, v in metric_values.items()})

        for k, v in metric_values.items():
            logger.info(f"Training - {k}: {v}")


def refine_and_test(
    parameters, run_name=None, log_params=True, log_model=True, log_on_file=True
):
    if run_name is None:
        run_name = str(uuid.uuid4())[:8] + ".log"
        run_name = os.path.join(OUT_FOLDER, run_name)
        os.makedirs(OUT_FOLDER, exist_ok=True)
    # model filename is log filename but with .pt instead of .log
    model_filename = run_name + ".pt"
    params_filename = run_name + ".yaml"
    if log_params:
        with open(params_filename, "w") as f:
            yaml.dump(parameters, f)

    log_filename = run_name + ".log" if log_on_file else None
    logger = get_logger("Refine", log_filename)
    logger.info("parameters:")
    logger.info(parameters)

    device = parameters.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {device}")

    # Seed all RNGs (student init, augmentation, substitutor randperm) from the
    # per-dataset `seed`. This makes a multi-seed sweep vary BOTH the support set
    # (dataset shuffle) AND the training dynamics coherently. seed=None -> unseeded
    # (original behaviour). Read from the first test_ dataset's config block.
    _seed = next((p.get("seed") for p in parameters["dataset"].get("datasets", {}).values()
                  if isinstance(p, dict) and p.get("seed") is not None), None)
    if _seed is not None:
        import random as _random
        torch.manual_seed(_seed)
        _random.seed(_seed)
        try:
            import numpy as _np
            _np.random.seed(_seed)
        except Exception:
            pass
        logger.info(f"Seeded all RNGs with seed={_seed}")

    test_loaders = get_testloaders(parameters["dataset"], parameters["dataloader"])
    image_size = parameters["dataset"]["preprocess"]["image_size"]

    model = build_model(parameters["model"])
    model.to(device)
    model.eval()

    tracker = wandb_experiment(parameters, logger=logger)

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
                ),
            ]
        ).to(device)
        examples = dataloader.dataset.extract_prompts()
        examples = to_device(examples, device)
        prompt_to_use = parameters["test"].get("prompt_to_use", None)
        if prompt_to_use is not None:
            examples = {k: v[:prompt_to_use] for k, v in examples.items()}

        if "refinement" in parameters:
            with tracker.train():
                refine_model(
                    model,
                    examples,
                    tracker,
                    logger,
                    parameters["refinement"],
                    metrics.clone(),
                    id2class,
                )
            if log_model and parameters.get("log_model", True):
                try:
                    torch.save(model.state_dict(), model_filename)
                except Exception as _e:
                    logger.warning(f"model save failed ({_e}); continuing to test")
                
        if parameters.get("push_to_hub", None):
            repo_name = parameters["push_to_hub"]["repo_name"]
            model.push_to_hub(
                repo_name,
                parameters=parameters
            )

        test(
            model,
            dataloader,
            examples,
            tracker,
            logger,
            dataset_name,
            image_size,
            metrics,
            device,
        )

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
@click.option(
    "--resume",
    default=False,
    is_flag=True,
    help="Resume the most recent grid whose hyperparams match the provided parameters",
)
@click.option(
    "--scheduler",
    default="slurm",
    type=click.Choice(["slurm", "condor"]),
    help="Cluster scheduler for --parallel: 'slurm' (Leonardo) or 'condor' (ReCaS)",
)
def grid(parameters, parallel, only_create=False, resume=False, scheduler="slurm"):
    parameters = load_yaml(parameters)
    grid_name = parameters.pop("grid")

    runs_parameters = create_experiment(parameters)

    if resume:
        log_folder, resume_from = find_grid_to_resume(parameters, grid_name, OUT_FOLDER)
        if log_folder is None:
            raise click.ClickException(
                f"No matching grid found for '{grid_name}' to resume"
            )
        grid_logger = get_logger("Grid", f"{log_folder}/grid.log")
        if resume_from >= len(runs_parameters):
            grid_logger.info(
                f"Grid '{grid_name}' is already complete "
                f"({len(runs_parameters)}/{len(runs_parameters)} runs done)"
            )
            return
        grid_logger.info(
            f"Resuming grid '{grid_name}' from run {resume_from + 1}/{len(runs_parameters)}"
        )
    else:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_folder = os.path.join(OUT_FOLDER, f"{current_time}_{grid_name}")
        os.makedirs(log_folder, exist_ok=True)
        with open(os.path.join(log_folder, "hyperparams.yaml"), "w") as f:
            yaml.dump(parameters, f)
        grid_logger = get_logger("Grid", f"{log_folder}/grid.log")
        grid_logger.info(f"Running {len(runs_parameters)} runs")
        resume_from = 0

    for i, run_parameters in enumerate(runs_parameters):
        if i < resume_from:
            continue
        run_name = f"{log_folder}/run_{i}"
        if resume and i == resume_from:
            log_file = f"{run_name}.log"
            if os.path.exists(log_file):
                restart_ts = datetime.now().strftime("[%m-%d %H:%M:%S]")
                with open(log_file, "a") as f:
                    f.write(f"INFO {restart_ts} [Grid] *** Run {i} RESTARTED ***\n")
        if parallel:
            run = ParallelRun(
                run_parameters,
                multi_gpu=False,
                logger=grid_logger,
                run_name=run_name,
                scheduler=scheduler,
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
            refine_and_test(run_parameters, run_name=run_name)


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
    refine_and_test(
        parameters,
        run_name,
        not disable_log_params,
        not disable_log_model,
        not disable_log_on_file,
    )


if __name__ == "__main__":
    cli()
