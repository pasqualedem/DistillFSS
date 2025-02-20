from contextlib import contextmanager
from copy import deepcopy
import time
import os
from typing import Optional, Union, Any

import pandas as pd
import numpy as np

import torch.nn.functional as F

import torch
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize

from fssweed.data.utils import BatchKeys
from fssweed.utils.logger import get_logger
from fssweed.logger.utils import get_tmp_dir
from fssweed.utils.utils import log_every_n, write_yaml


def unnormalize(image_tensor):
    MEAN = np.array([123.675, 116.280, 103.530]) / 255
    STD = np.array([58.395, 57.120, 57.375]) / 255
    unnormalized_image = (
        image_tensor.cpu().numpy() * np.array(STD)[:, None, None]
    ) + np.array(MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    return Image.fromarray(unnormalized_image)


logger = get_logger(__name__)

WANDB_ID_PREFIX = "wandb_id."
WANDB_INCLUDE_FILE_NAME = ".wandbinclude"


class WandBTracker:
    MAX_CLASSES = 100000  # For negative classes

    def __init__(
        self,
        project: str,
        resume: bool = False,
        offline_directory: str = None,
        save_checkpoints_remote: bool = True,
        save_tensorboard_remote: bool = True,
        save_logs_remote: bool = True,
        entity: Optional[str] = None,
        api_server: Optional[str] = None,
        save_code: bool = False,
        tags=None,
        run_id=None,
        resume_checkpoint_type: str = "best",
        group=None,
        tmp_dir=None,
        log_frequency: int = 100,
        train_image_log_frequency: int = 1000,
        val_image_log_frequency: int = 1000,
        test_image_log_frequency: int = 1000,
        experiment_save_delta: int = None,
        **kwargs,
    ):
        """

        :param experiment_name: Used for logging and loading purposes
        :param s3_path: If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally
        :param checkpoint_loaded: if true, then old tensorboard files will *not* be deleted when tb_files_user_prompt=True
        :param max_epochs: the number of epochs planned for this training
        :param tb_files_user_prompt: Asks user for Tensorboard deletion prompt.
        :param launch_tensorboard: Whether to launch a TensorBoard process.
        :param tensorboard_port: Specific port number for the tensorboard to use when launched (when set to None, some free port
                    number will be used
        :param save_checkpoints_remote: Saves checkpoints in s3.
        :param save_tensorboard_remote: Saves tensorboard in s3.
        :param save_logs_remote: Saves log files in s3.
        :param save_code: save current code to wandb
        """
        tracker_resume = "must" if resume else None
        self.resume = tracker_resume
        resume = run_id is not None
        if not tracker_resume and resume:
            if tags is None:
                tags = []
            tags = tags + ["resume", run_id]
        self.accelerator_state_dir = None
        if offline_directory:
            os.makedirs(offline_directory, exist_ok=True)
            os.environ["WANDB_ARTIFACT_LOCATION"] = offline_directory
            os.environ["WANDB_ARTIFACT_DIR"] = offline_directory
            os.environ["WANDB_CACHE_DIR"] = offline_directory
            os.environ["WANDB_CONFIG_DIR"] = offline_directory
            os.environ["WANDB_DATA_DIR"] = offline_directory
        if resume:
            self._resume(offline_directory, run_id, checkpoint_type=resume_checkpoint_type)
        experiment = None
        experiment = wandb.init(
            project=project,
            entity=entity,
            resume=tracker_resume,
            id=run_id if tracker_resume else None,
            tags=tags,
            dir=offline_directory,
            group=group,
        )
        logger.info(f"wandb run id  : {experiment.id}")
        logger.info(f"wandb run name: {experiment.name}")
        logger.info(f"wandb run dir : {experiment.dir}")
        wandb.define_metric("train/step")
        # set all other train/ metrics to use this step
        wandb.define_metric("train/*", step_metric="train/step")

        wandb.define_metric("validate/step")
        # set all other validate/ metrics to use this step
        wandb.define_metric("validate/*", step_metric="validate/step")
            
        self.experiment = experiment
        self.tmp_dir = tmp_dir
        self.local_dir = experiment.dir if hasattr(experiment, "dir") else None
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.log_frequency = log_frequency
        self.prefix_frequency_dict = {
            "train": train_image_log_frequency,
            "val": val_image_log_frequency,
            "test": test_image_log_frequency,
        }
        self.start_time = time.time()
        self.experiment_save_delta = experiment_save_delta
        if save_code:
            self._save_code()

        self.save_checkpoints_wandb = save_checkpoints_remote
        self.save_tensorboard_wandb = save_tensorboard_remote
        self.save_logs_wandb = save_logs_remote
        self.context = ""
        self.sequences = {}
                
    def _resume(self, offline_directory, run_id, checkpoint_type="latest"):
        if not offline_directory:
            offline_directory = "."
        wandb_dir = os.path.join(offline_directory, "wandb")
        runs = os.listdir(wandb_dir)
        runs = sorted(list(filter(lambda x: run_id in x, runs)))
        if len(runs) == 0:
            raise ValueError(f"Run {run_id} not found in {wandb_dir}")
        if len(runs) > 1:
            logger.warning(f"Multiple runs found for {run_id} in {wandb_dir}")
            for run in runs:
                logger.warning(run)
            logger.warning(f"Using {runs[0]}")
        run = runs[0]
        self.accelerator_state_dir = os.path.join(wandb_dir, run, "files", checkpoint_type)
        logger.info(f"Resuming from {self.accelerator_state_dir}")
        
    def _save_code(self):
        """
        Save the current code to wandb.
        If a file named .wandbinclude is avilable in the root dir of the project the settings will be taken from the file.
        Otherwise, all python file in the current working dir (recursively) will be saved.
        File structure: a single relative path or a single type in each line.
        i.e:

        src
        tests
        examples
        *.py
        *.yaml

        The paths and types in the file are the paths and types to be included in code upload to wandb
        """
        base_path, paths, types = self._get_include_paths()

        if len(types) > 0:

            def func(path):
                for p in paths:
                    if path.startswith(p):
                        for t in types:
                            if path.endswith(t):
                                return True
                return False

            include_fn = func
        else:
            include_fn = lambda path: path.endswith(".py")

        if base_path != ".":
            wandb.run.log_code(base_path, include_fn=include_fn)
        else:
            wandb.run.log_code(".", include_fn=include_fn)

    def log_parameters(self, config: dict = None):
        wandb.config.update(config, allow_val_change=self.resume)
        tmp = os.path.join(self.local_dir, "config.yaml")
        write_yaml(config, tmp)
        self.add_file("config.yaml")
        
    def add_tags(self, tags):
        wandb.run.tags = wandb.run.tags + tuple(tags)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0):
        wandb.log(data={tag: scalar_value}, step=global_step)


    def add_scalars(self, tag_scalar_dict: dict, global_step: int = 0):
        for name, value in tag_scalar_dict.items():
            if isinstance(value, dict):
                tag_scalar_dict[name] = value["value"]
        wandb.log(data=tag_scalar_dict, step=global_step)

    def add_image(
        self,
        tag: str,
        image: Union[torch.Tensor, np.array, Image.Image],
        data_format="CHW",
        global_step: int = 0,
    ):
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        if image.shape[0] < 5:
            image = image.transpose([1, 2, 0])
        wandb.log(data={tag: wandb.Image(image, caption=tag)}, step=global_step)

    def add_images(
        self,
        tag: str,
        images: Union[torch.Tensor, np.array],
        data_format="NCHW",
        global_step: int = 0,
    ):
        wandb_images = []
        for im in images:
            if isinstance(im, torch.Tensor):
                im = im.cpu().detach().numpy()

            if im.shape[0] < 5:
                im = im.transpose([1, 2, 0])
            wandb_images.append(wandb.Image(im))
        wandb.log({tag: wandb_images}, step=global_step)

    def add_video(
        self, tag: str, video: Union[torch.Tensor, np.array], global_step: int = 0
    ):
        if video.ndim > 4:
            for index, vid in enumerate(video):
                self.add_video(tag=f"{tag}_{index}", video=vid, global_step=global_step)
        else:
            if isinstance(video, torch.Tensor):
                video = video.cpu().detach().numpy()
            wandb.log({tag: wandb.Video(video, fps=4)}, step=global_step)

    def add_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.array],
        bins: str,
        global_step: int = 0,
    ):
        wandb.log({tag: wandb.Histogram(values, num_bins=bins)}, step=global_step)

    def add_plot(self, tag: str, values: pd.DataFrame, xtitle, ytitle, classes_marker):
        table = wandb.Table(columns=[classes_marker, xtitle, ytitle], dataframe=values)
        plt = wandb.plot_table(
            tag,
            table,
            {"x": xtitle, "y": ytitle, "class": classes_marker},
            {
                "title": tag,
                "x-axis-title": xtitle,
                "y-axis-title": ytitle,
            },
        )
        wandb.log({tag: plt})

    def add_text(self, tag: str, text_string: str, global_step: int = 0):
        wandb.log({tag: text_string}, step=global_step)

    def add_figure(self, tag: str, figure: plt.figure, global_step: int = 0):
        wandb.log({tag: figure}, step=global_step)

    def add_mask(self, tag: str, image, mask_dict, global_step: int = 0):
        wandb.log({tag: wandb.Image(image, masks=mask_dict)}, step=global_step)

    def add_table(self, tag, data, columns, rows):
        if isinstance(data, torch.Tensor):
            data = [[x.item() for x in row] for row in data]
        table = wandb.Table(data=data, rows=rows, columns=columns)
        wandb.log({tag: table})

    def end(self):
        wandb.finish()

    def add_file(self, file_name: str = None):
        wandb.save(
            glob_str=os.path.join(self.local_dir, file_name),
            base_path=self.local_dir,
            policy="now",
        )

    def add_summary(self, metrics: dict):
        wandb.summary.update(metrics)

    def upload(self):
        if self.save_tensorboard_wandb:
            wandb.save(
                glob_str=self._get_tensorboard_file_name(),
                base_path=self.local_dir,
                policy="now",
            )

        if self.save_logs_wandb:
            wandb.save(
                glob_str=self.log_file_path, base_path=self.local_dir, policy="now"
            )

    def add_checkpoint(self, tag: str, state_dict: dict, global_step: int = 0):
        name = f"ckpt_{global_step}.pth" if tag is None else tag
        if not name.endswith(".pth"):
            name += ".pth"

        path = os.path.join(self.local_dir, name)
        torch.save(state_dict, path)

        if self.save_checkpoints_wandb:
            if self.s3_location_available:
                self.model_checkpoints_data_interface.save_remote_checkpoints_file(
                    self.experiment_name, self.local_dir, name
                )
            wandb.save(glob_str=path, base_path=self.local_dir, policy="now")

    def _get_tensorboard_file_name(self):
        try:
            tb_file_path = self.tensorboard_writer.file_writer.event_writer._file_name
        except RuntimeError as e:
            logger.warning("tensorboard file could not be located for ")
            return None

        return tb_file_path

    def _get_wandb_id(self):
        for file in os.listdir(self.local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                return file.replace(WANDB_ID_PREFIX, "")

    def _set_wandb_id(self, id):
        for file in os.listdir(self.local_dir):
            if file.startswith(WANDB_ID_PREFIX):
                os.remove(os.path.join(self.local_dir, file))

    def _get_include_paths(self):
        """
        Look for .wandbinclude file in parent dirs and return the list of paths defined in the file.

        file structure is a single relative (i.e. src/) or a single type (i.e *.py)in each line.
        the paths and types in the file are the paths and types to be included in code upload to wandb
        :return: if file exists, return the list of paths and a list of types defined in the file
        """

        wandb_include_file_path = self._search_upwards_for_file(WANDB_INCLUDE_FILE_NAME)
        if wandb_include_file_path is not None:
            with open(wandb_include_file_path) as file:
                lines = file.readlines()

            base_path = os.path.dirname(wandb_include_file_path)
            paths = []
            types = []
            for line in lines:
                line = line.strip().strip("/n")
                if line == "" or line.startswith("#"):
                    continue

                if line.startswith("*."):
                    types.append(line.replace("*", ""))
                else:
                    paths.append(os.path.join(base_path, line))
            return base_path, paths, types

        return ".", [], []

    @staticmethod
    def _search_upwards_for_file(file_name: str):
        """
        Search in the current directory and all directories above it for a file of a particular name.
        :param file_name: file name to look for.
        :return: pathlib.Path, the location of the first file found or None, if none was found
        """

        try:
            cur_dir = os.getcwd()
            while cur_dir != "/":
                if file_name in os.listdir(cur_dir):
                    return os.path.join(cur_dir, file_name)
                else:
                    cur_dir = os.path.dirname(cur_dir)
        except RuntimeError as e:
            return None

        return None

    def log_batch(
        self,
        input_dict,
        gt,
        pred,
        step,
        substitution_step,
        id2class,
        phase,
        sequence_name
    ):
        if log_every_n(step, self.prefix_frequency_dict[phase]):
            self.log_prompts(   
                input_dict=input_dict,
                step=step,
                substitution_step=substitution_step,
                id2class=id2class,
                sequence_name=sequence_name,
            )
            self.log_gt_pred(
                input_dict,
                step,
                substitution_step,
                gt,
                pred,
                id2class,
                sequence_name,
            )
            
    def log_test_prediction(
        self,
        batch_idx,
        input_dict,
        gt,
        pred,
        input_shape,
        id2classes,
        dataset_name,
    ):
        if not log_every_n(batch_idx, self.prefix_frequency_dict["test"]):
            return
        dims = input_dict["dims"]
        images = input_dict["images"][:, 0]

        for b in range(gt.shape[0]):
            image = resize(images[b], dims[b])
            image = unnormalize(image)

            sample_gt = gt[b, : dims[b, 0], : dims[b, 1]].detach().cpu().numpy()

            sample_pred = pred[b, :, : dims[b, 0], : dims[b, 1]]
            sample_pred = torch.argmax(sample_pred, dim=0).detach().cpu().numpy()

            wandb_image = wandb.Image(
                image,
                masks={
                    "ground_truth": {
                        "mask_data": sample_gt,
                        "class_labels": id2classes,
                    },
                    "prediction": {
                        "mask_data": sample_pred,
                        "class_labels": id2classes,
                    },
                },
                classes=[
                    {"id": c, "name": name} for c, name in id2classes.items()
                ],
            )

            self.add_image_to_sequence(
                dataset_name,
                f"image_{batch_idx}_sample_{b}",
                wandb_image,
            )

    def log_gt_pred(
        self,
        input_dict,
        step,
        substitution_step,
        gt,
        pred,
        id2class,
        sequence_name,
    ):
        dims = input_dict["dims"]
        images = input_dict[BatchKeys.IMAGES][:, 0] # Query images

        for b in range(gt.shape[0]):
            image = resize(images[b], dims[b, 0])
            image = unnormalize(image)

            sample_gt = gt[b].detach().cpu().numpy()
            sample_pred = pred[b].detach().cpu().numpy()

            wandb_image = wandb.Image(
                image,
                masks={
                    "ground_truth": {
                        "mask_data": sample_gt,
                        "class_labels": id2class,
                    },
                    "prediction": {
                        "mask_data": sample_pred,
                        "class_labels": id2class,
                    },
                },
                classes=[
                    {"id": c, "name": name} for c, name in id2class.items()
                ],
            )

            self.add_image_to_sequence(
                sequence_name,
                f"step_{step}_sample_{b}_substep_{substitution_step}_gt_pred",
                wandb_image,
            )
            
    def log_test_prompts(
        self,
        input_dict,
        id2classes,
        dataset_name,
    ):
        sequence_name = f"{dataset_name}_prompts"
        self.create_image_sequence(sequence_name)
        all_masks = (
            input_dict["prompt_masks"].argmax(dim=1)
            if input_dict["prompt_masks"] is not None
            else None
        )
        flags_masks = input_dict["flag_examples"]
        images = input_dict["images"]

        for j in range(all_masks.shape[0]):
            image = unnormalize(images[j])
            # log masks

            masks = None
            if flags_masks[j].sum() > 0:
                cur_mask = all_masks[j].unsqueeze(0).unsqueeze(0).float()
                masks = {
                    "ground_truth": {
                        "mask_data": F.interpolate(
                            cur_mask,
                            images[j].shape[-2:],
                        )
                        .squeeze()
                        .cpu()
                        .numpy(),
                        "class_labels": id2classes,
                    }
                }

            wandb_image = wandb.Image(
                image,
                masks=masks,
                classes=[
                    {"id": c, "name": name}
                    for c, name in id2classes.items()
                ],
            )

            self.add_image_to_sequence(
                sequence_name,
                f"image_{j}_prompts",
                wandb_image,
            )
        self.add_image_sequence(sequence_name)

    def log_prompts(
        self,
        input_dict,
        step,
        substitution_step,
        id2class,
        sequence_name,
    ):
        images = input_dict[BatchKeys.IMAGES][:, 1:] # Support images
        dims = input_dict["dims"]
        all_masks = (
            input_dict["prompt_masks"].argmax(dim=2)
            if input_dict["prompt_masks"] is not None
            else None
        )
        flags_masks = input_dict["flag_examples"]
        for i in range(len(images)):
            sample_images = images[i]
            for j in range(all_masks.shape[1]):
                image = resize(sample_images[j], dims[i, j+1])
                image = unnormalize(image)
                masks = None
                if flags_masks[i, j].sum() > 0:
                    cur_mask = all_masks[i, j].unsqueeze(0).unsqueeze(0).float()
                    masks = {
                        "ground_truth": {
                            "mask_data": F.interpolate(
                                cur_mask,
                                sample_images[j].shape[-2:],
                            )
                            .squeeze()
                            .cpu()
                            .numpy(),
                            "class_labels": id2class,
                        }
                    }

                wandb_image = wandb.Image(
                    image,
                    masks=masks,
                    classes=[
                        {"id": c, "name": name}
                        for c, name in id2class.items()
                    ],
                )

                self.add_image_to_sequence(
                    sequence_name,
                    f"step_{step}_sample_{i}_substep_{substitution_step}_prompts",
                    wandb_image,
                )

    def create_image_sequence(self, name, columns=[]):
        self.sequences[name] = wandb.Table(["ID", "Image"] + columns)

    def add_image_to_sequence(
        self, sequence_name, name, wandb_image: wandb.Image, metadata=[]
    ):
        self.sequences[sequence_name].add_data(name, wandb_image, *metadata)

    def add_image_sequence(self, name):
        wandb.log({f"{self.context}_{name}": self.sequences[name]})
        del self.sequences[name]

    def log_asset_folder(self, folder, base_path=None, step=None):
        files = os.listdir(folder)
        for file in files:
            wandb.save(os.path.join(folder, file), base_path=base_path)

    def log_metric(self, name, metric, epoch=None):
        wandb.log({f"{self.context}/{name}": metric})

    def log_metrics(self, metrics: dict, epoch=None):
        wandb.log({f"{self.context}/{k}": v for k, v in metrics.items()})

    def __repr__(self):
        return "WandbLogger"

    @contextmanager
    def set_context(self, new_context):
        # Save the old context and set the new one
        old_context = self.context
        self.context = new_context

        yield self

        # Restore the old context
        self.context = old_context

    # You can still keep the specific context managers if needed
    @contextmanager
    def train(self):
        with self.set_context("train"):
            yield self

    @contextmanager
    def validate(self):
        with self.set_context("validate"):
            yield self

    @contextmanager
    def test(self):
        with self.set_context("test"):
            yield self

    @contextmanager
    def test(self):
        # Save the old context and set the new one
        old_context = self.context
        self.context = "test"

        yield self

        # Restore the old one
        self.context = old_context


def wandb_experiment(params: dict) -> WandBTracker:
    tracker_params = deepcopy(params.get("tracker", {}))
    global logger
    tmp_dir = get_tmp_dir()
    if tmp_dir:
        logger.info(
            f"Using {tmp_dir} as temporary directory from environment variables"
        )
    else:
        tmp_dir = tracker_params.get("tmp_dir", None)
        logger.info(
            f"No temporary directory found in environment variables, using {tmp_dir} for images"
        )
    os.makedirs(tmp_dir, exist_ok=True)
    tracker_params["tmp_dir"] = tmp_dir

    wandb_logger = WandBTracker(**tracker_params)
    wandb_logger.log_parameters(params)
    wandb_logger.add_tags(tracker_params.get("tags", ()))

    return wandb_logger