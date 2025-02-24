from enum import Enum
import os
import importlib
from datetime import datetime
from inspect import signature
import time
import torch
import yaml
from io import StringIO
import collections.abc
from typing import Mapping
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

# from label_anything.models.lam import Lam 


FLOAT_PRECISIONS = {
    "fp32": torch.float32,
    "fp64": torch.float64,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def strip_wandb_keys_recursive(data):
    
    if isinstance(data, dict):
        d = {}
        for k, v in data.items():
            if k in ["_wandb", "value"]:
                d = {**d, **strip_wandb_keys_recursive(v)}
            elif k in ["wandb_version", "desc"]:
                continue
            else:
                d[k] = strip_wandb_keys_recursive(v)
        return d
    elif isinstance(data, list):
        return [strip_wandb_keys_recursive(v) for v in data]
    else:
        return data
    

def strip_wandb_keys(data):
    return strip_wandb_keys_recursive(data) if "_wandb" in data else data


def load_yaml(file_path):
    try:
        with open(file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file.read())
            data = strip_wandb_keys(data)
            return data
    except FileNotFoundError as e:
        print(f"File '{file_path}' not found.")
        raise e
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise e
        

def write_yaml(data: dict, file_path: str = None, file=None):
    """ Write a dictionary to a YAML file.

    Args:
        data (dict): the data to write
        file_path (str): the path to the file
        file: the file object to write to (esclusive with file_path)
    """
    if file is not None:
        file.write(yaml.dump(data))
        return
    if file_path is None:
        raise ValueError("file_path or file must be specified")
    try:
        with open(file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        

def torch_dict_load(file_path):
    if file_path.endswith(".pth") or file_path.endswith(".pt") or file_path.endswith(".bin"):
        return torch.load(file_path)
    if file_path.endswith(".safetensors"):
        with safe_open(file_path, framework="pt") as f:
            d = {k: f.get_tensor(k) for k in f.keys()}
        return d
    raise ValueError("File extension not supported")
        
def torch_dict_save(data, file_path):
    if file_path.endswith(".pth") or file_path.endswith(".pt") or file_path.endswith(".bin"):
        torch.save(data, file_path)
    elif file_path.endswith(".safetensors"):
        save_file(data, file_path)
    else:
        raise ValueError("File extension not supported")
    

def load_state_dict(model, state_dict, strict=True):
    """
    """
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        try:
            print("Error loading state_dict, trying to load without 'model.' prefix")
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            print("Error loading state_dict, trying to load without 'module.' prefix")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=strict)
    print("State_dict loaded successfully")
    return model


# def unwrap_model_from_parallel(model, return_was_wrapped=False):
#     """
#     Unwrap a model from a DataParallel or DistributedDataParallel wrapper
#     :param model: the model
#     :return: the unwrapped model
#     """
#     if isinstance(
#         model,
#         (
#             torch.nn.DataParallel,
#             torch.nn.parallel.DistributedDataParallel,
#             Lam,
#         ),
#     ):
#         if return_was_wrapped:
#             return model.module, True
#         return model.module
#     else:
#         if return_was_wrapped:
#             return model, False
#         return model


def get_module_class_from_path(path):
    path = os.path.normpath(path)
    splitted = path.split(os.sep)
    module = ".".join(splitted[:-1])
    cls = splitted[-1]
    return module, cls


def update_collection(collec, value, key=None):
    if isinstance(collec, dict):
        if isinstance(value, dict):
            for keyv, valuev in value.items():
                collec = update_collection(collec, valuev, keyv)
        elif key is not None:
            if value is not None:
                collec[key] = value
        else:
            collec = {**collec, **value} if value is not None else collec
    else:
        collec = value if value is not None else collec
    return collec


def nested_dict_update(d, u):
    if u is not None:
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = nested_dict_update(d.get(k) or {}, v)
            else:
                d[k] = v
    return d


def instantiate_class(name, params):
    module, cls = get_module_class_from_path(name)
    imp_module = importlib.import_module(module)
    imp_cls = getattr(imp_module, cls)
    if (
        len(signature(imp_cls).parameters.keys()) == 1
        and "params" in list(signature(imp_cls).parameters.keys())[0]
    ):
        return imp_cls(params)
    return imp_cls(**params)


def substitute_values(x: torch.Tensor, values, unique=None):
    """
    Substitute values in a tensor with the given values
    :param x: the tensor
    :param unique: the unique values to substitute
    :param values: the values to substitute with
    :return: the tensor with the values substituted
    """
    if unique is None:
        unique = x.unique()
    lt = torch.full((unique.max() + 1,), -1, dtype=values.dtype, device=x.device)
    lt[unique] = values
    return lt[x]


# def load_yaml(path, return_string=False):
#     if hasattr(path, "readlines"):
#         d = convert_commentedmap_to_dict(YAML().load(path))
#         if return_string:
#             path.seek(0)
#             return d, path.read().decode("utf-8")
#     with open(path, "r") as param_stream:
#         d = convert_commentedmap_to_dict(YAML().load(param_stream))
#         if return_string:
#             param_stream.seek(0)
#             return d, str(param_stream.read())
#     return d


def log_every_n(image_idx: int, n: int):
    return False if n is None else image_idx % n == 0


def dict_to_yaml_string(mapping: Mapping) -> str:
    """
    Convert a nested dictionary or list to a string
    """
    string_stream = StringIO()
    yaml.dump(mapping, string_stream)
    output_str = string_stream.getvalue()
    string_stream.close()
    return output_str


def get_checkpoints_dir_path(
    project_name: str, group_name: str, ckpt_root_dir: str = None
):
    """Creating the checkpoint directory of a given experiment.
    :param experiment_name:     Name of the experiment.
    :param ckpt_root_dir:       Local root directory path where all experiment logging directories will
                                reside. When none is give, it is assumed that pkg_resources.resource_filename('checkpoints', "")
                                exists and will be used.
    :return:                    checkpoints_dir_path
    """
    if ckpt_root_dir:
        return os.path.join(ckpt_root_dir, project_name, group_name)


def get_timestamp():
    # Get the current timestamp
    timestamp = time.time()  # replace this with your timestamp or use time.time() for current time

    # Convert timestamp to a datetime object
    dt_object = datetime.fromtimestamp(timestamp)

    # Format the datetime object as a folder-friendly string
    return dt_object.strftime("%Y%m%d_%H%M%S")

def find_divisor_pairs(number):
    return [
        (i, number // i)
        for i in range(1, int(number**0.5) + 1)
        if number % i == 0
    ]


def get_divisors(n):
    """
    Returns a list of divisors of a given number.

    Args:
        n (int): The number to find divisors for.

    Returns:
        list: A list of divisors of the given number.
    """
    return [i for i in range(1, n + 1) if n % i == 0]


class RunningAverage:
    def __init__(self):
        self.accumulator = 0
        self.steps = 0
        
    def update(self, value):
        self.accumulator += value
        self.steps += 1
        
    def compute(self):
        return self.accumulator / self.steps


class StrEnum(str, Enum):
    pass


class ResultDict(StrEnum):
    CLASS_EMBS = "class_embeddings"
    LOGITS = "logits"
    EXAMPLES_CLASS_EMBS = "class_examples_embeddings"
    EXAMPLES_CLASS_SRC = "class_examples_src"
    LOSS = "loss"
    LAST_HIDDEN_STATE = 'last_hidden_state'
    LAST_BLOCK_STATE = 'last_block_state'
    ATTENTIONS = 'attentions'
    FG_RAW_ATTN_OUTS = 'fg_raw_attn_outs'
    BG_RAW_ATTN_OUTS = 'bg_raw_attn_outs'
    PRE_MIX = "pre_mix"
    SUPPORT_FEAT_1 = "support_feat_1"
    SUPPORT_FEAT_0 = "support_feat_0"
    COARSE_MASKS = "coarse_masks"
    COARSE_MASKS_RW = "coarse_masks_rw"
    QUERY_FEAT_1 = "query_feat_1"
    QUERY_FEAT_0 = "query_feat_0"
    QUERY_FEATS = "query_feats"
    SUPPORT_FEATS = "support_feats"
    MIX = "mix"
    MIX_1 = "mix1"
    MIX_2 = "mix2"
    NSHOT = "nshot"
    DISTILLED_COARSE = "distilled_coarse"
    DISTILLED_LOGITS = "distilled_logits"


class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        if isinstance(d, tuple):
            for t in d:
                setattr(self, t[0], t[1])
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)


def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(b, device) for b in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def linearize_metrics(metrics, id2class=None):
    linearized = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            linearized.update(linearize_metrics(v, id2class))
        if isinstance(v, torch.Tensor):
            # Check if it has a single item
            if len(v.shape) == 0:
                linearized[k] = v.item()
            else:
                for i, elem in enumerate(v):
                    class_name = id2class[i] if id2class is not None else f"class_{i}"
                    linearized[f"{k}_{class_name}"] = elem
                linearized[k] = v.mean().item()
                linearized[f"{k}_fg"] = v[1:].mean().item()
    return linearized

def hierarchical_uniform_sampling(N, M):
    selected_numbers = [0, N]  # Start with the base case M=2
    
    if M == 1:
        return [0]  # Return a single point if M=1
    
    while len(selected_numbers) < M:
        new_numbers = []
        prev_list = selected_numbers[:]
        
        for i in range(len(prev_list) - 1):
            midpoint = (prev_list[i] + prev_list[i + 1]) // 2
            if midpoint not in selected_numbers:
                new_numbers.append(midpoint)
            if len(selected_numbers) + len(new_numbers) >= M:
                break
        
        if not new_numbers:
            break  # Prevent infinite loop if no new numbers can be added
        
        selected_numbers.extend(new_numbers)
        selected_numbers.sort()
    
    return selected_numbers[:M]  # Ensure exactly M numbers

