from itertools import groupby, product
import os
import re
import subprocess
import sys
from typing import Mapping

import collections

import copy
import uuid

from distillfss.utils.utils import PrintLogger, load_yaml, nested_dict_update, write_yaml


def linearize(dictionary: Mapping):
    """
    Linearize a nested dictionary making keys, tuples
    :param dictionary: nested dict
    :return: one level dict
    """
    exps = []
    for key, value in dictionary.items():
        if isinstance(value, collections.abc.Mapping):
            exps.extend(
                ((key, lin_key), lin_value) for lin_key, lin_value in linearize(value)
            )
        elif isinstance(value, list):
            exps.append((key, value))
        elif value is None:
            exps.append((key, [{}]))
        else:
            raise ValueError(
                f"Only dict, lists or None!!! -> {value} is {type(value)} for key {key}"
            )
    return exps


def linearized_to_string(lin_dict):
    def linearize_key(key):
        if type(key) == tuple:
            return f"{key[0]}.{linearize_key(key[1])}"
        return key

    return [(linearize_key(key), value) for key, value in lin_dict]


def extract(elem: tuple):
    """
    Exctract the element of a single element tuple
    :param elem: tuple
    :return: element of the tuple if singleton or the tuple itself
    """
    if len(elem) == 1:
        return elem[0]
    return elem


def delinearize(lin_dict):
    """
    Convert a dictionary where tuples can be keys in na nested dictionary
    :param lin_dict: dicionary where keys can be tuples
    :return:
    """
    # Take keys that are tuples
    filtered = list(filter(lambda x: isinstance(x[0], tuple), lin_dict.items()))
    filtered.sort(key=lambda x: x[0][0])
    # Group it to make one level
    grouped = groupby(filtered, lambda x: x[0][0])
    # Create the new dict and apply recursively
    new_dict = {
        k: delinearize({extract(elem[0][1:]): elem[1] for elem in v})
        for k, v in grouped
    }
    # Remove old items and put new ones
    base_values = {k: v for k, v in lin_dict.items() if (k, v) not in filtered}
    delin_dict = {**base_values, **new_dict}
    return delin_dict


def make_grid(dict_of_list, return_cartesian_elements=False):
    """
    Produce a list of dict for each combination of values in the input dict given by the list of values
    :param return_cartesian_elements: if True return the elements that differs from the base dict
    :param dict_of_list: a dictionary where values can be lists
    :params return_cartesian_elements: return elements multiplied

    :return: a list of dictionaries given by the cartesian product of values in the input dictionary
    """
    # Linearize the dict to make the cartesian product straight forward
    linearized_dict = linearize(dict_of_list)
    # Compute the grid
    keys, values = zip(*linearized_dict)
    if any(map(lambda x: len(x) == 0, values)):
        raise ValueError("There shouldn't be empty lists in grid!!")
    grid_dict = list(dict(zip(keys, values_list)) for values_list in product(*values))
    # Delinearize the list of dicts
    grid = [delinearize(dictionary) for dictionary in grid_dict]
    if return_cartesian_elements:
        ce = list(filter(lambda x: len(x[1]) > 1, linearized_dict))
        return grid, ce
    return grid


def get_excluded_runs(exclude_paths):
    excluded_runs = []
    for exclude_path in exclude_paths:
        logs = filter(lambda x: re.match(r"^run_\d+\.log$", x), os.listdir(exclude_path))
        yamls = map(lambda x: x.replace(".log", ".yaml"), logs)
        for yaml in yamls:
            excluded_runs.append(load_yaml(os.path.join(exclude_path, yaml)))
            
    return excluded_runs


def create_experiment(settings):
    base_grid = settings["parameters"]
    other_grids = settings.get("other_grids")
    exclude_paths = settings.get("exclude_paths", [])
    excluded_runs = get_excluded_runs(exclude_paths)

    print("\n" + "=" * 100)
    complete_grids = [base_grid]
    if other_grids:
        complete_grids += [
            nested_dict_update(copy.deepcopy(base_grid), other_run)
            for other_run in other_grids
        ]

    grids, dot_elements = zip(
        *[
            make_grid(grid, return_cartesian_elements=True)
            for grid in complete_grids
        ]
    )
    
    start_from_grid = settings.get("start_from_grid", 0)
    grids = grids[start_from_grid:]
    
    print(f"Starting from grid {start_from_grid} out of {len(complete_grids)} complete grids")
    
    # linearize list of list into list
    grids = [grid for run in grids for grid in run]
    initial_runs_len = len(grids)
    print(f"Initial runs: {initial_runs_len}")
    
    # remove excluded runs
    grids = [
        grid for grid in grids
        if grid not in excluded_runs
    ]
    print(f"Read excluded runs: {len(excluded_runs)}")
    print(f"Remaining runs: {len(grids)}")
    
    return grids

def default_run_complete(run_log_path):
    """Default completion check: the run log must contain at least one test-result line.

    Replace this function (or pass a custom callable to find_resume_run /
    find_grid_to_resume) if the finish signal ever changes.
    """
    if not os.path.exists(run_log_path):
        return False
    with open(run_log_path, "r") as f:
        for line in f:
            if "] Test - " in line:
                return True
    return False


def find_resume_run(grid_folder, is_complete=None):
    """Return the index of the first run to execute when resuming a grid.

    Scans run_*.log files, finds the last complete one, and returns
    last_complete + 1.  Pass a custom *is_complete(log_path) -> bool*
    callable to change how completion is detected.
    """
    if is_complete is None:
        is_complete = default_run_complete

    run_indices = []
    for fname in os.listdir(grid_folder):
        m = re.match(r"^run_(\d+)\.log$", fname)
        if m:
            run_indices.append(int(m.group(1)))
    run_indices.sort()

    if not run_indices:
        return 0

    last_complete = -1
    for idx in run_indices:
        if is_complete(os.path.join(grid_folder, f"run_{idx}.log")):
            last_complete = idx

    return last_complete + 1


def find_grid_to_resume(parameters, grid_name, out_folder, is_complete=None):
    """Find the most recent grid folder whose hyperparams match *parameters*.

    Searches out_folder for directories ending in ``_{grid_name}``, sorted
    newest-first, and returns (folder_path, resume_from_index) for the first
    match, or (None, None) if none is found.

    Pass a custom *is_complete(log_path) -> bool* callable to change how
    completion is detected.
    """
    if not os.path.exists(out_folder):
        return None, None

    suffix = f"_{grid_name}"
    candidates = sorted(
        [
            d for d in os.listdir(out_folder)
            if os.path.isdir(os.path.join(out_folder, d)) and d.endswith(suffix)
        ],
        reverse=True,
    )

    for grid_dir in candidates:
        full_path = os.path.join(out_folder, grid_dir)
        hyperparams_path = os.path.join(full_path, "hyperparams.yaml")
        if not os.path.exists(hyperparams_path):
            continue
        if load_yaml(hyperparams_path) == parameters:
            return full_path, find_resume_run(full_path, is_complete=is_complete)

    return None, None


class ParallelRun:
    slurm_command = "sbatch"
    slurm_multi_gpu_script = "slurm/launch_run_multi_gpu"
    slurm_script_first_parameter = "--parameters="
    out_extension = "log"
    param_extension = "yaml"
    slurm_stderr = "-e"
    slurm_stdout = "-o"

    def __init__(self, params: dict, multi_gpu=False, logger=None, run_name=None, slurm_script=None):
        self.params = params
        self.multi_gpu = multi_gpu
        self.logger = logger or PrintLogger()
        self.run_name = run_name
        self.slurm_script = slurm_script or "slurm/launch_run"
        if "." not in sys.path:
            sys.path.extend(".")

    def launch(self, only_create=False, script_args=[]):
        out_file = f"{self.run_name}.{self.out_extension}"
        param_file = f"{self.run_name}.{self.param_extension}"
        write_yaml(self.params, param_file)
        slurm_script = self.slurm_multi_gpu_script if self.multi_gpu else self.slurm_script
        command = [
            self.slurm_command,
            self.slurm_stdout,
            out_file,
            self.slurm_stderr,
            out_file,
            slurm_script,
            self.slurm_script_first_parameter + param_file,
            *script_args,
        ]
        if only_create:
            self.logger.info(f"Creating command: {' '.join(command)}")
        else:
            self.logger.info(f"Launching command: {' '.join(command)}")
            subprocess.run(command)