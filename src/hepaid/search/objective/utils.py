import json
from omegaconf import OmegaConf

import numpy as np

from typing import Callable, Dict, Any, List, Any
from torch.quasirandom import SobolEngine

from hepaid.search.objective.metrics import VolumeCoverage, s_ns_history
from hepaid.search.parallel.modules import run_x_with_pool


def save_config(config: Any, path: str) -> bool:
    """
    Saves the given configuration to a file.

    Parameters:
        config (Any): The configuration object to be saved.
        path (str): The path of the file where the configuration will be saved.

    Returns:
        bool: True if the configuration is saved successfully.
    """
    with open(path, "w+") as fp:
        OmegaConf.save(config=config, f=fp.name, resolve=True)
    return True


def load_config(path: str) -> Any:
    """
    Loads a configuration from a file.

    Parameters:
        path (str): The path of the file from which to load the configuration.

    Returns:
        Any: The loaded configuration object.
    """
    with open(path, "r") as fp:
        loaded = OmegaConf.load(path)
    return loaded


def dict_to_eval_string(obj_dict: Dict) -> str:
    """
    Converts a dictionary of constraints into an evaluation string.

    Parameters:
        obj_dict (Dict): A dictionary containing constraints with keys 'double_constraint' and 'single_constraint'.

    Returns:
        str: A string representing the constraints in an evaluable format for Pandas Dataframes!
    """
    eval_string = ""
    for key, value in obj_dict.items():
        if key == "double_constraint":
            for constraint, conditions in value.items():
                eval_string += (
                    f"{conditions[0][1]} < {constraint} < {conditions[1][1]} and "
                )
        elif key == "single_constraint":
            for constraint, condition in value.items():
                if condition[0] == "gt":
                    eval_string += f"{condition[1]} < {constraint} and "
                else:
                    eval_string += f"{constraint} < {condition[1]} and "
        elif key == "gaussian_constraint":
            # Handle Gaussian constraints if needed
            pass

    # Removing the trailing 'and ' from the eval_string
    eval_string = eval_string.rstrip("and ")
    return eval_string


def us_ut_iteration(x, met, select, iteration_step):
    vc = VolumeCoverage()
    r_id = 0
    iterations = met.n_total_points
    r_parameter = met.r_parameter
    for i in range(len(x)):
        vc.add(x[i], r_parameter[r_id])
        if i == 0:
            continue
        if i == iterations[r_id]:
            r_id += 1
    s, ns = s_ns_history(vc.history, select)
    us_ut = (s / vc.cumulative)[::iteration_step]
    ns_ut = (ns / vc.cumulative)[::iteration_step]

    return us_ut, ns_ut


def generate_initial_dataset(
    n_points: int,
    objective_function,
    parallel: bool = True,
    n_workers: int = 1,
) -> bool:
    """
    Generates an initial dataset using a Sobol sequence.

    Parameters:
        n_points (int): Number of points to generate.
        objective_function (Any): The objective function instance containing the function and bounds.
        parallel (bool): Whether to run the function in parallel. Default is True.
        n_workers (int): Number of workers for parallel execution. Default is 1.

    Returns:
        bool 
    """

    function = objective_function.function
    bounds = objective_function.bounds

    lb, ub = bounds
    dim = len(lb)

    X = lb + (ub - lb) * SobolEngine(dim, scramble=True).draw(n_points).cpu().numpy()

    if parallel:
        X = np.array(objective_function.space.inverse_transform(X))

        results = run_x_with_pool(X=X, n_workers=n_workers, function=function)

        for result in results:
            objective_function.add_sample_dict(result)
    else:
        for x in X:
            objective_function.sample(x.reshape(1, -1), True)
    return True
