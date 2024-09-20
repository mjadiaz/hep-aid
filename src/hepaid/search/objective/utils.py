import json
from omegaconf import OmegaConf

import numpy as np

from typing import Callable, Dict, Any, List, Any
from torch.quasirandom import SobolEngine

from hepaid.search.objective.metrics import VolumeCoverage, s_ns_history
from hepaid.search.parallel.modules import run_x_with_pool
from hepaid.search.objective import Objective

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


def batch_evaluation(
    X: np.ndarray, 
    objective_function: Objective, 
    parallel: bool, 
    n_evaluation_workers: int, 
    add: bool = True, 
    is_normalised: bool = True
) -> None:
    """
    Evaluate a batch of points in the objective function either sequentially or in parallel.

    Parameters:
        X (np.ndarray): A batch of points to evaluate.
        objective_function (Objective): The objective function to evaluate the points.
        parallel (bool): Whether to run the evaluation in parallel.
        n_evaluation_workers (int): Number of workers to use for parallel evaluation.
        add (bool, optional): Whether to add the samples to the objective function. Defaults to True.
        is_normalised (bool, optional): Whether the batch is normalised. Defaults to True.

    Returns:
        None
    """
    if not parallel:
        for x in X:
            objective_function.sample(x, is_normalised=is_normalised, add=add)
    else:
        if is_normalised:
            X = objective_function.space.inverse_transform(X)
        results = run_x_with_pool(X, n_evaluation_workers, objective_function.function)
        if add:
            for sample_dict in results:
                objective_function.add_sample_dict(sample_dict)