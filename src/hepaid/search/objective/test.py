"""
Module that implements test objective functions, import init_function_name to 
initialise the ObjectiveFunction class. Dimensions correspond search space 
dimensionality and number of objectives:
    - init_egg_box_fn: (2,1)
    - init_him_boo_fn: (2,2)
"""

import numpy as np
from hepaid.search.objective.objective_fn import ObjectiveFunction
import importlib.resources as pkg_resources
from omegaconf import OmegaConf


def _load_config_test(path: str):
    """
    Loads a configuration from a file.

    Parameters:
        path (str): The path of the file from which to load the configuration.

    Returns:
        Any: The loaded configuration object.
    """
    with pkg_resources.open_text('hepaid.search.objective.configs', path) as file:
        loaded = OmegaConf.load(file)
    return loaded

def booth(x):
    x, y = x[0], x[1]
    term1 = (x+2*y-7)**2
    term2 = (2*x+y-5)**2
    return np.log(term1 + term2)

def himmelblau(x):
    x, y = x[0], x[1]
    term1 = (x**2+y-11)**2
    term2 = (x+y**2-7)**2
    return np.log(term1 + term2)


def himmelblau_booth(x):
    output = {
        't1': x[0],
        't2': x[1],
        'him': himmelblau(x),
        'boo': booth(x)
        }
    return output


def egg_box(x):
    f =  (2 + np.cos(x[0] / 2) * np.cos(x[1] / 2)) ** 5
    output = {
        'x1': x[0],
        'x2': x[1],
        'f': f,
        }
    return output


def init_him_boo_fn(cas=False):
    """
    Initializes a test function for multi-objective search 
    combining the Himmelblau and Booth functions.

    This function loads the configuration from 'hb_fn.yml' and 
    creates an ObjectiveFunction instance using the himmelblau_booth 
    function.

    Returns:
        ObjectiveFunction: An instance of the ObjectiveFunction class 
        initialized with the himmelblau_booth function and its configuration.
    """
    function_config = _load_config_test('hb_fn.yml')
    function = ObjectiveFunction(
            function=himmelblau_booth,
            function_config=function_config,
            cas=cas
    )
    return function

def init_egg_box_fn(cas=False):
    """
    Initializes the egg box model as described in the original paper on arXiv:1708.06615.

    This function sets up the egg box model, which is a standard test function used in various 
    optimization and sampling problems. The model is characterized by a sinusoidal pattern 
    that resembles an egg carton, making it a challenging landscape for optimization algorithms.

    Returns:
        A function representing the egg box model.
    """
    function_config = _load_config_test('egg_box_fn.yml' )
    function = ObjectiveFunction(
            function=egg_box,
            function_config=function_config,
            cas=cas
    )
    return function

