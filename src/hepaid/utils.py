from omegaconf import OmegaConf
from typing import Callable, Dict, Any, List, Any
import importlib.resources as pkg_resources
from pathlib import Path

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
    loaded = OmegaConf.load(path)
    return loaded

