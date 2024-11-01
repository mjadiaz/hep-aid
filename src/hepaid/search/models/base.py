from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import importlib.resources as pkg_resources
from hepaid.search.objective import Objective
from hepaid.search.objective.metrics import Metrics
from hepaid.utils import save_config, load_config


def _load_config_method(path: str):
    """
    Loads a configuration from a file.

    Parameters:
        path (str): The path of the file from which to load the configuration.

    Returns:
        Any: The loaded configuration object.
    """
    with pkg_resources.open_text('hepaid.search.models.configs', path) as file:
        loaded = OmegaConf.load(file)
    return loaded

class Model:
    """
    Base class for machine learning models. This class is intended to be inherited by specific model implementations.

    Args:
        hyper_parameters (Union[DictConfig, str, None]): The configuration for model hyperparameters. 
            - If a string is provided, it is assumed to be a path to a configuration file.
            - If a `DictConfig` is provided, it is directly used as the hyperparameters.
            - If None is provided, a default configuration is loaded based on the model's class name.
    
    Attributes:
        hp (DictConfig): The hyperparameter configuration for the model.
        hp_name (str): The name of the hyperparameter file.
    """

    def __init__(
        self,
        hyper_parameters: DictConfig | str | None,
    ) -> None:
        """
        Initializes the model with the provided hyperparameters.

        Args:
            hyper_parameters (Union[DictConfig, str, None]): The hyperparameter configuration for the model.
        """
        if isinstance(hyper_parameters, str):
            self.hp = load_config(hyper_parameters)
        elif isinstance(hyper_parameters, DictConfig):
            self.hp = hyper_parameters
        else:
            self.hp = _load_config_method(self.__class__.__name__ + '.yml')
    
        self.hp_name = 'model_hprms.yaml'

    def train(self, train_x, train_y):
        """
        Placeholder method for training the model. This should be implemented in subclasses.

        Args:
            train_x: Training input data.
            train_y: Training target data.

        Returns:
            NotImplementedError: Raised if the method is not implemented in a subclass.
        """
        raise NotImplementedError("The 'train' method should be implemented in a subclass")

    def predict(self, test_x):
        """
        Placeholder method for making predictions using the model. This should be implemented in subclasses.

        Args:
            test_x: Test input data.

        Returns:
            NotImplementedError: Raised if the method is not implemented in a subclass.
        """
        raise NotImplementedError("The 'predict' method should be implemented in a subclass")
    
    def __call__(self, test_x):
        """
        Predict on test_x in evaluation mode.
        """
        return self.predict(test_x=test_x)
    
    def save_checkpoint(self, path: str) -> None:
        """
        Method to save the hyperparameters of the model. 
        todo: Implement saving the actual model to continue search. Currently not needed,
        since it can be trained easily with our dataset sizes.
        """
        path = Path(path)
        save_config(self.hp, path / self.hp_name)

    def load_checkpoint(self):
        """
        todo: Implement loading the model.
        """
        pass