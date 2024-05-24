from pathlib import Path
from omegaconf import OmegaConf, DictConfig


from hepaid.search.objective.objective_fn import ObjectiveFunction
from hepaid.search.objective.metrics import Metrics
from hepaid.search.objective.utils import save_config, load_config


class Policy:
    """
    Policy base class to be inherited by Policy search methods. Load hyperparameters 
    config from a file if a string path is provided. Initialize metrics and iteration counter. 
    Creates a general save path.

    Attributes:
        objective_function (ObjectiveFunction): The objective function to be optimized.
        hp (DictConfig): The hyperparameters for the policy.
        hp_name (str): The name of the hyperparameters file. Default is `hprms.yaml`.
        metrics (Metrics): An instance of the Metrics class to track performance.
        save_path (Path): Path to save the datasets and checkpoints.
        iteration (int): The current iteration counter.

    Methods:
        __init__(self, objective_function: ObjectiveFunction, hyper_parameters: DictConfig | str) -> None:
            Initialize the Policy class.
        
        save_checkpoint(self, iteration: int) -> None:
            Save the objective function checkpoint, metrics, and search hyperparameters.

        load_checkpoint(self) -> None:
            Load the objective function and metrics to continue the search or for data analysis.
    """
 

    def __init__(
        self,
        objective_function: ObjectiveFunction,
        hyper_parameters: DictConfig | str,
    ) -> None:
        """
        Initialize the Policy class. Load hyperparameters config from a file if a string path is 
        provided. Initialize metrics and iteration counter. Creates a general save path.

        Parameters:
            objective_function (ObjectiveFunction): The objective function to be optimized.
            hyper_parameters (DictConfig | str): The hyperparameters for the policy. 
                                            Can be a dictionary or a path to a configuration file.
        """
        self.objective_function = objective_function

        if isinstance(hyper_parameters, str):
            self.hp = load_config(hyper_parameters)
        else:
            self.hp = hyper_parameters
        self.hp_name = 'hprms.yaml'

        self.metrics = Metrics()

        # Create general save path
        self.save_path = Path("datasets") / Path(self.hp.run_name)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
            print(f"Directory '{self.save_path}' created.")

        # Initialise iteration
        self.iteration = 0

    def save_checkpoint(self, iteration: int) -> None:
        """
        Save the objective function checkpoint, metrics, and search hyperparameters.
        The name for search hyperparameters can be modified by changing the .hp_name 
        attribute.

        Parameters:
            iteration (int): The current iteration number.
        """
        if (iteration % self.hp.checkpoint.n_step_save
                == 0) or (iteration == self.hp.total_iterations):
            self.objective_function.save(self.save_path /
                                         self.hp.checkpoint.name)
            self.metrics.save(self.save_path, iteration)
            save_config(self.hp, self.save_path / self.hp_name)

    def load_checkpoint(self):
        """
        Load the objective function and metrics to continue the search or for data analysis.
        """
        self.objective_function.load(self.save_path / self.hp.checkpoint.name)
        self.metrics.load(self.save_path)
        self.iteration = self.metrics.metrics['iteration'][-1]
