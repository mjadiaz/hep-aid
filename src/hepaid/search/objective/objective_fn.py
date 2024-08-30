import numpy as np
import pandas as pd
import gzip
import json

from skopt.space.space import Space, Real
from omegaconf import OmegaConf, DictConfig

from pathlib import Path
from typing import Type, Union, Tuple, Dict, Optional, Callable, Any, List

import os

import time
import torch


from hepaid.utils import load_config, save_config


tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

def obj_fn_export(
    objective_function,
    scaler=None
) -> Tuple[np.ndarray, Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exports the objective function data to a format required by PyTorch models and CAS.

    If a scaler is provided, it is applied to Y.

    Parameters:
        objective_function (ObjectiveFunction): The objective function instance.
        scaler (Scaler, optional): An optional scaler to apply to Y. Default is None.

    Returns:
        Tuple[bool, Constraints, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing:
            - A boolean array indicating which samples are valid (i.e., not NaN).
            - The constraints extracted from the objective function.
            - The bounds for the input parameters as a PyTorch tensor.
            - The valid input data as a PyTorch tensor.
            - The valid (and optionally scaled) output data as a PyTorch tensor.
    """
    valid = np.prod(~np.isnan(objective_function.Y), axis=1).astype(np.bool8)

    train_x_ = objective_function.X[valid]
    train_x = torch.tensor(train_x_).to(**tkwargs)

    Y = objective_function.Y[valid]
    if scaler is not None:
        scaler.fit(Y)
        Y = scaler.transform(Y)
    train_y = torch.tensor(Y).to(**tkwargs)

    bounds = torch.tensor(objective_function.bounds).to(**tkwargs)
    constraints = objective_function.constraints
    return valid, constraints, bounds, train_x, train_y


def identify_samples_which_satisfy_constraints(
    X: np.ndarray, constraints: List[Tuple[str, float]]
) -> np.ndarray:
    """
    Identifies which samples satisfy the given m constraints individually.
    Adapted from BoTorch implementation of CAS: https://botorch.org/tutorials/constraint_active_search.

    Parameters:
        X (np.ndarray): A numpy array of shape (n_samples, n_outputs) containing the sample values.
        constraints (List[Tuple[str, float]]): A list of m constraints, where each constraint is a tuple (direction, value).
            The direction is either 'lt' for less than or 'gt' for greater than.

    Returns:
        np.ndarray: A boolean numpy array of shape (n_samples, m) indicating which samples satisfy the constraints.
    """
    successful = np.ones(X.shape)
    for model_index in range(X.shape[-1]):
        these_X = X[..., model_index]
        direction, value = constraints[model_index]
        successful[..., model_index] = (
            these_X < value if direction == "lt" else these_X > value
        )
    return successful

def check_keys_exist(dictionary, keys):
    # Dictionary to hold the keys and their existence status
    keys_status = {}
    # Iterate over the keys to check if they exist in the dictionary
    for key in keys:
        keys_status[key] = key in dictionary
    return keys_status

def identify_samples_which_satisfy_constraints_non_cas(
    Y: np.ndarray, objectives: dict, output_parameters: dict
) -> np.ndarray:
    """
    Identifies which samples satisfy the given m constraints individually, for non-CAS objetive functions.

    Parameters:
        Y (np.ndarray): A numpy array of shape (n_samples, n_outputs) containing the sample values.
        constraints (List[Tuple[str, float]]): A list of m constraints, where each constraint is a tuple (direction, value).
        The direction is either 'lt' for less than or 'gt' for greater than.

    Returns:
        np.ndarray: A boolean numpy array of shape (n_samples, n_outputs) indicating which samples satisfy the constraints.

    """
    succ = np.ones_like(Y).astype(bool)
    key_status = check_keys_exist(objectives, ['double_constraint', 'single_constraint'])
    for i, param in enumerate(output_parameters):
        if key_status['double_constraint'] == True:
            if param in objectives["double_constraint"]:
                select = np.ones(len(Y)).astype(bool)
                select *= Y[:, i] > objectives["double_constraint"][param][0][1]
                select *= Y[:, i] < objectives["double_constraint"][param][1][1]
                succ[:, i] = select
        if key_status['single_constraint'] == True:
            if param in objectives["single_constraint"]:
                select = np.ones(len(Y)).astype(bool)
                if objectives["single_constraint"][param][0] == "lt":
                    select *= Y[:, i] < objectives["single_constraint"][param][1]
                else:
                    select *= Y[:, i] > objectives["single_constraint"][param][1]
                succ[:, i] = select
    return succ


def create_constraints_from_objectives(objectives: dict) -> list:
    """
    Creates a list of constraints from the given objectives dictionary.

    Parameters:
        objectives (Dict):  A dictionary defining the objectives, which must include
                            'double_constraint' and 'single_constraint' keys.

    Returns:
        List: A list of constraints extracted from the objectives dictionary.
    """
    constraints = []
    key_status = check_keys_exist(objectives, ['double_constraint', 'single_constraint'])
    if key_status['double_constraint'] == True:
        for p in objectives["double_constraint"].keys():
            constraints.append(objectives["double_constraint"][p][0])
            constraints.append(objectives["double_constraint"][p][1])
    if key_status['single_constraint'] == True:
        for p in objectives["single_constraint"].keys():
            constraints.append(objectives["single_constraint"][p])
    return constraints


def generate_x_from_sample(sample: Dict, parameters: Dict) -> np.ndarray:
    """
    Generates a 1D numpy array from the given sample and parameters.

    Parameters:
        sample (Dict): A dictionary containing the sample data.
        parameters (Dict): A dictionary defining the parameters to extract from the sample.

    Returns:
        np.ndarray: A 1D numpy array representing the input data vector X.
    """
    X = np.zeros((1, len(parameters)))
    for i, param in enumerate(parameters):
        X[:, i] = np.array(sample[param])
    return X


def generate_y_from_sample(
    sample: Dict,
    objectives: Dict,
) -> np.ndarray:
    """
    Generates the output data vector Y from the given sample and objectives when
    using CAS (Constraint Active Search) mode if is not CAS, please use generate_x_from_sample.

    Parameters:
        sample (Dict): A dictionary containing the sample data.
        objectives (Dict): A dictionary defining the objectives,
                        including 'double_constraint' and 'single_constraint' keys.

    Returns:
        np.ndarray: A numpy array representing the output data vector Y.
    """
    constraints = create_constraints_from_objectives(objectives)
    Y = np.zeros((1, len(constraints)))
    i = 0
    key_status = check_keys_exist(objectives, ['double_constraint', 'single_constraint'])
    if key_status['double_constraint'] == True:
        for p in objectives["double_constraint"].keys():
            Y1 = np.array(sample[p])
            Y[:, i] = Y1
            i += 1
            Y[:, i] = Y1
            i += 1
    if key_status['single_constraint'] == True:
        for p in objectives["single_constraint"].keys():
            Y1 = np.array(sample[p])
            Y[:, i] = Y1
            i += 1
    return Y


def single_bounds(input_parameters: dict, interval: tuple = (0, 1)) -> np.ndarray:
    """
    Generates an array of bounds for the input parameters.

    Parameters:
        input_parameters (Dict): A dictionary of input parameters.
        interval (Tuple[float, float]): A tuple specifying the lower and upper bounds. Default is (0, 1).

    Returns:
        np.ndarray: A 2D array of shape (2, len(input_parameters)), where the first row contains the lower
                    bounds and the second row contains the upper bounds.
    """
    lb, ub = interval
    bounds = np.array([[lb], [ub]])
    bounds = np.repeat(bounds, len(input_parameters))
    return bounds.reshape(2, len(input_parameters))


def create_space(space_config: Dict) -> Space:
    """
    Creates a `Space` object from the given configuration.

    Parameters:
        space_config (Dict): A dictionary representing the configuration of the input space,
                             where each key corresponds to an input parameter and its value is a dictionary specifying
                             'lower', 'upper', and 'distribution' attributes.

    Returns:
        Space: A `Space` object constructed based on the provided configuration.
    """
    dimensions = []
    for p in space_config.keys():
        dimensions.append(
            Real(
                space_config[p]["lower"],
                space_config[p]["upper"],
                space_config[p]["distribution"],
                transform="normalize",
                name=p,
            )
        )
    space = Space(dimensions)
    return space


def extract_input_parameters(space_config: Dict) -> List[str]:
    """
    Extracts the input parameter names from the space configuration.

    Parameters:
        space_config (Dict): A dictionary representing the configuration of the input space.

    Returns:
        List[str]: A list of input parameter names.
    """
    input_parameters = list(space_config.keys())
    return input_parameters


class ObjectiveFunction:
    def __init__(
        self,
        function_config: DictConfig | str,
        function: Callable[..., Dict[str, Any]] | None = None,
        cas: bool = True,
    ) -> None:
        """
        Initializes the ObjectiveFunction class.

        Parameters:
            function_config (Dict | str): Configuration for the function. Can be a dictionary or a path to a configuration file.
            function (Callable[..., Dict[str, Any]] | None): The function to be used. Default is None.
            cas (bool): Flag indicating whether to use CAS (Constraint Active Search) mode. Default is True.
        """
        self.function = function

        if isinstance(function_config, str):
            self.config = load_config(function_config)
        else:
            self.config = function_config

        self.config_name = "obj_fn_config.yml"
        self.initialise_configs_and_space()

        self._dataset = []
        self.X = None
        self.Y = None
        self.cas = cas
        self.counter = 0

    def initialise_configs_and_space(self):
        """
        Initializes configurations and the input space.
        """
        self.input_space_config = self.config.input_space
        self.input_parameters = extract_input_parameters(self.input_space_config)
        self.output_parameters = self.config.output_parameters
        self.objectives = self.config.objectives
        self.space = create_space(self.input_space_config)

    @property
    def constraints(self):
        """
        Returns constraints created from objectives.
        """
        return create_constraints_from_objectives(self.objectives)

    @property
    def bounds(self):
        """
        Returns the bounds of the input parameters.
        """
        return single_bounds(self.input_parameters)

    @property
    def dataset(self):
        """
        Returns the internal dataset.
        """
        return self._dataset

    @property
    def len(self):
        """
        Returns the length of the dataset.
        """
        return len(self._dataset)

    @property
    def satisfactory(self):
        """
        Returns a binary array indicating whether each point satisfies the constraints.
        Adapted from BoTorch CAS implementation.

        Returns:
            np.ndarray: A binary array where each element represents whether the corresponding point satisfies all objectives.
        """
        valid = np.prod(~np.isnan(self.Y), axis=1).astype(np.bool8)
        if self.cas:
            succ = identify_samples_which_satisfy_constraints(
                self.Y[valid], self.constraints
            )
        else:
            Y = self.Y[valid]
            succ = identify_samples_which_satisfy_constraints_non_cas(
                Y, self.objectives, self.output_parameters
            )
        return succ

    def add_xy(self, x: np.ndarray, y: np.ndarray, x_unnorm: np.ndarray) -> None:
        """
        Adds x_unnorm, x and y data points to the internal record .x (unnormalised input),
        .X (normalised input), .Y (unnormalised output).

        Parameters:
            x (np.ndarray): The normalized input data point.
            y (np.ndarray): The output data point.
            x_unnorm (np.ndarray ): The unnormalized input data point.
        """
        if self.X is None:
            self.x = x_unnorm.reshape(1, -1)
            self.X = x.reshape(1, -1)
            self.Y = y.reshape(1, -1)
        else:
            self.x = np.vstack([self.x, x_unnorm])
            self.X = np.vstack([self.X, x])
            self.Y = np.vstack([self.Y, y])

    def add_sample_dict(self, sample_dict: dict):
        """
        Adds a sample dictionary to the dataset and processes it.

        Parameters:
            sample_dict (Dict): The sample dictionary to be added.

        Returns:
            bool: True if the sample is added successfully.
        """
        self._dataset.append(sample_dict)
        self.counter += 1

        x_unnorm, x, y = self.process_sample(None, sample_dict, False)
        self.add_xy(x, y, x_unnorm)
        return True

    def sample(
        self,
        input_point: np.ndarray,
        is_normalised: bool = True,
        add: bool = True,
    ) -> dict:
        """
        Generates an output sample from the input point by calling the objective function.

        Parameters:
            input_point (np.ndarray): The input data point.
            is_normalised (bool): Whether the input point is normalized. Default is True.
            add (bool): Whether to add the output sample to the dataset. Default is True.

        Returns:
            dict: The output sample.
        """

        x_unnorm, _, _ = self.process_sample(input_point, None, is_normalised)

        output_sample = self.function(x_unnorm.flatten())
        if add:
            self.add_sample_dict(output_sample)

        return output_sample

    def process_sample(
        self,
        input_point: np.ndarray | None = None,
        sample_dict: Dict | None = None,
        is_normalised: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes an input point and output sample to produce normalized and unnormalized versions.

        Parameters:
            input_point (np.ndarray | None): The input data point. Default is None.
            sample_dict (Dict | None): The output sample dictionary. Default is None.
            is_normalised (bool): Whether the input point is normalized. Default is True.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The unnormalized input, normalized input, and output array.
        """

        # If input_point is none generate it from sample_dict
        if input_point is None:
            input_point = generate_x_from_sample(sample_dict, self.input_parameters)
            is_normalised = False

        # If input_point is normalised get unnormalised
        if is_normalised:
            normalised_input_point = input_point
            input_point = np.array(
                self.space.inverse_transform(input_point.reshape(1, -1))
            )
        else:
            normalised_input_point = np.array(
                self.space.transform(input_point.reshape(1, -1))
            )

        x_unnorm = input_point
        x = normalised_input_point

        # If output_sample is none generate it from sample_dict
        if sample_dict is not None:
            if self.cas:
                # Cas need repeated dimensions for double constrained dims
                y = generate_y_from_sample(
                    sample_dict,
                    self.objectives,
                )
            else:
                y = generate_x_from_sample(sample_dict, self.output_parameters)
        else:
            y = None
        return x_unnorm, x, y

    def save(self, path: str) -> bool:
        """
        Saves the internal dataset to a compressed JSON file.

        Parameters:
            path (str): The file path to save the dataset.

        Returns:
            bool: True if the dataset is saved successfully, otherwise raises an IOError.

        Raises:
            IOError: If there are issues writing to the file.
        """
        try:
            dataset_path = Path(path)
            directory = dataset_path.parent
            file_path = directory / f"{dataset_path.name}.json.gz"

            # Ensure the directory exists
            directory.mkdir(parents=True, exist_ok=True)

            # Write the dataset to a compressed JSON file
            with gzip.open(file_path, "wt", encoding="utf-8") as file:
                json.dump(self._dataset, file)

            # Save obj_fn_config
            _ = save_config(self.config, directory / self.config_name)

            return True
        except IOError as e:
            raise IOError(f"Failed to save dataset to {path}: {e}")

    def load(self, path, process=True):
        """
        Loads a dataset from a compressed JSON file and obj_fn_config file.

        Parameters:
            path (str): The file path to load the dataset from.
            process (bool): Whether to process the loaded samples. Default is True.
        """
        dataset_path = Path(path)
        name = dataset_path.name
        directory = dataset_path.parent
        file_path = directory.joinpath("{}.json.gz".format(name))
        with gzip.open(file_path, "r") as fin:
            data = json.loads(fin.read().decode("utf-8"))
            len_new_data = len(data)
            self._dataset.extend(data)
        self.counter += len_new_data

        if process:
            for sample in self.dataset:
                input_point = generate_x_from_sample(sample, self.input_parameters)
                x_unnorm, x, y = self.process_sample(input_point, sample, False)
                self.add_xy(x, y, x_unnorm)
        # Save obj_fn_config
        self.config = load_config(directory / self.config_name)

    def as_dataframe(self, satisfactory=True):
        """
        Converts the internal dataset to a pandas DataFrame.

        Parameters:
            satisfactory (bool): Whether to include a 'satisfactory' column based on constraints. Default is True.

        Returns:
            pd.DataFrame: A DataFrame representation of the dataset.
        """
        # Convert the internal dataset to a DataFrame
        df = pd.DataFrame(self._dataset)

        # Optionally add a 'satisfactory' column based on the product of elements in each row
        if satisfactory:
            if hasattr(self, "satisfactory") and self.satisfactory.size > 0:
                df["satisfactory"] = self.satisfactory.prod(axis=1)
            else:
                raise ValueError("The 'satisfactory' attribute is missing or empty.")

        return df


class ObjectiveDataSet:
    """
    Utility class for opening objective datasets with .json.gz format.
    Todo: turn this to a function in Utils.
    """

    def __init__(self):
        self._dataset = []
        self.counter = 0

    @property
    def dataset(self):
        return self._dataset

    def add(self, x):
        self._dataset.append(x)
        return True

    def load_json(self, path):
        """
        Load dataset saved in `path: path/to/file` without the
        extension
        """
        dataset_path = Path(path)
        name = dataset_path.name
        directory = dataset_path.parent
        file_path = directory.joinpath("{}.json.gz".format(name))
        with gzip.open(file_path, "r") as fin:
            data = json.loads(fin.read().decode("utf-8"))
            len_new_data = len(data)
            self._dataset.extend(data)
        self.counter += len_new_data

    def save_dataset(
        self,
        path: str,
    ) -> bool:
        """
        Save internal dataset to disk in path.
        """
        dataset_path = Path(path)
        name = dataset_path.name
        directory = dataset_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory.joinpath("{}.json.gz".format(name))
        with gzip.open(file_path, "w") as file:
            file.write(json.dumps(self._dataset).encode("utf-8"))
        return True
