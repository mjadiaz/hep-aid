import gzip
import json
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, List, Union, Any

from hepaid.hep.utils import find_hepdata_files
from hepaid.hep.utils import feature_vector
from hepaid.hep.utils import resolve_key_chain


class HEPDataSet:
    """
    Manages and stores HEP (High Energy Physics) data in a versatile way.

    This class provides functionality to:

    *   Store HEP data objects efficiently in a list.
    *   Easily add new data points (single or multiple) as dictionaries.
    *   Track the number of data points and valid data points.
    *   Save the entire dataset to a compressed JSON file.
    *   Load datasets from JSON files (optionally gzipped), with automatic extension handling.
    *   Batch-load multiple HEPDataSet files from a directory, optionally specifying a percentage to load.
    *   Extract feature vectors (lists or NumPy arrays) based on key paths.
    *   Convert the dataset into a pandas DataFrame for analysis.

    Attributes:
        _data (list): The internal storage for HEP data points (dictionaries).
        counter (int): The total number of data points added.
        complete_stack_ids (list): Indices of valid (non-empty) data points.

    Methods:
        add(data: Union[list, dict]): Adds new data points.
        reset(): Clears all data and resets counters.
        save_json(path: str): Saves the dataset to a gzipped JSON file.
        load_json(path: str): Loads data from a gzipped JSON file.
        load_from_directory(directory: str, percentage: float = 1.0, data_name: str = "HEPDataSet"):
            Loads multiple datasets from a directory.
        is_none(idx: int, stack: str = "SLHA"): Checks if a data point is valid. 
        feature_vector(keys: list, as_numpy: bool = False): Extracts a feature vector from the data.
        as_dataframe(keys_dict: dict, as_numpy: bool = True): Converts the dataset to a DataFrame.
    """


    def __init__(self):
        """Initializes an empty HEPDataSet.
        """
        self._data = []
        self.counter = 0
        self.complete_stack_ids = []

    def __repr__(self):
        return "HEPDataSet. Size = {}. Complete Stack Points = {}".format(
            self.counter, len(self.complete_stack_ids)
        )

    def __getitem__(self, key):
        """Allows accessing data points using indexing (e.g., `dataset[5]`).
        """
        return self._data[key]

    def __setitem__(self, key, value):
        """Allows modifying data points using indexing (e.g., `dataset[5] = new_data`).

        Parameters:
            key (int): The index of the data point to modify.
            value (dict): The new data point (dictionary) to assign.
        """
        self._data[key] = value

    def __iter__(self):
        """Allows iterating over the data points in the dataset.
        This makes it possible to use `for data_point in dataset`.
        """
        return iter(self._data)

    def __len__(self):
        """Returns the number of data points in the dataset.
        """
        return self.counter

    @property
    def data(self):
        """Provides read-only access to the internal data storage.

        Returns:
            list: A list of dictionaries representing the HEP data points.
        """
        return self._data

    def add(self, data: Union[List, Dict]):
        """Adds new data to the dataset.

        This method supports adding either a single data point (as a dictionary) 
        or multiple data points (as a list of dictionaries).

        Args:
            data (Union[list, dict]): The data to add. If a list, it should be a list of 
                dictionaries representing individual data points. If a dictionary, it 
                represents a single data point.

        Raises:
            TypeError: If the input `data` is not a list or dictionary.
            ValueError: If a list is provided and any of its elements is not a dictionary.
        """
        if isinstance(data, list):
            self._data.extend(data)
            for idx in range(self.counter, self.counter + len(data)):
                if not self.is_none(idx=idx):
                    self.complete_stack_ids.append(idx)
            self.counter += len(data)
        elif isinstance(data, dict):
            self._data.append(data)
            if not self.is_none(idx=self.counter):
                self.complete_stack_ids.append(self.counter)
            self.counter += 1

    def reset(self):
        """Reset counter and internal data storage.
        """
        self.counter = 0
        self._data.clear()

    def save_json(self, path: str):
        """Saves the dataset to a gzipped JSON file.

        Parameters:
            path (str): The full path where the dataset should be saved, including the desired filename 
                (e.g., "path/to/dataset.json.gz").

        Returns:
            bool: True if the file was saved successfully, False otherwise.
        """
        if not path.endswith(".json.gz"):
            path += ".json.gz" # Add the extension if missing

        dataset_path = Path(path)
        name = dataset_path.name
        directory = dataset_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory.joinpath("{}".format(name))
        with gzip.open(file_path, "w") as file:
            file.write(json.dumps(self._data).encode("utf-8"))
        return True

    def load_json(self, path: str):
        """Loads data from a gzipped JSON file and appends it to the internal data store.

        Parameters:
            path (str): The path to the JSON file. The path can include or 
                        not the extension (with a ".json.gz").

        Returns:
            bool: True if the file was loaded successfully, False otherwise.
        """
        if not path.endswith(".json.gz"):
            path += ".json.gz" # Add the extension if missing

        with gzip.open("{}".format(path), "r") as fin:
            data = json.loads(fin.read().decode("utf-8"))
            len_new_data = len(data)
            self._data.extend(data)
        for idx in range(self.counter, self.counter + len_new_data):
            if not self.is_none(idx=idx):
                self.complete_stack_ids.append(idx)
        self.counter += len_new_data

    def load_from_directory(
        self, 
        directory: str, 
        percentage: float = 1.0, 
        data_name: str = "HEPDataSet"
    ) -> None:
        """Loads HEPDataSet files from a directory.

        This method searches for HEPDataSet files within the specified directory, 
        optionally loading only a percentage of them. It then attempts to load each 
        selected file and reports any corrupted files encountered.

        Parameters:
            directory (str): The path to the directory containing the HEPDataSet files.
            percentage (float, optional): The percentage (0.0 to 1.0) of files to load. 
                Defaults to 1.0 (load all files).
            data_name (str, optional): The file name pattern to search for (e.g., "HEPDataSet*"). 
                Defaults to "HEPDataSet".

        Returns:
            None
        """
        dataset_files = find_hepdata_files(directory, data_name=data_name)
        percentage_slice = dataset_files[: int(len(dataset_files) * percentage)]
        corrupted_files = 0
        for file in percentage_slice:
            loaded = self.load_json(file)
            corrupted_files += 1 if not loaded else 0
        print("EOFError: corrupted files: ", corrupted_files)

    def is_none(self, idx: int, stack: str = "SLHA"):
        """
        Checks if a data file in the Stack is None.

        Parameters:
            idx (int): query number for the dataset
            stack (str): Name of the data file to be checked
        Returns:
            bool: True if the data file is empty (None).
        """
        return True if self._data[idx][stack] is None else False

    def feature_vector(self, keys: list, as_numpy: bool = False):
        """Creates an array from the values associated with the final key of a list of keys to query a nested dictionary (HEPDataSet).

        Parameters:
            keys (list): A list of keys to traverse within the dictionary.
            as_numpy (bool, optional):  If True, returns a NumPy array. 
                                        Otherwise, returns a list. Defaults to False.

        Returns:
            list or numpy.ndarray: An array containing the values associated with the final key
                in the `keys` list for each data point in the dataset. 
        """

        if as_numpy:
            return np.array(feature_vector(self._data, keys)).astype(float)
        else:
            return feature_vector(self._data, keys)

    def as_dataframe(self, keys_dict: dict, as_numpy: bool = True):
        """
        Creates a pandas DataFrame from a dictionary of keys.
        The dictionary of has the form: {'variable' : [key1, key2, ..., final_key], ...}.

        Parameters:
            keys_dict (dict): A dictionary where each key is a variable name and 
                the value is a list of keys to extract from the underlying data.
            as_numpy (bool, optional): If True, returns values as NumPy arrays. Defaults to True.

        Returns:
            pd.DataFrame: A pandas DataFrame where each column corresponds to a variable
                in `keys_dict` and the rows contain the extracted values.
        """
        df = pd.DataFrame()
        for k in keys_dict.keys():
            df[k] = self.feature_vector(keys=keys_dict[k], as_numpy=as_numpy)

        return df
