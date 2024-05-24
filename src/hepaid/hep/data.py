import gzip
import json
import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from json import JSONEncoder
from collections import deque
from typing import Dict, List, Union


class DequeEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, deque):
            return list(obj)
        return JSONEncoder.default(self, obj)


def merge_hepstacks(hepstack_list: List, idx: int = 0) -> Dict:
    """
    Takes a list of HEPStack Structures and merge them in a single
    indexed dictionary. The index starts from idx.
    """
    hepstack_list_dict = {str(i): file for i, file in enumerate(hepstack_list, idx)}
    return hepstack_list_dict


def _get_key_chain(obj, args):
    """Apply successive requests to an obj that implements __getitem__ and
    return result if something is found, else return default"""
    for a in args:
        try:
            obj = obj.__getitem__(a)
        except:
            obj = None
            break
    return obj


def feature_vector(database, keys):
    """
    Creates a list with the values obtained from querying
    a HEPDataSet object with a chain of keys. Example:
    m0 = feature_vector(blssm, ['LHE', 'MINPAR', 'entries', '1', 'value'])

    Added assertion since sometimes the value is in a list of len 1.
    """
    feature_array = []
    for i in range(len(database)):
        value = _get_key_chain(database[i], keys)
        if isinstance(value, list):
            # assert len(value) == 1, \
            #    'Value for key chain has more than one value'
            if len(value) == 1:
                value = value[0]
        feature_array.append(value)
    return feature_array


def find_hepdata_files(directory: str, data_name: str = "HEPDataSet"):
    """
    Identify `data_name = HEPDataSet` files in a directory. Default name is HEPDataSet
    """
    directory = Path(directory)
    dataset_files = []
    for file in directory.iterdir():
        if data_name in file.name:
            dataset_files.append(directory.joinpath(file.name))
    return dataset_files


class HEPDataSet:
    """
    Creates a data set structure to store objects in a deque, export
    as JSON to disk, reset and load from JSON.

    Methods:
       add(data: Dict) = Adds a data object into the deque.
       reset() = Clear the deque and reset the counter
       save(path: str) = Save to disk as a JSON file.
       load(path: str) = Loads from JSON file.

    """

    def __init__(self):
        # self._data = deque()
        self._data = []
        self.counter = 0
        self.complete_stack_ids = []
        self.save_mode = "pickle"

    def __repr__(self):
        return "HEPDataSet. Size = {}. Complete Stack Points = {}".format(
            self.counter, len(self.complete_stack_ids)
        )

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return self.counter

    @property
    def data(self):
        return self._data

    def add(self, data: Union[List, Dict]):
        """Add a data point or a list of data points"""
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
        self.counter = 0
        self._data.clear()

    def save_json(self, path: str):
        """
        Save dataset in json.gz format.
        Parameters:
            path (str): path to dataset.
        """
        dataset_path = Path(path)
        name = dataset_path.name
        directory = dataset_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        file_path = directory.joinpath("{}.json.gz".format(name))
        with gzip.open(file_path, "w") as file:
            file.write(json.dumps(self._data).encode("utf-8"))
        return True

    def load_json(self, path: str):
        """
        The path must include the format.
        Args:
        ----
        path: str = "path/to/data/set.json.gz"
        """
        with gzip.open("{}".format(path), "r") as fin:
            data = json.loads(fin.read().decode("utf-8"))
            len_new_data = len(data)
            self._data.extend(data)
        for idx in range(self.counter, self.counter + len_new_data):
            if not self.is_none(idx=idx):
                self.complete_stack_ids.append(idx)
        self.counter += len_new_data

    def load_from_directory(
        self, directory: str, percentage: float = 1.0, data_name: str = "HEPDataSet"
    ):
        dataset_files = find_hepdata_files(directory, data_name=data_name)
        percentage_slice = dataset_files[: int(len(dataset_files) * percentage)]
        corrupted_files = 0
        for file in percentage_slice:
            loaded = self.load_json(file)
            corrupted_files += 1 if not loaded else 0
        print("EOFError: corrupted files: ", corrupted_files)

    def is_none(self, idx, stack: str = "SLHA"):
        return True if self._data[idx][stack] is None else False

    def feature_vector(self, keys: list, as_numpy: bool = False):
        """
        Create an array from a list of `keys`: [key, ..., key]. If `as_numpy` is
        True returns a float np.array.
        """
        if as_numpy:
            return np.array(feature_vector(self._data, keys)).astype(float)
        else:
            return feature_vector(self._data, keys)

    def as_dataframe(self, keys_dict: dict, as_numpy: bool = True):
        """
        Create a pandas DataFrame from a dictionary of the form:
        {'variable' : [key, ..., key], ...}
        """
        df = pd.DataFrame()
        for k in keys_dict.keys():
            df[k] = self.feature_vector(keys=keys_dict[k], as_numpy=as_numpy)

        return df
