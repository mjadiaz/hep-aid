from typing import Dict, List, Any

import random
import string

from pathlib import Path



def id_generator(size=7, chars=string.ascii_lowercase + string.digits):
    '''
    Generate random string to use it as a temporary id.
    '''
    return ''.join(random.choice(chars) for _ in range(size))


def check_valid_sample_dict(hepstack_point):
    valid = True
    for key in hepstack_point.keys():
        if hepstack_point[key] is None:
            valid = False
            break
    return valid

def get_key_chain(dictionary: dict, keys: list):
    '''
    Given a nested dictionary and a series of keys in a list, returns the
    final key value.
    '''
    result = dictionary
    for key in keys:
        if result is None or key not in result:
            return None
        result = result[key]
    return result

def extract_element_or_return(result):
    """Extracts the single element from a list, or returns the entire list.

    Parameters:
        result: The input value (potentially a list).

    Returns:
        The single element if the input is a list with one item, otherwise the 
        entire input.
    """

    if isinstance(result, list) and len(result) == 1:
        return result[0]  # Extract the element
    else:
        return result  # Return as-is

def extract_parameter_key_chains(obj_hep_config):
    input_parameters = {p: obj_hep_config.input_space[p].key_chain for p in obj_hep_config.input_space.keys()}
    output_parameters = {p: obj_hep_config.output_parameters[p] for p in obj_hep_config.output_parameters.keys()}
    parameters = {**input_parameters, **output_parameters}
    return parameters

def create_simple_dict(obj_hep_config, sample_dict):
    '''
    Create one level dictionary from values located in chains of keys.
    '''
    # Simple output_dict
    output_dict = {}
    parameters = extract_parameter_key_chains(obj_hep_config)
    for param in parameters:
        output_dict[param] = resolve_key_chain(
            sample_dict,
            parameters[param]
            )
    return output_dict

def resolve_key_chain(nested_dictionary: dict, keys: list[str]) -> Any:
    """Traverses a nested dictionary using a list of keys and returns the final value.

    This function iterates through a list of keys, successively accessing nested
    dictionaries until it reaches the final key. If a key is not found at any point,
    it returns `None`.

    Parameters:
        nested_dictionary (dict): The nested dictionary to traverse.
        keys (list[str]): The list of keys that form the path to the desired value.

    Returns:
        Any: The value found at the end of the key path. This can be of any type 
        that is present within the nested dictionary. Returns `None` if any key is missing.
    """
    obj = nested_dictionary
    for a in keys:
        try:
            obj = extract_element_or_return(obj.__getitem__(a))
        except KeyError:
            obj = None
            break
    return obj


def feature_vector(database: list[dict], keys: list[str]) -> list:
    """Extracts values from a nested dictionary dataset based on a sequence of keys.

    This function takes a list of dictionaries representing the dataset and a list
    of keys that form a path to the desired value within each dictionary. It 
    traverses the nested structure using the keys and collects the resulting values
    into a new list.

    Parameters:
        database (list[dict]): The dataset as a list of dictionaries.
        keys (list[str]): A list of keys to follow within each dictionary to 
            reach the target values.

    Returns:
        list: A list of the extracted values.  If a value found in the nested dictionary
            is a list of length 1, the single element of the list is returned instead.
    
    Example:
        Given a dataset `blssm` with the following structure:

        ```
        [
            {"LHE": {"MINPAR": {"entries": {"1": {"value": 123.45}}}}},
            {"LHE": {"MINPAR": {"entries": {"1": {"value": 67.89}}}}},
            # ... more data points
        ]
        ```

        Calling the function like this:

        ```python
        m0 = feature_vector(blssm, ["LHE", "MINPAR", "entries", "1", "value"])
        ```

        will return the list: `[123.45, 67.89, ...]`. 
    """

    feature_array = []
    for i in range(len(database)):
        value = resolve_key_chain(database[i], keys)
        if isinstance(value, list):
            if len(value) == 1:
                value = value[0]
        feature_array.append(value)
    return feature_array


def find_hepdata_files(directory: str, data_name: str = "HEPDataSet") -> list[Path]:
    """Finds HEPDataSet files within a directory.

    This function searches the specified directory for files matching the given data name pattern
    and returns a list of their paths.

    Parameters:
        directory (str): The path to the directory to search.
        data_name (str, optional): The filename pattern to match (e.g., "HEPDataSet*"). Defaults to "HEPDataSet".

    Returns:
        list[Path]: A list of Path objects representing the found HEPDataSet files.
    """
    directory = Path(directory)
    dataset_files = []
    for file in directory.iterdir():
        if data_name in file.name:
            dataset_files.append(directory.joinpath(file.name))
    return dataset_files
