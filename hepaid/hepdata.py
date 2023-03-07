from json import JSONEncoder
from collections import deque
import gzip
import json
from pathlib import Path
from rich.progress import track
import pickle 

from typing import Dict, List, Union

class DequeEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, deque):
            return list(obj)
        return JSONEncoder.default(self, obj)

def hepstack(
        lhs: Dict,
        slha: Dict, 
        hb_result: Dict, 
        hs_result: Dict) -> Dict:
    '''
    Merge SLHA dict to HiggsBounds and HiggsSignals results 
    into Dict. This is HEPStack Data Structure
    '''
    stack = {'LesHouches': lhs, 'SLHA': slha, 'HiggsBounds': hb_result, 'HiggsSignals': hs_result}
    return stack

def merge_hepstacks(hepstack_list: List, idx: int=0) -> Dict:
    '''
    Takes a list of HEPStack Structures and merge them in a single
    indexed dictionary. The index starts from idx. 
    '''
    hepstack_list_dict = {str(i): file for i,file in enumerate(hepstack_list, idx)}
    return hepstack_list_dict

class HEPDataSet:
    '''
    Creates a data set structure to store objects in a deque, export
    as JSON to disk, reset and load from JSON.

    Methods:
       add(data: Dict) = Adds a data object into the deque.
       reset() = Clear the deque and reset the counter
       save(path: str) = Save to disk as a JSON file.
       load(path: str) = Loads from JSON file.

    '''
    def __init__(self):
        self._data = deque()
        self.counter = 0
        self.complete_stack_ids = []
        self.save_mode = 'pickle' 

    
    def __repr__(self):
        return 'HEPDataSet. Size = {}. Complete Stack Points = {}'.format(
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

    def add(self, data: Union[List, Dict]):
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

    def save_json(self, path):
        json_string = json.dumps(self._data, cls=DequeEncoder)
        with gzip.GzipFile('{}.json.gz'.format(path),"w") as f:
            f.write(json_string.encode())

    def save_pickle(self, path):
        pickled_data = pickle.dumps(self._data)
        with gzip.GzipFile('{}.p.gz'.format(path),"wb") as f:
            f.write(pickled_data)

    def save(self, path):
        self.save_pickle(path)


    def load_json(self, path):
        with gzip.open('{}'.format(path),"r") as f:
            json_string = f.read()
            my_list = json.loads(json_string)
            len_new_data = len(my_list)
            self._data.extend(my_list)
        for idx in range(self.counter, self.counter + len_new_data):
            if not self.is_none(idx=idx):
                self.complete_stack_ids.append(idx)
        self.counter += len_new_data

    def load_pickle(self, path):
        with gzip.open('{}'.format(path),"r") as f:
            depickled_data = f.read()
        try:
            data = pickle.loads(depickled_data)
            len_new_data = len(data)
            self._data += data
            for idx in range(self.counter, self.counter + len_new_data):
                if not self.is_none(idx=idx):
                    self.complete_stack_ids.append(idx)
            self.counter += len_new_data
            return True
        except EOFError:
            return False

    def load(self, path):
        self.load_pickle(path)

    def find_hepdata_files(self, directory: str):
        ''' Identify HEPData files in a directory '''
        directory = Path(directory)
        dataset_files = []
        data_name = 'HEPDataSet'
        for file in directory.iterdir():
            if data_name in file.name:
                dataset_files.append(directory.joinpath(file.name))
        return dataset_files

    def load_from_directory(self, directory: str, percentage: float =1.0):
        dataset_files = self.find_hepdata_files(directory)
        percentage_slice = dataset_files[:int(len(dataset_files)*percentage)]
        corrupted_files = 0
        for file in track(percentage_slice, description=f'Loading HEPDataSets. {percentage*100}%'):
            loaded = self.load(file)
            corrupted_files += 1 if not loaded else 0
        print('EOFError: corrupted files: ', corrupted_files)
            

    def is_none(self, idx, stack: str='SLHA'):
        return True if self._data[idx][stack] is None else False



