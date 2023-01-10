from json import JSONEncoder
from collections import deque
import gzip
import json

class DequeEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, deque):
            return list(obj)
        return JSONEncoder.default(self, obj)

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
    
    def __repr__(self):
        return 'HEPDataSet. Size = {}'.format(self.counter)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return self.counter

    def add(self, data):
        self.counter += 1
        self._data.append(data)

    def reset(self):
        self.counter = 0
        self._data.clear()

    def save(self, path):
        json_string = json.dumps(self._data, cls=DequeEncoder)
        with gzip.GzipFile('{}.json.gz'.format(path),"w") as f:
            f.write(json_string.encode())

    def load(self, path):
        with gzip.open('{}'.format(path),"r") as f:
            json_string = f.read()
            my_list = json.loads(json_string)
            len_new_data = len(my_list)
            self._data.extend(my_list)
        self.counter += len_new_data


