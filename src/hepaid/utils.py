from typing import Dict, List

def br_2dict(block_body: List) -> Dict:
    '''Turn branching ratios into a dict'''
    br_dict = {'value':[], 'nda':[], 'pids':[],'comment':[]}
    for line in block_body:
        br_dict['value'] += [float(line.value) if line.value != None else None]
        br_dict['nda'] += [int(line.entries[0])]
        br_dict['pids'] += [int(i) for i in line.entries[1:]], 
        br_dict['comment'] += [str(line.comment)]
    return br_dict

def decay_block_2dict(block) -> Dict:
    '''Turn Decay Block info into a dict'''
    body_dict = {
        'block': block.block_name,
        'comment': block.block_comment,
        'q_values': block.q_values,
        'pid': block.pid,
        'decay_width': block.decay_width,
        'block_category': block.block_category,
        'branching_ratios': br_2dict(block.block_body)
        }
    return body_dict 

def body_values_2dict(block_body: List) -> Dict:
    '''Turn the value for every non Decay block body in to a dict'''
    values_dict = {'value':[], 'entries':[],'comment':[]}
    for line in block_body:
        values_dict['value'] += [float(line.value) if line.value != None else None]
        try:
        # Some blocks like 'SPINFO' have string entries
            values_dict['entries'] += [int(i) for i in line.entries], 
        except ValueError:
            values_dict['entries'] += [i for i in line.entries], 
        values_dict['comment'] += [str(line.comment)]
    return values_dict 

def body_values_2dict_lhs(block_body: List) -> Dict:
    '''Turn the value for every non Decay block body in to a dict in LesHouches Files'''
    values_dict = {'value':[], 'entries':[],'comment':[]}
    for line in block_body:
        values_dict['value'] += [float(line.value) if line.value != None else None]
        try:
        # Some blocks like 'SPINFO' have string entries
            values_dict['entries'] += [int(i) for i in line.options], 
        except ValueError:
            values_dict['entries'] += [i for i in line.options], 
        values_dict['comment'] += [str(line.comment)]
    return values_dict 

def generic_block_2dict(block) -> Dict:
    '''Turn every Non-Decay block into a dict'''
    body_dict = {
        'block_name': block.block_name,
        'comment': block.block_comment,
        'q_values': block.q_values,
        'block_category': block.block_category,
        'values': body_values_2dict(block.block_body)
        }
    return body_dict 

def generic_block_2dict_lhs(block) -> Dict:
    '''Turn every Non-Decay block into a dict in LesHouches Files'''
    body_dict = {
        'block_name': block.block_name,
        'comment': block.block_comment,
        'values': body_values_2dict_lhs(block.block_body)
        }
    return body_dict 
def block2dict(block) -> Dict:
    '''Turn any block into dict'''
    if block.block_category == 'DECAY':
        return decay_block_2dict(block)
    else:
        return generic_block_2dict(block)

def slha2dict(slha) -> Dict:
    '''Convert a SLHA object into dict'''
    slha_dict = {b: block2dict(slha.block(b)) for b in slha.block_list}
    return slha_dict

def lhs2dict(lhs) -> Dict:
    '''Convert a LesHouches object into dict'''
    lhs_dict = {b: generic_block_2dict_lhs(lhs[b]) for b in lhs.block_list}
    return lhs_dict


def merge_slha_files(slha_list: List, idx: int=0) -> Dict:
    '''
    Takes a list of SLHA files and merge them in a single
    indexed dictionary. The index starts from idx. 
    '''
    slha_list_dict = {str(i): file for i,file in enumerate(slha_list, idx)}
    return slha_list_dict

def dict2json(my_dict: Dict, path: str) -> None:
    '''Takes dictionaries and save it in path as a compressed JSON file'''
    import json
    import gzip
    json_string = json.dumps(my_dict)
    with gzip.GzipFile("{}.json.gz".format(path), "w") as f:
        f.write(json_string.encode())

def json2dict(path: str) -> Dict:
    '''Loads a copressed JSON file from path and returns a dict'''
    import json
    import gzip
    with gzip.open("{}.json.gz".format(path), "r") as f:
        json_string = f.read()
        my_dict = json.loads(json_string)
        return my_dict 

def merge_datasets(ds_1: Dict, ds_2: Dict) -> Dict:
    '''
    Merge two DataSets structures into one. 
    To-do: Implement SLHADataSet/HEPStack type.
    '''
    import numpy as np
    max_1 = np.fromiter(ds_1.keys(),dtype=int).max()
    max_2 = np.fromiter(ds_2.keys(),dtype=int).max()
    if max_2 > max_1:
        new_dataset = {**ds_1, **ds_2}
    else:
        new_dataset = {**ds_2, **ds_1}
    return new_dataset

