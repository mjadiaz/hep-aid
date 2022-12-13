from hepaid.hepread import SLHA, BlockSLHA, BlockLineSLHA
from typing import Dict, List

def br_2dict(block_body: List[BlockLineSLHA]) -> Dict:
    '''Turn branching ratios into a dict'''
    br_dict = {'value':[], 'nda':[], 'pids':[],'comment':[]}
    for line in block_body:
        br_dict['value'] += [float(line.value) if line.value != None else None]
        br_dict['nda'] += [int(line.entries[0])]
        br_dict['pids'] += [int(i) for i in line.entries[1:]], 
        br_dict['comment'] += [str(line.comment)]
    return br_dict

def decay_block_2dict(block: BlockSLHA) -> Dict:
    '''Turn Decay Block info into a dict'''
    body_dict = {
        'block': block.block_name,
        'comment': block.block_comment,
        'q_values': block.q_values,
        'pid': block.pid,
        'decay_with': block.decay_width,
        'block_category': block.block_category,
        'branching_ratios': br_2dict(block.block_body)
        }
    return body_dict 

def body_values_2dict(block_body: List[BlockLineSLHA]) -> Dict:
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

def generic_block_2dict(block: BlockSLHA) -> Dict:
    '''Turn every Non-Decay block into a dict'''
    body_dict = {
        'block_name': block.block_name,
        'comment': block.block_comment,
        'q_values': block.q_values,
        'block_category': block.block_category,
        'values': body_values_2dict(block.block_body)
        }
    return body_dict 

def block2dict(block: BlockSLHA) -> Dict:
    '''Turn any block into dict'''
    if block.block_category == 'DECAY':
        return decay_block_2dict(block)
    else:
        return generic_block_2dict(block)

def slha2dict(slha: SLHA) -> Dict:
    '''Convert a SLHA object into dict'''
    slha_dict = [block2dict(slha.block(b)) for b in slha.block_list]
    return slha_dict


