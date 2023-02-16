import re 
import os
from shutil import copy
import numpy as np

from collections.abc import MutableMapping, Mapping
from typing import Dict, List, Tuple, Union

#from hepaid.utils import lhs2dict 



PATTERNS =  dict(   
    block_header=\
            r'(?P<block>BLOCK)\s+(?P<block_name>\w+)\s+((Q=.*)?(?P<q_values>-?\d+\.\d+E.\d+))?(\s+)?(?P<comment>#.*)',
    nmatrix_value =\
            r'(?P<entries>.+)\s+(?P<value>-?\d+\.\d+E.\d+)\s+(?P<comment>#.*)',
    model_param_pattern =\
            r'(?P<entries>.+)\s+(?P<comment>#.*)',
    decay_header=\
            r'DECAY\s+(?P<particle>\w+)\s+(?P<value>-?\d+\.\d+E.\d+)\s+(?P<comment>#.*)',
    decay1l_header=\
            r'DECAY1L\s+(?P<particle>\w+)\s+(?P<value>-?\d+\.\d+E.\d+)\s+(?P<comment>#.*)',
    decay_body_pattern=\
            r'(?P<value>.?\d+\.\d+E.\d+)\s+(?P<entries>.+)\s+(?P<comment>#.*)',
    )

def extract_line_elements(line: str)-> dict:
    #import warnings
    patterns = dict(
        comment = r'(?P<comment>#.*)',
        values = r'(?P<value>.?\d+\.\d+E.\d+)',
        entries = r'(?P<entries>[+-]?\d+)',
        )
    line_elements = {}
    _line = line
    for p in patterns:
        line_elements[p] = re.findall(patterns[p], _line)
        _line = re.sub(patterns[p], '' , _line)
    #if len(_line.strip()) != 0:
    #     warnings.warn(f"Line is not fully read: {_line}")
    return line_elements

def block2dict(block):
    block_dict = {}
    entries_dict = {}
    for i, entries in enumerate(block.entries()):
        entries_dict[','.join(entries)] = {
            'value': block.values()[i], 
            'comment': block.comments()[i],
            'line': block.lines()[i],
        }
    block_dict['entries'] = entries_dict
    block_dict['block_name'] = block.block_name
    block_dict['block_comment'] = block.block_comment
    block_dict['q_values'] = block.q_values
    block_dict['block_category'] = block.block_category
    block_dict['header_line'] = block.header_line
    if (block.block_category == 'DECAY') or (block.block_category == 'DECAY1L'):
        block_dict['pid'] = block.pid
        block_dict['decay_width'] = block.decay_width
    return block_dict

def lheblock2dict(block):
    block_dict = {}
    entries_dict = {}
    for i, entries in enumerate(block.keys()):
        entries_dict[','.join(entries)] = {
            'value': block.values()[i], 
            'comment': block.comments()[i],
            'line': block.lines()[i],
        }
    block_dict['entries'] = entries_dict
    block_dict['block_name'] = block.block_name
    block_dict['block_comment'] = block.block_comment
    block_dict['block_category'] = block.category
    block_dict['header_line'] = block.header_line
    return block_dict

def lhe2dict(lhe):
    lhe_dict = {}
    for i, block in enumerate(lhe.block_list):
        lhe_dict[block] = lheblock2dict(lhe[block])
    return lhe_dict 

def slha2dict(slha):
    slha_dict = {}
    for i, block in enumerate(slha.block_list):
        slha_dict[block] = block2dict(slha[block])
    return slha_dict

#########################################
# Classes for reading LesHouches files. #
# Focusing on Spheno.                   #
#########################################


class BlockLine: 
    def __init__(
        self, 
        entries,
        line_category, 
        line=None
        ):
        self.entries = entries
        self.line_category = line_category
        self.line_format = self.fline(line_category)
        self.line = line

    def fline(self, cat):
        if cat == 'block_header':
            return '{:6s} {:20s}  {:13s}'
        elif cat == 'on_off':
            return '{:6s} {:18s}  {:13s}'
        elif cat == 'value':
            return '{:6s} {:18s}  {:13s}'
        elif cat == 'matrix_value':
            return '{:3s}{:3s} {:18}  {:13s}'

    def __repr__(self):
        return self.fline(self.line_category).format(*self.entries)

    @property
    def comment(self):
        return self.entries[-1]
    @property
    def value(self):
        return self.entries[-2]
    @property 
    def options(self):
        return self.entries[:-2]


class Block(MutableMapping):
    '''
    It holds each line of a block.\n
    Call .set(parameter_number, value) to change the parameter value in the instance.
    '''
    def __init__(
        self, 
        block_name: str, 
        block_comment: str = None, 
        category: str = None, 
        output_mode: bool = False,
        header_line: str = None
        ):
        self.block_name = block_name
        self.block_comment = block_comment
        self.block_body = []
        self.category = category
        self.output_mode = output_mode
        self.header_line = header_line

    def __repr__(self):
        block_header ='{} {}   {:10s}\n'.format('Block',self.block_name,self.block_comment) 
        block_format = ''
        for line in self.block_body:
            block_format += str(line).format(*line.entries) + '\n'
        return block_header+block_format

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key,value):
        self.set(key,value)

    def __delitem__(self, key):
        pass

    def __iter__(self):
        return iter(self.block_body)

    def __len__(self):
        return len(self.block_body)

    def keys(self):
        entries = [i.entries for i in self]
        return entries

    def values(self):
        values = [i.value for i in self]
        return values 

    def comments(self):
        comments = [i.comment for i in self]
        return comments

    def lines(self):
        lines = [i.line for i in self]
        return lines

    def get(self,option):
        '''
        Call set(option, param_value) method to modify the option N with a parameter_value \n.
        -option = Can be an int or a list [i, j]. \n
        -param_value = Can be an int (on/off) or a float. \n
        '''
        for line in self.block_body:
            if (line.line_category == 'matrix_value'):
                if (option == [int(line.entries[0]),int(line.entries[1])]):
                    item = line.entries[2] 
                    break
            if (line.line_category == 'value') & (option == int(line.entries[0])):              
                item = line.entries[1] 
                break                  
            elif (line.line_category == 'on_off') & (option == int(line.entries[0])):
                item = line.entries[1]
                break
        return item

    def set(self,option, param_value):
        '''
        Call set(option, param_value) method to modify the option N with a parameter_value \n.
        -option = Can be an int or a list [i, j]. \n
        -param_value = Can be an int (on/off) or a float. \n
        '''
        for line in self.block_body:
            if (line.line_category == 'matrix_value'):
                if (option == [int(line.entries[0]),int(line.entries[1])]):
                    line.entries[2] = '{:E}'.format(param_value)
                    if self.output_mode:
                        print('{} setted to : {}'.format(line.entries[-1], line.entries[1]))
                    break
            if (line.line_category == 'value') & (option == int(line.entries[0])):              
                line.entries[1] = '{:E}'.format(param_value) 
                if self.output_mode:
                    print('{} setted to : {}'.format(line.entries[-1], line.entries[1])) 
                break                  
            elif (line.line_category == 'on_off') & (option == int(line.entries[0])):
                if isinstance(param_value, int):
                    line.entries[1] = '{}'.format(param_value)
                    if self.output_mode:
                        print('{} setted to : {}'.format(line.entries[-1], line.entries[1]))
                else:
                    if self.output_mode:
                        print('param_value={} is not integer'.format(param_value))
                break



class LesHouches(Mapping):
    '''
    Reading LesHouces files. Format used for input for SPheno.
    '''
    def __init__(self, file_dir, work_dir, model, output_mode=False):
        self.file_dir = file_dir
        self.output_mode = output_mode
        if self.output_mode:
            print(f'Reading LesHouches from : {file_dir}')

        self._blocks = self.read_leshouches(file_dir, output_mode)
        self.block_list = [name.block_name for name in self._blocks]
        self.work_dir = work_dir
        self.model = model
        # Experimental
        self._spheno_blocks = ['MODSEL', 'SMINPUTS', 'SPHENOINPUT', 'DECAYOPTIONS']

    def model_param_blocks(self):
        param_blocks = []
        for block_name in self.block_list:
            if not(block_name in self._spheno_blocks):
                param_blocks.append(block_name)
        return param_blocks

    def __repr__(self):
        return 'LesHouches: {} model: {} blocks'.format(
                        self.model, len(self.block_list)
                        )

    def __getitem__(self, key):
        return self.block(key)

    def keys(self):
        return self.block_list
    def __iter__(self):
        return iter(self._blocks)

    def __len__(self):
        return len(self.block_list)

    def block(self, name):
        block = self.find_block(name.upper(), self._blocks)
        return block  

    @staticmethod
    def find_block(name, block_list):
       try:
           if isinstance(name, str):
               for b in block_list:
                   if b.block_name == name:
                       b_found = b
                       break
                   else:
                       None
           return b_found
       except:
           print('block not found')

    def read_leshouches(self, file_dir, output_mode):
        assert isinstance(file_dir, str) or isinstance(file_dir, dict)
        if isinstance(file_dir, dict):
            lhs = self.read_leshouches_from_dict(
                file=file_dir, 
                output_mode=output_mode
                )
        else:
            lhs = self.read_leshouches_from_dir(
                file_dir=file_dir,
                output_mode=output_mode
            )
        return lhs

    def read_leshouches_from_dir(self, file_dir, output_mode):
        block_list = []
        paterns = dict(
            block_header= r'(?P<block>BLOCK)\s+(?P<block_name>\w+)\s+(?P<comment>#.*)',
            on_off= r'(?P<index>\d+)\s+(?P<on_off>-?\d+\.?)\s+(?P<comment>#.*)',
            value= r'(?P<index>\d+)\s+(?P<value>-?\d+\.\d+E.\d+)\s+(?P<comment>#.*)',
            matrix_value= r'(?P<i>\d+)\s+(?P<j>\d+)\s+(?P<value>-?\d+\.\d+E.\d+)\s+(?P<comment>#.*)'
            )
        with open(file_dir, 'r') as f:
            for line in f:
                # Match in 3 groups a pattern like: '1000001     2.00706278E+02   # Sd_1'
                m_block = re.match(paterns['block_header'], line.upper().strip()) 
                if not(m_block == None):

                    if m_block.group('block_name') in ['MODSEL','SPHENOINPUT','DECAYOPTIONS']:
                        block_list.append(
                            Block(
                                block_name=m_block.group('block_name'), 
                                block_comment=m_block.group('comment'),
                                category= 'spheno_data' ,
                                output_mode=output_mode,
                                header_line=line
                                )
                        )
                        in_block = m_block.group('block_name')
                        block_from = 'spheno_data'

                    else:
                        block_list.append(
                            Block(
                                block_name=m_block.group('block_name'), 
                                block_comment=m_block.group('comment'),
                                category= 'parameters_data',
                                output_mode=output_mode,
                                header_line=line
                                )
                            )
                        in_block = m_block.group('block_name')
                        block_from = 'parameters_data'

                m_body =  re.match(paterns['on_off'], line.strip())
                if not(m_body == None):            
                    self.find_block(
                        in_block,
                        block_list
                        ).block_body.append(
                            BlockLine(
                                entries=list(m_body.groups()),
                                line_category='on_off', 
                                line=line
                                )
                            )
                m_body =  re.match(paterns['value'], line.strip())
                if not(m_body == None):            
                    self.find_block(
                        in_block,
                        block_list
                        ).block_body.append(
                            BlockLine(
                                entries=list(m_body.groups()),
                                line_category='value',
                                line=line
                                )
                            )
                m_body =  re.match(paterns['matrix_value'], line.strip())
                if not(m_body == None):            
                    self.find_block(
                        in_block,
                        block_list
                        ).block_body.append(
                            BlockLine(
                                entries=list(m_body.groups()), 
                                line_category='matrix_value',
                                line=line
                                )
                            )
        return block_list

    def read_leshouches_from_dict(self, file, output_mode):
        block_list = []
        paterns =   dict(   block_header= r'(?P<block>BLOCK)\s+(?P<block_name>\w+)\s+(?P<comment>#.*)',
                on_off= r'(?P<index>\d+)\s+(?P<on_off>-?\d+\.?)\s+(?P<comment>#.*)',
                value= r'(?P<index>\d+)\s+(?P<value>-?\d+\.\d+E.\d+)\s+(?P<comment>#.*)',
                matrix_value= r'(?P<i>\d+)\s+(?P<j>\d+)\s+(?P<value>-?\d+\.\d+E.\d+)\s+(?P<comment>#.*)')

        for b in file:
            m_block = re.match(paterns['block_header'], file[b]['header_line'].upper().strip()) 
            if not(m_block == None):

                if m_block.group('block_name') in ['MODSEL','SPHENOINPUT','DECAYOPTIONS']:
                    block_list.append(
                        Block(
                            block_name=m_block.group('block_name'), 
                            block_comment=m_block.group('comment'),
                            category= 'spheno_data' ,
                            output_mode=output_mode,
                            header_line=file[b]['header_line']
                            )
                        )
                    in_block = m_block.group('block_name')
                    block_from = 'spheno_data'
                else:
                    block_list.append(
                        Block(
                            block_name=m_block.group('block_name'), 
                            block_comment=m_block.group('comment'),
                            category= 'parameters_data',
                            output_mode=output_mode
                            )
                        )
                    in_block = m_block.group('block_name')
                    block_from = 'parameters_data'
            for  k in file[b]['entries']:
                line = file[b]['entries'][k]['line']
                m_body =  re.match(paterns['on_off'], line.strip())
                if not(m_body == None):            
                    self.find_block(
                        in_block,
                        block_list
                        ).block_body.append(
                            BlockLine(
                                entries=list(m_body.groups()),
                                line_category='on_off',
                                line=line
                                )
                            )
                m_body =  re.match(paterns['value'], line.strip())
                if not(m_body == None):            
                    self.find_block(
                        in_block,
                        block_list
                        ).block_body.append(
                            BlockLine(
                                entries=list(m_body.groups()),
                                line_category='value',
                                line=line
                                )
                            )
                m_body =  re.match(paterns['matrix_value'], line.strip())
                if not(m_body == None):            
                    self.find_block(
                        in_block,
                        block_list
                        ).block_body.append(
                            BlockLine(
                                entries=list(m_body.groups()),
                                line_category='matrix_value',
                                line=line
                                )
                            )
        return block_list

    def as_dict(self):
        return lhe2dict(self)

    def new_file(self, new_file_name):
        '''
        Writes a new LesHouches file with the blocks defined in the instance. \n
        Possibly with new values for the parameters and options.
        '''
        new_file_dir = os.path.join(self.work_dir, 'SPheno'+self.model+'_input')

        if not(os.path.exists(new_file_dir)):
            os.makedirs(new_file_dir)
        file_dir=os.path.join(new_file_dir,new_file_name)
        if self.output_mode:
            print(f'Writing new LesHouches in :{file_dir}')

        with open(file_dir,'w+') as f:
            for block in self._blocks:
                head = '{} {}  {:10s}'.format('Block',block.block_name,block.block_comment)+'\n'
                f.write(head)
                for b in block.block_body:
                    f.write(b.line_format.format(*b.entries)+'\n')



#########################################
# Classes for reading SLHA files V2     #
# Focusing on Madraph.                  #
#########################################

class BlockLineSLHA:
    '''
    Line contraining the elements of each line in a SLHA file.

    Args:
    -----
    entries: List[str, str, ..] = n-entries of a line (pid, option index, ...).
    value: str (to float internally). The usual value in scientific notation.
    comment: str
    line_category: DECAY or BLOCK. Internal parameter.
    '''
    def __init__(self, entries, value, comment, line_category, line=None):
        self.entries = entries
        self.value = [float(v) if v != None else None for v in value] 
        self.comment = comment[0] if len(comment) != 0 else None
        self._total_entries_list = [entries] + [value] + [comment] 
        self.line_category = line_category
        self.line = line

    def __repr__(self):
        return self.line
    

class BlockSLHA(MutableMapping):
    '''
    Class that holds each line of a block in the block_body attribute.
    
    Args:
        block_name
        block_comment
        q_values
        block_category: DECAY or BLOCK
        decay_width: if block_categor is Decay
    '''  
    def __init__(   self, block_name, block_comment=None, q_values = None, 
                    block_category=None, decay_width=None, header_line=None):
        self.block_name = block_name
        self.block_comment = block_comment
        self.q_values = q_values 
        self.block_body = []
        self.block_category = block_category
        self.header_line = header_line
        if (self.block_category == 'DECAY') or (self.block_category == 'DECAY1L'):
            self.pid = int(self.block_name.split()[-1])
            self.decay_width = float(decay_width)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key,value):
        self.set(key,value)

    def __delitem__(self, key):
        pass

    def __iter__(self):
        return iter(self.block_body)

    def __len__(self):
        return len(self.block_body)

    def keys(self):
        '''
        A list for all the entries in each line to behave as dict
        '''
        entries = [i.entries for i in self]
        return entries
    
    def entries(self):
        '''
        Note: I don't know why I don't just use .keys()
        '''
        return self.keys()

    def values(self):
        values = [i.value for i in self]
        return values 

    def comments(self):
        comments = [i.comment for i in self]
        return comments

    def lines(self):
        lines = [i.line for i in self]
        return lines
        

    def __repr__(self):
        block_format = self.header_line
        for l in self.lines():
            block_format += l
        return block_format


        
    def set(self, find: Tuple[int], value: float, request:str='value') -> None:
        '''
        Method to change or set the value for a line in a block with 
        acording to the given entries or comment (Which usually is the 
        name or description).

        Args:
            find: Tuple[int] = Tuple for values of entries. ex. (2,25,25)
                in a Decay Block.
            value: float = Value to set.
            request: str = search according to 'value' or 'comment'.
        Return:
            None
        '''
        # Define find iterable
        find = [find] if isinstance(find, int) else find
        find = [str(i) for i in find]

        # Define search according to value or comment
        if request == 'value':
            get_ = lambda line: line.value
        if request == 'comment':
            get_ = lambda line: line.comment

        try:
            for i,b in enumerate(self.block_body):
                line_entries = b.entries
                if line_entries == find:
                    # Save the index
                    b_found = i
                    break
                else:
                    None
            self.block_body[i].value = value
        except:
            print('Entry not found')

    def get(self, find: Tuple[Union[int,str]], request:str='value') -> float:
        '''
        Method to get the value for a line in a block with acording to
        the given entries or comment (Which usually is the name or 
        description).

        Args:
            find: Tuple[int] = Tuple for values of entries. ex. (2,25,25)
                in a Decay Block.
            request: str = search according to 'value' or 'comment'.
        Return:
            requested value
        '''

        # Define find iterable
        find = [find] if not isinstance(find, Tuple) else find
        find = [str(i) for i in find]

        # Define search according to value or comment
        if request == 'value':
            get_ = lambda line: line.value
        if request == 'comment':
            get_ = lambda line: line.comment

        try:
            for b in self.block_body:
                line_entries = b.entries
                if line_entries == find:
                    b_found = b
                    break
                else:
                    None
            return get_(b_found)
        except:
            print('Entry not found')


class SLHA(Mapping):
    '''
    Read a SLHA file (usually the param_card.dat or first section of 
    and LHE file) and stores each block in BlockSLHA classes.
    
    Args:
        file: Union[str, dict] = Path/dict for the SLHA file to read
        work_dir: str = Working directory
        model: str = Name of the model 

    Atributes:
        block_list: List with the names of all the blocks in the SLHA file.

    Methods:
        block(name): Call a Block object stored in the SLHA instance.  
        new_file(new_file_name): Save the instance as a new SLHA file.
    '''
    def __init__(self, 
            file:Union[str,dict], 
            work_dir:str=None, 
            model:str=None,
            ) -> None:

        # Initialize 
        if isinstance(file, dict):
            self._blocks = self.read_from_dict(file)
        else:
            self._blocks = self.read_slha_from_file(file)
        self.block_list = [name.block_name for name in self._blocks]
        self.work_dir = work_dir
        self.model = model

    def __getitem__(self, key):
        return self.block(key)

    def __repr__(self):
        return 'SLHA: {} model: {} blocks'.format(
                        self.model, len(self.block_list)
                        )
    def keys(self):
        return self.block_list
    def __iter__(self):
        return iter(self._blocks)

    def __len__(self):
        return len(self.block_list)

    def block(self, name):
        block = self.find_block(name.upper(), self._blocks)
        return block  

    @staticmethod
    def find_block(name, block_list):
            try:
                if isinstance(name, str):
                    for b in block_list:
                        if b.block_name == name:
                            b_found = b
                            break
                        else:
                            None
                return b_found
            except:
                print('block not found')
    
    def read_from_dict(self, slha_dict):
        block_list = []
        for i, block in enumerate(slha_dict.keys()):
            if slha_dict[block]['block_category'] == 'BLOCK':
                block_list.append(BlockSLHA(
                    block_name=slha_dict[block]['block_name'],
                    block_comment=slha_dict[block]['block_comment'],
                    q_values=slha_dict[block]['q_values'],
                    block_category=slha_dict[block]['block_category'],
                    header_line=slha_dict[block]['header_line']
                    ))
            else:
                block_list.append(BlockSLHA(
                    block_name=slha_dict[block]['block_name'],
                    block_comment=slha_dict[block]['block_comment'],
                    q_values=slha_dict[block]['q_values'],
                    block_category=slha_dict[block]['block_category'],
                    header_line=slha_dict[block]['header_line'],
                    decay_width=slha_dict[block]['decay_width']
                    ))
            for entry in slha_dict[block]['entries'].keys():
                block_list[-1].block_body.append(
                    BlockLineSLHA(
                        entries=entry.split(','),
                        value=slha_dict[block]['entries'][entry]['value'],
                        comment=slha_dict[block]['entries'][entry]['comment'],
                        line_category=slha_dict[block]['block_name'].split()[0],
                        line=slha_dict[block]['entries'][entry]['line']
                        )
                    )
        return block_list
    
    def read_blocks(self, file)->list:
        block_list = []
        in_block = None
        for line in file:
            m_block = re.match(PATTERNS['block_header'], line.upper().strip()) 
            if not(m_block == None):
                block_list.append(BlockSLHA(   block_name=m_block.group('block_name'), 
                                            block_comment=m_block.group('comment'),
                                            q_values = m_block.group('q_values'),
                                            block_category= 'BLOCK' ,
                                            header_line=line
                                            ))
                in_block, block_from = m_block.group('block_name'), 'parameter_data'
                continue

            m_block = re.match(PATTERNS['decay_header'], line.upper().strip()) 
            if not(m_block == None):
                block_name = 'DECAY {}'.format(m_block.group('particle')) 
                block_list.append(BlockSLHA(   block_name=block_name, 
                                            block_comment=m_block.group('comment'),
                                            block_category= 'DECAY' ,
                                            decay_width= m_block.group('value'),
                                            header_line=line
                                            ))
                in_block, block_from = block_name, 'decay_data'
                continue

            m_block = re.match(PATTERNS['decay1l_header'], line.upper().strip()) 
            if not(m_block == None):
                block_name = 'DECAY1L {}'.format(m_block.group('particle')) 
                block_list.append(BlockSLHA(   
                                            block_name=block_name, 
                                            block_comment=m_block.group('comment'),
                                            block_category= 'DECAY1L' ,
                                            decay_width= m_block.group('value'),
                                            header_line=line
                                            ))
                in_block, block_from = block_name, 'decay_data'
                continue

            if in_block is not None:
                line_elements = extract_line_elements(line)
                self.find_block(in_block, block_list).block_body.append(
                        BlockLineSLHA(
                            entries=line_elements['entries'],
                            value=line_elements['values'],
                            comment=line_elements['comment'],
                            line_category=in_block.split()[0],
                            line=line
                            )
                        )
        return block_list

    def read_slha_from_file(self, file_dir):
        with open(file_dir, 'r') as f:
            block_list = self.read_blocks(f)
        return block_list

    def as_txt(self):
        txt = ''
        for b in self._blocks:
            txt += b.__repr__()
        return txt
    
    def as_dict(self):
        '''Return SLHA data as a dictionary'''
        return  slha2dict(self)

    def new_file(self, new_file_name, work_dir=None):
        '''
        Write the instance as a new SLHA file in /{work_dir}/SLHA_{model}/{new_file_name}.
        Args:
        -----
        new_file_name: str
        '''
        if work_dir is None:
            work_dir = self.work_dir
        new_file_dir = os.path.join(work_dir, 'SLHA_'+self.model)

        if not(os.path.exists(new_file_dir)):
            os.makedirs(new_file_dir)
        file_dir=os.path.join(new_file_dir,new_file_name)
        
        with open(file_dir,'w+') as f:
            for block in self._blocks:
                f.write(str(block))





#######################################
# Class for writing a Madgraph script #
#######################################

class MG5Script:
    '''
    Create a object containing mostly* all the necesary commands to compute a process in madgraph with the \n
    text-file-input/script mode. A default or template input file can be preview with the show() method within this class.
    '''
    def __init__(self,work_dir, ufo_model):
        self.work_dir = work_dir
        self.ufo_model = ufo_model 
        self._default_input_file() 

    def import_model(self):
        out = 'import model {} --modelname'.format(self.ufo_model)
        self._import_model = out
      
        
    def define_multiparticles(self,syntax = ['p = g d1 d2  u1 u2  d1bar d2bar u1bar u2bar', 'l+ = e1bar e2bar', 'l- = e1 e2']):
        out = []
        if not(syntax==None):
            [out.append('define {}'.format(i)) for i in syntax]         
            self._define_multiparticles = out            
        else:
            self._define_multiparticles = None

    def process(self,syntax = 'p p > h1 > a a'):
        '''
        Example: InputFile.process('p p > h1 > a a')
        '''
        out = 'generate {}'.format(syntax)
        self._process = out

        
    
    def add_process(self,syntax = None):
        '''
        Example: InputFile.add_process('p p > h2 > a a')
        '''
        out = []
        if not(syntax==None):
            [out.append('add process {}'.format(i)) for i in syntax]         
            self._add_process= out       
        else:
            self._add_process= None
    
    def output(self, name='pph1aa'):
        output_dir = os.path.join(self.work_dir,name)
        out = 'output {}'.format(output_dir)
        self._output = out

        

    def launch(self, name='pph1aa'):
        launch_dir = os.path.join(self.work_dir,name)
        out = 'launch {}'.format(launch_dir)
        self._launch = out

        
    def shower(self,shower='Pythia8'):
        '''
        Call .shower('OFF') to deactivate shower effects.
        '''
        out ='shower={}'.format(shower)
        self._shower = out
 
         
    def detector(self,detector='Delphes'):
        '''
        Call .detector('OFF') to deactivate detector effects.
        '''
        out ='detector={}'.format(detector)
        self._detector = out

               
    def param_card(self, path=None):
        if path==None:
            self._param_card = None
        else:
            self._param_card = path
 
            
    def delphes_card(self, path=None):
        if path==None:
           self._delphes_card = None
        else:
            self._delphes_card = path

            
    def set_parameters(self,set_param=None):
        if set_param==None:
            self._set_parameters = None
        else:
            out = ['set {}'.format(i) for i in set_param]
            self._set_parameters = out

            
    def _default_input_file(self):
        self.import_model()
        self.define_multiparticles()
        self.process()
        self.add_process()
        self.output()
        self.launch()
        self.shower()
        self.detector()
        self.param_card()
        self.delphes_card()
        self.set_parameters()

    def show(self):
        '''
        Print the current MG5InputFile
        '''
        write = [self._import_model, self._define_multiparticles, self._process, self._add_process, self._output, self._launch, self._shower,
        self._detector, '0', self._param_card, self._delphes_card, self._set_parameters, '0'] 
        for w in write:
            if not(w==None):
                if isinstance(w,str):
                    print(w)
                elif isinstance(w,list):
                    [print(i) for i in w]
    def write(self):
        '''
        Write a new madgraph script as MG5Script.txt used internally by madgraph.
        '''
        write = [self._import_model, self._define_multiparticles, self._process, self._add_process, self._output, self._launch, self._shower,
        self._detector, '0', self._param_card, self._delphes_card, self._set_parameters, '0']
        f = open(os.path.join(self.work_dir,'MG5Script.txt'),'w+')
        for w in write:
            if not(w==None):
                if isinstance(w,str):
                    f.write(w+'\n')
                elif isinstance(w,list):
                    [f.write(i+'\n') for i in w]
        f.close()
        return 
        

##########################################
# Class for reading a HiggsBounds output #
##########################################


class HiggsBoundsResults:
    def __init__(self, work_dir, model=None):
        self.work_dir = work_dir
        self.model = model
    
    def read(self, direct_path=None):
        '''
        Read HiggsBounds_results.dat and outputs a dict with all the final results. For example for the BLSSM: \n
        [n, Mh(1), Mh(2), Mh(3), Mh(4), Mh(5), Mh(6), Mhplus(1), HBresult, chan, obsratio, ncomb]
        '''
        if direct_path:
            file_path = direct_path
        else:
            file_path = os.path.join(self.work_dir, 'HiggsBounds_results.dat')
        names = []
        values = []
        with open(file_path , 'r') as f:
            for line in f:
                line = line.strip()
                if ('HBresult' in line) & ('chan' in line):
                    names = line.split()[1:]
                    
                    for subline in f:
                        values = subline.split()
        results = {}
        for n, v in zip(names,values):
            results[n] = float(v)
        
        return results        

    def save(self, output_name,in_spheno_output = True):
        '''Save the HiggsBounds results from the working directory to the Spheno output directory \n.
        To save all the higgs bounds results for scans for example.'''
###########################################


class HiggsSignalsResults:
    def __init__(self, work_dir, model=None):
        self.work_dir = work_dir
        self.model = model
    
    def read(self, direct_path=None):
        '''
        Read HiggsSignals_results.dat and outputs a dict all the results. For example for the BLSSM: \n
        [n, Mh(1), Mh(2), Mh(3), Mh(4), Mh(5), Mh(6), Mhplus(1), csq(mu), csq(mh), csq(tot), nobs(mu), nobs(mh), nobs(tot), Pvalue]
        '''
        if direct_path:
            file_path = direct_path
        else:
            file_path = os.path.join(self.work_dir, 'HiggsSignals_results.dat')
        names = []
        values = []
        with open(file_path , 'r') as f:
            for line in f:
                line = line.strip()
                if ('Mh(1)' in line) & ('Pvalue' in line):
                    names = line.split()[1:]
                    
                    for subline in f:
                        values = subline.split()
        results = {}
        for n, v in zip(names,values):
            results[n] = float(v)
        
        return results        

    def save(self, output_name,in_spheno_output = True):
        '''Save the HiggsSignals results from the working directory to the Spheno output directory \n.
           To save all the higgs Signals results for scans for example.'''
        if in_spheno_output:
            copy(os.path.join(self.work_dir, 'HiggsSignals_results.dat'), os.path.join(self.work_dir,'SPheno'+self.model+'_output' ,'HiggsSignals_results_'+str(output_name)+'.dat'))
        pass                
