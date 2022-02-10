import re 
import os
from shutil import copy
import numpy as np



#########################################
# Classes for reading LesHouches files. #
# Focusing on Spheno.                   #
#########################################




class BlockLine: 
	def __init__(self, entries, line_category):
		self.entries = entries
		self.line_category = line_category
		self.line_format = self.fline(line_category)
	def fline(self, cat):
		if cat == 'block_header':
			return '{:6s} {:20s}  {:13s}'
		elif cat == 'on_off':
			return '{:6s} {:18s}  {:13s}'
		elif cat == 'value':
			return '{:6s} {:18s}  {:13s}'
		elif cat == 'matrix_value':
			return '{:3s}{:3s} {:18}  {:13s}'
	@property
	def comment(self):
		return self.entries[-1]
	@property
	def value(self):
		return self.entries[-2]
	@property 
	def options(self):
		return self.entries[:-2]


class Block:
    '''
    ## Block
    It holds each line of a block.\n
    Call .show() to print a block. \n
    Call .set(parameter_number, value) to change the parameter value in the instance.
    '''
    def __init__(self, block_name, block_comment=None, category=None, output_mode=False):
        self.block_name = block_name
        self.block_comment = block_comment
        self.block_body = []
        self.category = category
        self.output_mode = output_mode

    def show(self):
        '''
        Print block information in the LesHouches format.
        '''
        print('{} {}   {:10s}'.format('Block',self.block_name,self.block_comment))
        for b in self.block_body:
            print(b.line_format.format(*b.entries))
    
    
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



class LesHouches:
    '''
    ## LesHuches
    Read a LesHouches file and stores each block in block classes. \n
    - To get all the names of the blocks call .block_list. \n
    - work_dir is the directory where all the outputs will be saved. \n
    - The new LesHouches files will be saved in a folder called SPhenoMODEL_input since is the input for spheno.
    '''
    def __init__(self, file_dir, work_dir, model, output_mode=False):
        self.file_dir = file_dir
        self.output_mode = output_mode
        if self.output_mode:
            print(f'Reading LesHouches from : {file_dir}')

        self._blocks = LesHouches.read_leshouches(file_dir, output_mode)
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
	


    def block(self, name):
        block = LesHouches.find_block(name.upper(), self._blocks)
        return block  


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

    def read_leshouches(file_dir, output_mode):
        block_list = []
        paterns =   dict(   block_header= r'(?P<block>BLOCK)\s+(?P<block_name>\w+)\s+(?P<comment>#.*)',
                            on_off= r'(?P<index>\d+)\s+(?P<on_off>-?\d+\.?)\s+(?P<comment>#.*)',
                            value= r'(?P<index>\d+)\s+(?P<value>-?\d+\.\d+E.\d+)\s+(?P<comment>#.*)',
                            matrix_value= r'(?P<i>\d+)\s+(?P<j>\d+)\s+(?P<value>-?\d+\.\d+E.\d+)\s+(?P<comment>#.*)')



        with open(file_dir, 'r') as f:
            for line in f:
                # Match in 3 groups a pattern like: '1000001     2.00706278E+02   # Sd_1'
                m_block = re.match(paterns['block_header'], line.upper().strip()) 
                if not(m_block == None):

                    if m_block.group('block_name') in ['MODSEL','SPHENOINPUT','DECAYOPTIONS']:
                        block_list.append(Block(    block_name=m_block.group('block_name'), 
                                                    block_comment=m_block.group('comment'),
                                                    category= 'spheno_data' ,
                                                    output_mode=output_mode))
                        in_block, block_from = m_block.group('block_name'), 'spheno_data'

                    else:
                        block_list.append(Block(        block_name=m_block.group('block_name'), 
                                                        block_comment=m_block.group('comment'),
                                                        category= 'parameters_data',
                                                        output_mode=output_mode))
                        in_block, block_from = m_block.group('block_name'), 'parameters_data'

                m_body =  re.match(paterns['on_off'], line.strip())
                if not(m_body == None):            
                    LesHouches.find_block(in_block,block_list).block_body.append(BlockLine(list(m_body.groups()),'on_off'))
                m_body =  re.match(paterns['value'], line.strip())
                if not(m_body == None):            
                    LesHouches.find_block(in_block,block_list).block_body.append(BlockLine(list(m_body.groups()),'value'))
                m_body =  re.match(paterns['matrix_value'], line.strip())
                if not(m_body == None):            
                    LesHouches.find_block(in_block,block_list).block_body.append(BlockLine(list(m_body.groups()), 'matrix_value'))
        return block_list

  
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
    def __init__(self, entries, value, comment, line_category):
        self.entries = entries.split()
        self.value = float(value) if value != None else None 
        self.comment = comment
        self._total_entries_list = [entries] + [value] + [comment] 
        self.line_category = line_category
    
    def __repr__(self):
        if (self.line_category == 'DECAY'):
            # Aligns and Widths for Decay Blocks
            br_f  = '{:>19.8E}'
            nda_f = '{:>5}'
            id1_f = '{:>13}'
            idn_f = '{:>11}'
            comment_f = '    {}'
            line_format = br_f+nda_f+id1_f+idn_f*(len(self.entries)-2)+comment_f
            return line_format.format(self.value, *self.entries, self.comment)
        else:
            # Aligns and Widths for Parameter Blocks
            n_entries = len(self.entries)
            max_len = max([len(entry) for entry in self.entries])
            if (n_entries == 2) & (max_len <= 2):
                entries_f = '{:>5}'
            else:    
                entries_f = '{:>11}'
            value_f  = '{:>18.8E}'
            comment_f = '    {}'
            if self.value == None:
                line_format = (entries_f+' ')*len(self.entries) + comment_f
                return line_format.format(*self.entries, self.comment)
            else:
                line_format = entries_f*len(self.entries)+value_f+comment_f
                return line_format.format(*self.entries, self.value, self.comment)

class BlockSLHA:
    '''
    # Block
    It holds each line of a block.
    Call .set(parameter_number, value) to change the parameter value in the instance.
    
    Args:
    ----
    block_name
    block_comment
    q_values
    block_category: DECAY or BLOCK
    decay_width
    output_mode: Print internal processes
    '''  
    def __init__(   self, block_name, block_comment=None, q_values = None, 
                    block_category=None, decay_width=None):
        self.block_name = block_name
        self.block_comment = block_comment
        self.q_values = q_values 
        self.block_body = []
        self.block_category = block_category
        if self.block_category == 'DECAY':
            self.pid = int(self.block_name.split()[-1])
            self.decay_width = float(decay_width)
 
    def __repr__(self):
        if (self.block_category == 'DECAY'):
            block_header = 'DECAY {:>10}{:>19.8E}    {}\n'.format(self.pid, self.decay_width, self.block_comment)
            block_format = '#    BR                NDA      ID1      ID2   ... \n'
            for line in self.block_body:
                block_format += str(line) + '\n'
            return block_header+block_format
        else:
            len_name = len(self.block_name)+4
            block_header = 'BLOCK {name:>{len_name}}'.format( name=self.block_name, 
                                                              len_name=len(self.block_name)+4)
            block_header_comment = '   {comment}'.format(comment=self.block_comment)
            if not(self.q_values == None):
                block_header_q = ' Q={q_values:>16.8E}'.format(q_values=float(self.q_values))
                block_header += block_header_q + block_header_comment
                block_format = block_header+'\n'
            else:
                block_header += block_header_comment
                block_format = block_header.format(self.block_name, self.block_comment)+'\n'
            for line in self.block_body:
                block_format += str(line) + '\n'
            return block_format

        
    def set(self, line_number, param_value):
        for n, line in enumerate(self.block_body, start=1):
            if n == line_number:
                line.value = param_value


class SLHA:
    '''
    # SLHA
    Read a SLHA file (usually the param_card.dat or first section of and LHE file) and
    stores each block in BlockSLHA classes.
    
    Args:
    ----
    file_path: str = Path for the SLHA file to read
    work_dir: str = Working directory
    model: str = Name of the model 
    Atributes:
    ---------
    .block_list: List with the names of all the blocks in the SLHA file.
    Methods:
    -------
    .block(name): Call a Block object stored in the SLHA instance.  
    .new_file(new_file_name): Save the instance as a new SLHA file.
    '''
    def __init__(self, file_path: str, work_dir: str, model: str) -> None:
        self._blocks = self.read_slha(file_path)
        self.block_list = [name.block_name for name in self._blocks]
        self.work_dir = work_dir
        self.model = model
    

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

    def read_slha(self, file_dir):
        block_list = []
        paterns =   dict(   
                        block_header=\
                                r'(?P<block>BLOCK)\s+(?P<block_name>\w+)\s+((Q=.*)?(?P<q_values>-?\d+\.\d+E.\d+))?(\s+)?(?P<comment>#.*)',
                        nmatrix_value =\
                                r'(?P<entries>.+)\s+(?P<value>-?\d+\.\d+E.\d+)\s+(?P<comment>#.*)',
                        model_param_pattern =\
                                r'(?P<entries>.+)\s+(?P<comment>#.*)',
                        decay_header=\
                                r'DECAY\s+(?P<particle>\w+)\s+(?P<value>-?\d+\.\d+E.\d+)\s+(?P<comment>#.*)',
                        decay_body_pattern=\
                                r'(?P<value>.?\d+\.\d+E.\d+)\s+(?P<entries>.+)\s+(?P<comment>#.*)',
                        )



        with open(file_dir, 'r') as f:
            for line in f:
                m_block = re.match(paterns['block_header'], line.upper().strip()) 
                if not(m_block == None):
                    block_list.append(BlockSLHA(   block_name=m_block.group('block_name'), 
                                                block_comment=m_block.group('comment'),
                                                q_values = m_block.group('q_values'),
                                                block_category= 'BLOCK' ,
                                                ))
                    in_block, block_from = m_block.group('block_name'), 'parameter_data'
                    continue
                m_block = re.match(paterns['decay_header'], line.upper().strip()) 
                if not(m_block == None):
                    block_name = 'DECAY '+m_block.group('particle') 
                    block_list.append(BlockSLHA(   block_name=block_name, 
                                                block_comment=m_block.group('comment'),
                                                block_category= 'DECAY' ,
                                                decay_width= m_block.group('value'),
                                                ))
                    in_block, block_from = block_name, 'decay_data'
                    continue

                m_body =  re.match(paterns['nmatrix_value'], line.strip())
                if not(m_body == None):            
                    self.find_block(in_block,block_list).block_body.append(BlockLineSLHA(
                                                                                entries=m_body.group('entries'),
                                                                                value=m_body.group('value'),
                                                                                comment=m_body.group('comment'),
                                                                                line_category='BLOCK'))
                    continue
                m_body =  re.match(paterns['decay_body_pattern'], line.strip())
                if not(m_body == None):            
                    self.find_block(in_block,block_list).block_body.append(BlockLineSLHA(
                                                                                entries=m_body.group('entries'),
                                                                                value=m_body.group('value'),
                                                                                comment=m_body.group('comment'),
                                                                                line_category='DECAY'))
                    continue
                m_body =  re.match(paterns['model_param_pattern'], line.strip())
                if not(m_body == None):            
                    self.find_block(in_block,block_list).block_body.append(BlockLineSLHA(
                                                                                entries=m_body.group('entries'),
                                                                                value=None,
                                                                                comment=m_body.group('comment'),
                                                                                line_category='BLOCK'))
                    continue
                

        return block_list

  
    def new_file(self, new_file_name):
        '''
        Write the instance as a new SLHA file in /{work_dir}/SLHA_{model}/{new_file_name}.
        Args:
        -----
        new_file_name: str
        '''
        new_file_dir = os.path.join(self.work_dir, 'SLHA_'+self.model)

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
