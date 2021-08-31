import re 
import os


import awkward as ak
import numpy as np
from tabulate import tabulate


# Todo:
# - Explain LesHouches.

#########################################
# Classes for reading LesHouches files. #
# Focusing on Spheno.                   #
#########################################

class BlockLine:
    '''## Line:
        Each line can be of a different category:
        - block_header
        - on_off
        - value
        - matrix_value '''
    def __init__(self, entries, line_category):
        self.entries = entries
        self.line_category = line_category
        self.line_format = BlockLine.fline(line_category)
    def fline(cat):
        ''' Text format of each line'''
        if cat == 'block_header':
            line_format = '{:6s} {:20s}  {:13s}'
        elif cat == 'on_off':
            line_format = '{:6s} {:18s}  {:13s}'
        elif cat == 'value':
            line_format = '{:6s} {:18s}  {:13s}'
        elif cat == 'matrix_value':
            line_format = '{:3s}{:3s} {:18s}  {:13s}'
        return line_format


class Block:
    '''
    ## Block
    It holds each line of a block.\n
    Call .show() to print a block. \n
    Call .set(parameter_number, value) to change the parameter value in the instance.
    '''
    def __init__(self, block_name, block_comment=None, category=None):
        self.block_name = block_name
        self.block_comment = block_comment
        self.block_body = []
        self.category = category

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
                    print('{} setted to : {}'.format(line.entries[-1], line.entries[1]))
                    break
            if (line.line_category == 'value') & (option == int(line.entries[0])):              
                line.entries[1] = '{:E}'.format(param_value) 
                print('{} setted to : {}'.format(line.entries[-1], line.entries[1])) 
                break                  
            elif (line.line_category == 'on_off') & (option == int(line.entries[0])):
                if isinstance(param_value, int):
                    line.entries[1] = '{}'.format(param_value)
                    print('{} setted to : {}'.format(line.entries[-1], line.entries[1]))
                else:
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
    def __init__(self, file_dir, work_dir, model):
        self.file_dir = file_dir
        print(f'Reading LesHouches from : {file_dir}')
        self._blocks = LesHouches.read_leshouches(file_dir)
        self.block_list = [name.block_name for name in self._blocks]
        self.work_dir = work_dir
        self.model = model
    

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

    def read_leshouches(file_dir):
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
                                                    category= 'spheno_data' ))
                        in_block, block_from = m_block.group('block_name'), 'spheno_data'

                    else:
                        block_list.append(Block(        block_name=m_block.group('block_name'), 
                                                        block_comment=m_block.group('comment'),
                                                        category= 'parameters_data'))
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
        print(f'Writing new LesHouches in :{file_dir}')
        
        with open(file_dir,'w+') as f:
            for block in self._blocks:
                head = '{} {}  {:10s}'.format('Block',block.block_name,block.block_comment)+'\n'
                f.write(head)
                for b in block.block_body:
                    f.write(b.line_format.format(*b.entries)+'\n')

################################### 
# Classes for reading SLHA files. #  
# Focusing on madgraph.           #
###################################

class Particle():
    def __init__(self, pid, ufo_name=None, total_width=None, mass=None, comment=None, decays=None):
        self.pid = pid
        self._ufo_name = ufo_name  # To link it with madgraph but maybe it is not necessary.
        self.total_width = total_width
        self.mass = mass
        self.comment = comment 

        # How to store the decays
        self.decays = decays
    
    def find_particle(id, particles_list):
        try:
            if isinstance(id, int):        
                for p in particles_list:
                    if p.pid == id:
                        p_found = p
                        break
                    else:
                        None
            elif isinstance(id, str):
                for p in particles_list:
                    if p.comment == id:
                        p_found = p
                        break
                    else:
                        None
            return p_found
        except:
            print('No particle found')

    def show(self):
        #particle = Particle.find_particle(pid, particles_list)
        n_data = lambda n: [self.decays[n][i] for i in self.decays.fields]
        n_decays = ak.num(self.decays,axis=0)
        data = [n_data(n) for n in range(n_decays)]
        headers=[field.upper() for field in self.decays.fields]
        print(tabulate([[self.mass,self.total_width]], headers=['MASS GeV', "TOTAL WIDTH GeV"],tablefmt="fancy_grid"))
        print(tabulate(data, headers=headers,tablefmt="fancy_grid"))


class Slha:
    '''
    ## Read SLHA
    Read the given model file in SLHA format. It stores PID of all the particles of the models in .model_particles \n
    in a list of tuples (PID, NAME) to have a sense of what particles are in the model. \n
    The internal list _particles contain each particle saved as a Particle class so that we can use \n
    all the internal methods and properties of this class, for example: 
    - .particle('25').show() ---> It display all the information about the particle.
    - .particle('25').mass    
    \n
    This class focuses on extracting information not writing the file because we always can \n
    change the parameters inside madgraph by the command: set parameter = value.

    '''
    def __init__(self, model_file):
        self.model_file = model_file
        self._particles, self.model_particles = Slha.read_file(model_file)
    
    def read_file(model_file):
        '''
        Read all the particles of the model saving it in the particles object list. \n
        It creates a Particle class for each particle of the model reading it \n
        from the MASS block of the SLHA file.
        '''
        with open(model_file, 'r') as f:
            slha_file = f.read() # Read the whole file
            slha_blocks = re.split('BLOCK|Block|block|DECAY|Decay|decay', slha_file) #  Separate the text file into parameter blocks and decay blocks
            
            particles = []      # Initiate the Partcle objects list.
            

            for block in slha_blocks:
                block = re.split('\n',block)    # Split block in lines
                # Create the particles of the model from the MASS block and populate the
                # particles list with Particles objects.
                if "MASS" in block[0].split()[0].upper():
                    for line in block:
                        line = re.split('\s+',line.strip()) 
                        #print(l)
                        if line[0].isnumeric():                            #  To skip to the first particle information, since the first element is the PID (numeric)
                            particles.append(Particle(int(line[0])))       #  Append to particles list a Particle object with the PID given by l[0]
                            particles[-1].mass = float(line[1])            #  Assign the mass for the given Particle object
                            particles[-1].comment = line[3]                #  Assign the ufo_name wich is the comment (last element of the line).
                                                                            #  Note: Not always is the ufo_name.
                            #print(particles[-1].pid, particles[-1].mass, particles[-1].comment) # Just to check
                    break                

            # Another foor loop because we want to create the particles first            
            for block in slha_blocks:
                block = re.split('\n',block)    # Split block in lines again            

                # If the first element on the first line is numeric correspond to decay 
                # block with that ID value else is a parameter block.

                if block[0].split()[0].isnumeric(): 
                    tmp_pid = int(block[0].split()[0]) # Temporal particle id for a decay block
                    decays = ak.ArrayBuilder() #  Initiate the decays as an awkward array builder for the current block

                    for line in block: 


                        # Match in 3 groups a pattern like: '1000001     2.00706278E+02   # Sd_1'
                        match = re.match(r'(?P<pid>\d+)\s+(?P<width>\d+\.\d+E.\d+)\s+#(?P<comment>.*)', line.strip()) 
                        if not(match == None):
                            #print('PID: {}, Width: {:E}, Comment: {}'.format(int(match.group('pid')),float(match.group('width')) ,match.group("comment").strip()))
                            Particle.find_particle(tmp_pid,particles).total_width = float(match.group('width'))
                            #print(Particle.find_particle(tmp_pid,particles).pid, Particle.find_particle(tmp_pid,particles).total_width)
                        
                        # Match in 4 groups a pattern like: '2.36724485E-01    2            2   -1000024   # BR(Sd_1 -> Fu_1 Cha_1 )'
                        match = re.match(r'(?P<br>\d+\.\d+E.\d+)\s+(?P<nda>\d+)\s+(?P<fps>.*?)\s+#(?P<comment>.*)', line.strip())

                        if not(match == None):
                            tmp_decay = ak.Array([{ "br": float(match.group('br')),
                                                    "nda": int(match.group('nda')), 
                                                    "fps": [int(p) for p in match.group('fps').split()], 
                                                    "comment": match.group('comment').strip()}])
                            decays.append(tmp_decay, at=0)

                        #print(tmp_decay)                    
                    Particle.find_particle(tmp_pid,particles).decays = decays.snapshot()

        def particle_ids(particles_list):
            ''' Returns a simple list holding all the particle ID of the model'''
            pid_list=[(p.pid, p.comment) for p in particles_list]
            #comment_list = [p.comment for p in particles_list]
            #pid_list.sort()
            return pid_list

        particles_list = particle_ids(particles) 

        return particles, particles_list



    def particle(self, pid):
        particle = Particle.find_particle(pid, self._particles)
        return particle

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
        
