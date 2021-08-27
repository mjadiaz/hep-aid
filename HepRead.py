import re 
import os



class BlockLine:
    def __init__(self, entries, line_category):
        self.entries = entries
        self.line_category = line_category
        self.line_format = BlockLine.fline(line_category)
    def fline(cat):
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
    def __init__(self, block_name, block_comment=None, category=None):
        self.block_name = block_name
        self.block_comment = block_comment
        self.block_body = []
        self.category = category

    def show(self):
        '''
        Print block information in the LesHouches format.
        '''
        print('{:6s} {:15s}  {:10s}'.format('Block',self.block_name,self.block_comment))
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
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self._blocks = LesHouches.read_leshouches(file_dir)
        self.block_list = [name.block_name for name in self._blocks]
    
    #@property
    #def block_list(self):
    #    print('Blocks in LesHouches file:')
    #    return [name.block_name for name in self._blocks]


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
                    #LesHouches.find_block(in_block,block_list).block_body.append()
        return block_list

    def new_file(self, new_file_name, new_file_dir=None):
        '''
        Writes a new LesHouches file with the blocks defined in the instance. \n
        Possibly with new values for the parameters and options.
        '''
        if new_file_dir == None:
            file_dir = new_file_name            
        else:
            if not(os.path.exists(new_file_dir)):
                os.makedirs(new_file_dir)
            file_dir=os.path.join(new_file_dir,new_file_name)
        print(f'Writing new LesHouches in :{file_dir}')
        
        with open(file_dir,'w+') as f:
            for block in self._blocks:
                head = '{:6s} {:15s}  {:10s}'.format('Block',block.block_name,block.block_comment)+'\n'
                f.write(head)
                for b in block.block_body:
                    f.write(b.line_format.format(*b.entries)+'\n')
        

