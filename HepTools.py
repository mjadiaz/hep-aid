
import os 
import re
import subprocess

from scipy.stats import loguniform
import matplotlib.pyplot as plt
import numpy as np


import HepRead



class Spheno:
    def __init__(self, spheno_dir, work_dir, model=None, input_lhs=None):
        self._dir=spheno_dir
        self.model_list = Spheno._models_in_dir(self)
        self._model = Spheno._model_init(self, model)
        self.work_dir = work_dir
        #self.lhs = Spheno._read_input_leshouches(input_lhs)

    def _model_init(self, model):
        if model == None:
            if len(self.model_list) == 1:
                return self.model_list[0]
            else:
                return print('Define model. \nModels available: {}'.format(self.model_list))
        else:
            print(f'{model} model activated.')
            return model

    #def _read_input_leshouches(input_lhs):
    #    if input_lhs==None:
    #        print('Insert the path for the Leshouches.in.Model file as input_lhs')
    #    else:
    #        return HepRead.LesHouches(input_lhs)
            
    def _models_in_dir(self):
        models_in_dir = []
        for f in os.listdir(self._dir+'/bin'):
            if not (re.search(r'SPheno\w+',f)==None):
                m = re.search('(?<=SPheno).*',f)[0]
                models_in_dir.append(m)
        return models_in_dir

    #def run(self, in_file, out_file_name, out_dir=None):
    def run(self, in_file_name, out_file_name):
        out_dir = os.path.join(self.work_dir, 'SPheno'+self._model+'_output')
        in_file = os.path.join(self.work_dir, 'SPheno'+self._model+'_input',in_file_name)

        #if out_dir == None:
        #    file_dir = out_file_name            
        #else:
        if not(os.path.exists(out_dir)):
            os.makedirs(out_dir)
        file_dir=os.path.join(out_dir,out_file_name)

        print(f'Save {out_file_name} in :{file_dir}')
        
        run = subprocess.run([self._dir+'/bin'+'/SPheno'+self._model, in_file, file_dir], capture_output=True,  text=True)        
        if 'Finished' in run.stdout:
            print(run.stdout)                
        else:
            print('Parameer Error, check this!')

    #def scan


# In[3]:


class Scanner:
    
    
    def rlog_array(min, max, n_points, show=False):
        '''
        Creates a random array distributed uniformly in log scale. \n
        - from min to max with n_points number of points. \n
        - turn show=True to see the histogram of the distribution.
        '''
        ar = loguniform.rvs(min, max, size=n_points)

        if show == True:
            plt.hist(np.log10(ar),density=True, histtype='step',color='orange')
            plt.hist(np.log10(ar),density=True, histtype='stepfilled', alpha=0.2, color='orange')
            plt.xlabel('Variable [log]')
            plt.show()
        
        return ar
    def rlog_float(min, max):
        '''
        Creates a random float distributed uniformly in log scale.
        '''
        ar = loguniform.rvs(min, max, size=1)
        return ar[0]


    def runiform_array(min,max, n_points, show=False):
        '''
        Create a uniform random array from min to mas with n_points.
        - show= True to see the distribution histogram.
        '''
        ar=np.random.uniform(min, max ,size = n_points)

        if show == True:
            plt.hist(ar,density=True, histtype='step',color='orange')
            plt.hist(ar,density=True, histtype='stepfilled', alpha=0.2, color='orange')
            plt.xlabel('Variable')
            plt.show()
        
        return ar

    def runiform_float(min,max):
        '''
        Create a uniform random float within min and max.
        '''
        ar=np.random.uniform(min, max ,size = 1)        
        return ar[0]

    
