
import os 
import re
import subprocess

from scipy.stats import loguniform
import matplotlib.pyplot as plt
import numpy as np


import HepRead



class Spheno:
    '''
    ## Spheno class
    -   Identifies what models are available in the work direction. \n 
        and store all the models in .model_list  \n
        
    -   Call .run(in, out) to run Spheno with input and output files. \n
        The output file will be stored in a folder called SPhenoModel_output \n
        inside work_dir.  
    Todo:
    - Figure out how to redirect the additional files that spheno creates in a run.
    '''
    def __init__(self, spheno_dir, work_dir, model=None, input_lhs=None):
        self._dir=spheno_dir
        self.model_list = Spheno._models_in_dir(self)
        self._model = Spheno._model_init(self, model)
        self.work_dir = work_dir
  

    def _model_init(self, model):
        if model == None:
            if len(self.model_list) == 1:
                return self.model_list[0]
            else:
                return print('Define model. \nModels available: {}'.format(self.model_list))
        else:
            print(f'{model} model activated.')
            return model


            
    def _models_in_dir(self):
        models_in_dir = []
        for f in os.listdir(self._dir+'/bin'):
            if not (re.search(r'SPheno\w+',f)==None):
                m = re.search('(?<=SPheno).*',f)[0]
                models_in_dir.append(m)
        return models_in_dir

    def run(self, in_file_name, out_file_name, mode='local'):
        
        out_dir = os.path.join(self.work_dir, 'SPheno'+self._model+'_output')
        in_file = os.path.join(self.work_dir, 'SPheno'+self._model+'_input',in_file_name)

        if not(os.path.exists(out_dir)):
            os.makedirs(out_dir)
        file_dir=os.path.join(out_dir,out_file_name)

        print(f'Save {out_file_name} in :{file_dir}')
        if mode == 'local':
            run = subprocess.run([self._dir+'/bin'+'/SPheno'+self._model, in_file, file_dir], capture_output=True,  text=True)        
            if 'Finished' in run.stdout:
                print(run.stdout) 
                return  file_dir            
            else:
                print('Parameer Error, check this!')
                return None
        elif mode == 'cluster':
            print('Implement cluster mode')




class Scanner:
    '''
    Class containing useful tools to do scans with SPheno and Madgraph.
    Maybe it will a bunch of functions.
    To do: 
    - Implement general scanner function (for loops)
    - How to store these functions Class or what...
    '''
    
    
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


class Madgraph:
    '''
    Basicaly just run madgraph with in the script mode with the scripts created \n
    by the MG5Script class. \n
    Todo:
    - Figure out how to print some output from madgraph.
    '''
    def __init__(self, madgraph_dir, work_dir):
        self._dir = madgraph_dir
        self.work_dir = work_dir       
        
    def run(self, input_file = 'MG5Script.txt', mode='local'):
        '''
        Run madgraph with an script named MG5Script.txt (created by the MG5Script class) in within work_dir. \n
        Change input_file to change to another script within work_dir.
        '''
        if mode == 'local':
            subprocess.run([os.path.join(self._dir,'bin/mg5_aMC'), os.path.join(self.work_dir,input_file)])    
        elif mode == 'cluster':
            print('Implement cluster mode.')