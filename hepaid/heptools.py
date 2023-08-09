import os 
import re
import subprocess

import numpy as np

from hepaid.hepread import SLHA

class Spheno:
    def __init__(self, spheno_dir, work_dir, model_name):
        self._spheno_dir = spheno_dir
        self._work_dir = work_dir
        self._model_name = model_name

    def run(self,in_file_name,out_file_name):
        '''
        Run SPhenoMODEL with the input_file_name in work_dir. 

        Args:
        -----
        in_file_name: str = input file's name located in work_dir/SPhenoMODEL_input/in_file_name
        out_file_name: str = output file's name located in work_dir/SPhenoMODEL_output/out_file_name

        Return:
        ------
        out_file: str = Global path to the output file name if SPheno runs successfully. None if SPheno gets an Error.
        spheno_stdout = stdout of the subprocess that runs SPheno.
        '''
        # Reads the input file created with LesHouches.new_file in work_dir
        in_file = os.path.join(self._work_dir, 'SPheno'+self._model_name+'_input',in_file_name)
        # Create and output directory in work_dir/SPhenoMODEL_output
        out_dir = os.path.join(self._work_dir, 'SPheno'+self._model_name+'_output')		
        if not(os.path.exists(out_dir)):
            os.makedirs(out_dir)
        out_file=os.path.join(out_dir, out_file_name)

        run = subprocess.run(
                [self._spheno_dir+'/bin'+'/SPheno'+self._model_name, in_file, out_file],
                capture_output=True,
                text=True,
                cwd=self._work_dir
                )
        if 'Finished' in run.stdout:
            return out_file, run.stdout
        else:
            return None, run.stdout



class Madgraph:
    '''
    Basicaly just run madgraph with in the script mode with the scripts created \n
    by the MG5Script class. \n
    Todo:
    - Figure out how to print some output from madgraph.
    '''
    def __init__(self, madgraph_dir, work_dir):
        self._dir = madgraph_dir
        self._work_dir = work_dir       

    def run(self, input_file = 'MG5Script.txt', mode='local'):
        '''
        Run madgraph with an script named MG5Script.txt (created by the MG5Script class) in within work_dir. \n
        Change input_file to change to another script within work_dir.
        '''
        if mode == 'local':
            subprocess.run(
                    [os.path.join(self._dir,'bin/mg5_aMC'), input_file],
                    text=True,
                    cwd=self._work_dir
                    )
        elif mode == 'cluster':
            print('Implement cluster mode.')

class HiggsBounds:
    def __init__(self, higgs_bounds_dir, work_dir, model=None, neutral_higgs=None, charged_higgs=None, output_mode=False):
        self._dir = higgs_bounds_dir
        self.work_dir = work_dir
        self.model = model
        self.neutral_higgs = neutral_higgs
        self.charged_higgs = charged_higgs
        self.output_mode = output_mode


    def run(self):
        '''
        Runs HiggsBounds with the last point calculated by SPheno. 
        Returns the path for HiggsBounds_results.dat if SPheno and 
        HiggsBounds succeed. If there is an Error in one them it 
        returns None and prints the error output.
        '''
        run = subprocess.run(
                [os.path.join(self._dir, 'HiggsBounds'), 
                 'LandH', 
                 'effC', 
                 str(self.neutral_higgs), 
                 str(self.charged_higgs), 
                 self.work_dir+'/'], 
                capture_output=True,  
                text=True,
                cwd=self.work_dir)        
        if 'finished' in run.stdout:
            if self.output_mode:
                print(run.stdout) 
            return  os.path.join(self.work_dir, 'HiggsBounds_results.dat')            
        else:
            if self.output_mode:
                print('HiggsBound not finished!')
                print(run.stdout)
            return None

class HiggsSignals:
    def __init__(
            self, 
            higgs_signals_dir, 
            work_dir, model=None, 
            neutral_higgs=None, 
            charged_higgs=None, 
            output_mode=False):
        self._dir = higgs_signals_dir
        self.work_dir = work_dir
        self.model = model
        self.neutral_higgs = neutral_higgs
        self.charged_higgs = charged_higgs
        self.output_mode = output_mode


    def run(self):
        '''
        Runs HiggsSignals with the last point calculated by SPheno.
        '''
        run = subprocess.run(
                [os.path.join(self._dir, 'HiggsSignals'), 
                 'latestresults', 
                 '2', 
                 'effC', 
                 str(self.neutral_higgs), 
                 str(self.charged_higgs), 
                 self.work_dir+'/'], 
                capture_output=True,  
                text=True,
                cwd=self.work_dir)        
        if not('Error') in run.stdout:
            if self.output_mode:
                print(run.stdout) 
            return  os.path.join(self.work_dir, 'HiggsSignals_results.dat')            
        else:
            if self.output_mode:
                print('HiggsSignals Error')
                print(run.stdout)
            return None

class THDMC:
    '''
    Utility class to run 2HDMC programs. 
    '''
    def __init__(
        self,
        tool_dir: str,
        work_dir: str,
        program_name: str,
    ):
        self.tool_dir = tool_dir 
        self.work_dir = work_dir
        self.program_name = program_name
        self.program = os.path.join(tool_dir,program_name)
        self.thdm_dir = os.path.join(self.work_dir, '2HDMC')
    
    def run(
        self,
        parameters: np.ndarray,
        ) -> bool:
        '''
        Parameters are in the standard basis:
        lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7,\
            m12_2, tan_beta = parameters
        '''

        if not(os.path.exists(self.thdm_dir)):
            os.makedirs(self.thdm_dir)

        parameters = [str(val) for val in parameters]

        #lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7,\
        #    m12_2, tan_beta = parameters
        try: 
            run = subprocess.run([
                self.program, 
                *parameters,
                #lambda1,
                #lambda2,
                #lambda3,
                #lambda4,
                #lambda5,
                #lambda6,
                #lambda7,
                #m12_2,
                #tan_beta
                ],
                capture_output=True,
                text=True,
                cwd=self.thdm_dir ,
                )
            return True
        except:
            return False
    
    def result(self) -> SLHA:
        '''Returns the SLHA file saved in working directory, after .run()'''
        file=os.path.join(self.thdm_dir, 'Demo_out.lha')
        if os.path.exists(file):
            slha = SLHA(
                file=file
                )
            return slha
        else:
            return None
        