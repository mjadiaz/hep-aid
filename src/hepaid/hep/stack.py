"""
This module provides infrastructure for managing and executing High Energy Physics (HEP) software stacks, a collection of HEP tools that run sequentially.
It aims to simplify the setup and execution of complex HEP analyses by abstracting away the details of tool initialisation and execution.
It defines a base class `BaseStack` for initializing and running HEP tools such as SPheno, HiggsBounds, HiggsSignals, 
and Madgraph based on a configuration object (`hep_config`). Every HEPStack is saved into the `HEPSTACK` dictionary.

Key features include:
- A HEPSTACK dictionary that holds every implementation of a HEP Stack. Each stack can be obtained by calling HEPSTACK['StackName'].
- Dynamic registration of HEP stack implementations to the HEPSTACK dictionary via the `register` decorator.
- Support for converting input vectors to SLHA/LesHouches format for parameter updates.
- Flexible execution of HEP stacks through the `hep_stack_fn` function, allowing for single-point usage of any registered HEP stack.

Each HEP tool-specific stack implementation (e.g., `SPhenoStack`, `SPhenoHBHS`, `SPhenoHBHSMG5`) extends the `BaseStack` class, 
providing tool-specific initialization and execution logic. These classes enable the automatic modification of SLHA files, 
execution of the HEP tools, and collection of results into a standardized format.


For more detailed documentation, refer to the individual class docstrings and method annotations throughout the module.
"""
import os
import shutil
import numpy as np

from pathlib import Path
from typing import Callable, Any, Dict, List
from omegaconf import OmegaConf, DictConfig

from hepaid.hep.read import SLHA#, LesHouches
from hepaid.hep.tools import HiggsBounds, HiggsSignals
from hepaid.hep.tools import SPheno
from hepaid.hep.tools import Madgraph
from hepaid.hep.utils import id_generator
from hepaid.utils import load_config, save_config


HEPSTACK: Dict[str, Callable[..., Any]] = dict()


def register(hep_stack: Callable[..., Any]) -> Callable[..., Any]:
    HEPSTACK[hep_stack.__name__] = hep_stack
    return hep_stack

def hep_stack_fn(
    x,
    hep_config,
    close=True,
    ):
    '''
    Single point usage of a HEP Stack class defined in hep_config.hep_stack
    '''
    if isinstance(hep_config, str):
        hep_config = load_config(hep_config)

    hepsmplr = HEPSTACK[hep_config.hep_stack.name](
        hep_config=hep_config,
        )
    result = hepsmplr.sample(x)

    return result

def input_vector_to_slha(
    sample: np.ndarray, slha: SLHA, model_input: dict
) -> SLHA:
    """
    Updates a SLHA parameter blocks with new values from the sample input vector.

    Parameters:
    sample (np.ndarray): A numpy array of sampled values to be updated in the LesHouches object.
    slha (SLHA): An instance of a LesHouches class representing the LHS file to be modified.
    model_input (dict): A dictionary mapping parameter names to their respective 'block_name'
                        and 'block_index' within the LHS structure.

    Returns:
    LesHouches: The modified instance of the LesHouches class with updated parameter values.

    """
    params_iterate_ = zip(model_input.keys(), sample)
    for params in params_iterate_:
        name, value = params
        block_name = model_input[name]["block_name"]
        index = model_input[name]["block_index"]
        slha[block_name][index] = value
    return slha 


# def input_vector_to_lhs(
#     sample: np.ndarray, lhs: LesHouches, model_input: dict
# ) -> LesHouches:
#     """
#     Updates a LesHouches's parameter blocks with new values from the sample input vector.

#     This function is designed to integrate the parameter values (sample) into an LesHouches object,
#     updating specific block parameters with values provided in the `sample` array. Each parameter
#     is associated with a block name and an index within that block, as defined in `model_input`.

#     Parameters:
#     sample (np.ndarray): A numpy array of sampled values to be updated in the LesHouches object.
#     lhs (LesHouches): An instance of a LesHouches class representing the LHS file to be modified.
#     model_input (dict): A dictionary mapping parameter names to their respective 'block_name'
#                         and 'block_index' within the LHS structure.

#     Returns:
#     LesHouches: The modified instance of the LesHouches class with updated parameter values.

#     """
#     params_iterate_ = zip(model_input.keys(), sample)
#     for params in params_iterate_:
#         name, value = params
#         block_name = model_input[name]["block_name"]
#         index = model_input[name]["block_index"]
#         lhs[block_name][index] = value
#     return lhs

class BaseStack:
    """
    Base class for managing HEP Software Stacks.

    This class handles the initialization of HEP tools. Based on the hep_config, 
    the stack can sample from an array that automatically changes the SLHA file 
    and runs the HEP tools. The 'run_stack_from_input_slha' method must be 
    implemented in a subclass to be able to sample, and it must return a dictionary
    hep_stack_data point.

    Parameters:
        hep_config (DictConfig): Configuration object containing HEP settings.
        stack_id (str | None, optional): Unique identifier for the stack. 
                                        If None, an ID is auto-generated. Defaults to None.

    Attributes:
        hp (DictConfig): HEP configuration object.
        hp_input (DictConfig): Input model configuration from the HEP config.
        scan_dir (Path): Directory where scan results are stored.
        stack_id (str): Unique stack ID.
        stack_id_dir (Path): Directory specific to this stack's results.
        block (List[str]): List of block names for input parameters.
        index (List[int]): List of indices for input parameters.

    Methods:
        sample (self, parameter_point): Runs the HEPStack from an numpy array where each element corresponds
                                        to an parameter in the SLHA file.
    """
    def __init__(
        self,
        hep_config: DictConfig | str,
        stack_id: str | None = None,
    ):
        """Initialize the HEP tools"""

        
        if isinstance(hep_config, str):
            self.hp = load_config(hep_config)
        else:
            self.hp = hep_config

        self.hp_input = self.hp.model.input
        self.scan_dir = Path(self.hp.hep_stack.scan_dir)
        self.scan_dir.mkdir(parents=True, exist_ok=True)

        if stack_id is None:
            self.stack_id = id_generator()
        else:
            self.stack_id = stack_id

        self.stack_id_dir = self.scan_dir / str(self.stack_id)

        # Parameter information
        self.block = [self.hp_input[p].block_name for p in self.hp_input.keys()]
        self.index = [self.hp_input[p].block_index for p in self.hp_input.keys()]

    def __call__(self, parameter_point: np.ndarray) -> np.ndarray:
        """Allow the object to be called like a function for sampling."""
        return self.sample(parameter_point)

    def __del__(self):
        """Clean up stack directory when the object is deleted, if configured."""
        if self.hp.hep_stack.delete_on_exit:
            self.close()

    def sample(self, parameter_point: np.ndarray) -> Dict[str, float] | None:
        """
        Sample the HEP stack at the given parameter point.

        Parameters:
            parameter_point (np.ndarray): Array of parameter values.

        Returns:
            Dict[str, float] | None: Dictionary of results from the HEP stack, 
        """
        _ = input_vector_to_slha(parameter_point, self.input_slha, self.hp_input)
        hep_stack_data = self.run_stack_from_input_slha()
        return hep_stack_data

    def close(self):
        """Delete the stack directory and its contents."""
        shutil.rmtree(self.stack_id_dir)

    def run_stack_from_input_slha(self) -> dict:
        raise NotImplementedError(
            "The 'run_stack_from_input_slha' method must be implemented in a subclass."
            ) 

@register
class SPhenoStack(BaseStack):
    """
    SPheno Stack.

    This class handles the initialization of SPheno. Based on the hep_config, 
    the stack can sample from an array that automatically changes the input SLHA file 
    and runs the HEPStack. The 'run_stack_from_input_slha' method is implemented 
    to run SPheno and return a dictionary containing the output SLHA. 

    Parameters:
        hep_config (DictConfig): Configuration object containing HEP settings.
        stack_id (str | None, optional): Unique identifier for the stack. 
                                        If None, an ID is auto-generated. Defaults to None.

    Attributes:
        hp (DictConfig): HEP configuration object.
        hp_input (DictConfig): Input model configuration from the HEP config.
        scan_dir (Path): Directory where scan results are stored.
        stack_id (str): Unique stack ID.
        stack_id_dir (Path): Directory specific to this stack's results.
        block (List[str]): List of block names for input parameters.
        index (List[int]): List of indices for input parameters.

    Methods:
        sample(self, parameter_point): Runs the HEPStack from an numpy array where each element corresponds
                                        to an parameter in the SLHA file.
        run_stack_from_input_slha(self): This method executes the HEP Stack processing pipeline with the SLHA file specified in the `self.input_slha` attribute.  
    """
    def __init__(
        self,
        hep_config: DictConfig | str,
        stack_id: str | None = None, 
    ):
        super().__init__(hep_config, stack_id)

        self.spheno = SPheno(
            heptool_dir=self.hp.spheno.directory,
            output_dir=self.stack_id_dir,
            model_name=self.hp.spheno.model,
        )

        self.input_slha = SLHA(
            file=self.hp.spheno.reference_slha,
        )

    

    def run_stack_from_input_slha(self):
        """Runs the HEP stack using the input SLHA file.

        This method executes the HEP Stack processing pipeline with the SLHA file specified in the `self.input_slha` attribute. 

        Returns:
            dict: A dictionary containing the HEP stack data, with keys:
                - "LHS": The input SLHA file as a dictionary (converted using `as_dict()`).
                - "SLHA": The output SLHA file as a dictionary (or None if no output was produced).
        """
        
        self.spheno.run(self.input_slha)
        self.output_slha = self.spheno.results
        if self.output_slha is not None:
            hep_stack_data = {
                "LHS": self.input_slha.as_dict(),
                "SLHA": self.output_slha.as_dict(),
            }
        else:
            hep_stack_data = {
                "LHS": self.input_slha.as_dict(),
                "SLHA": None,
            }
        return hep_stack_data


    
@register
class SPhenoHBHS(SPhenoStack):
    """
    SPheno-HiggsBounds-HiggsSignals Stack.

    This class handles the initialization of SPheno, HiggsBounds and HiggsSignals. Based on the hep_config, 
    the stack can sample from an array that automatically changes the input SLHA file 
    and runs the HEPStack. The 'run_stack_from_input_slha' method is implemented 
    to run SPheno and return a dictionary containing the output SLHA. 

    Parameters:
        hep_config (DictConfig): Configuration object containing HEP settings.
        stack_id (str | None, optional): Unique identifier for the stack. 
                                        If None, an ID is auto-generated. Defaults to None.

    Attributes:
        hp (DictConfig): HEP configuration object.
        hp_input (DictConfig): Input model configuration from the HEP config.
        scan_dir (Path): Directory where scan results are stored.
        stack_id (str): Unique stack ID.
        stack_id_dir (Path): Directory specific to this stack's results.
        block (List[str]): List of block names for input parameters.
        index (List[int]): List of indices for input parameters.

    Methods:
        sample(self, parameter_point): Runs the HEPStack from an numpy array where each element corresponds
                                        to an parameter in the SLHA file.
        run_stack_from_input_slha(self): This method executes the HEP Stack processing pipeline with the SLHA file specified in the `self.input_slha` attribute.  
    """
    def __init__(
        self,
        hep_config: DictConfig | str,
        stack_id: str | None = None,
    ):
        super().__init__(hep_config, stack_id)

        self.higgs_bounds = HiggsBounds(
            heptool_dir=self.hp.higgsbounds.directory,
            output_dir=self.stack_id_dir,
            neutral_higgs=self.hp.higgsbounds.neutral_higgs,
            charged_higgs=self.hp.higgsbounds.charged_higgs,
        )
        self.higgs_signals = HiggsSignals(
            heptool_dir=self.hp.higgssignals.directory,
            output_dir=self.stack_id_dir,
            neutral_higgs=self.hp.higgssignals.neutral_higgs,
            charged_higgs=self.hp.higgssignals.charged_higgs,
        )

    def run_stack_from_input_slha(self):
        """Runs the HEP stack using the input SLHA file.

        This method executes the HEP Stack processing pipeline with the SLHA file specified in the `self.input_slha` attribute. 

        Returns:
            dict: A dictionary containing the HEP stack data, with keys:
                - "LHS": The input SLHA file as a dictionary (converted using `as_dict()`).
                - "SLHA": The output SLHA file as a dictionary (or None if no output was produced).
                - "HB": The output from HiggsBounds as a dictionary (or None if no output was produced).
                - "HS": The output from HiggsSignals as a dictionary (or None if no output was produced).
        """
        
        self.spheno.run(self.input_slha)
        self.output_slha = self.spheno.results
        if self.output_slha is not None:
            self.higgs_bounds.run()
            self.higgs_signals.run()

            hep_stack_data = {
                "LHS": self.input_slha.as_dict(),
                "SLHA": self.output_slha.as_dict(),
                "HB": self.higgs_bounds.results,
                "HS": self.higgs_signals.results
            }
        else:
            hep_stack_data = {
                "LHS": self.input_slha.as_dict(),
                "SLHA": None,
                "HB": None,
                "HS": None
            }
        return hep_stack_data

@register
class SPhenoHBHSMG5(SPhenoHBHS):
    """
    SPheno-HiggsBounds-HiggsSignals-MagGraph Stack.

    This class handles the initialization of SPheno, HiggsBounds, HiggsSignals and Magraph. 
    Based on the hep_config, the stack can sample from an array that automatically changes the input SLHA file 
    and runs the HEPStack. The 'run_stack_from_input_slha' method is implemented 
    to run SPheno and return a dictionary containing the output SLHA. 

    Parameters:
        hep_config (DictConfig): Configuration object containing HEP settings.
        stack_id (str | None, optional): Unique identifier for the stack. 
                                        If None, an ID is auto-generated. Defaults to None.

    Attributes:
        hp (DictConfig): HEP configuration object.
        hp_input (DictConfig): Input model configuration from the HEP config.
        scan_dir (Path): Directory where scan results are stored.
        stack_id (str): Unique stack ID.
        stack_id_dir (Path): Directory specific to this stack's results.
        block (List[str]): List of block names for input parameters.
        index (List[int]): List of indices for input parameters.

    Methods:
        sample(self, parameter_point): Runs the HEPStack from an numpy array where each element corresponds
                                        to an parameter in the SLHA file.
        run_stack_from_input_slha(self): This method executes the HEP Stack processing pipeline with the SLHA file specified in the `self.input_slha` attribute.  
        update_mg5_script(self, process, mg5_script): Takes the template madgraph script and updated with current directories
                                                        to be consistent with the HEPSTACK.
    """

    def __init__(
        self,
        hep_config: DictConfig | str,
        stack_id: str | None = None,
    ):
        super().__init__(hep_config, stack_id)
        self.mg5_script = self.hp.madgraph.scripts
        self.mg5_output_files = self.stack_id_dir / "mg5"
        self.mg5_output_files.mkdir(parents=True, exist_ok=True)

        self.madgraph = Madgraph(
            heptool_dir=self.hp.madgraph.directory, output_dir=self.stack_id_dir
        )

    def update_mg5_script(self, process, mg5_script):
        """
        Takes the template madgraph script and updated with current directories
        to be consistent with the HEPSTACK.

        Parameters:
            process: name for the madgraph process.
            mg5_script: Path to the madgraph script template.
        """
        with open(mg5_script, "r") as file:
            content = file.read()
        # Replace WORK_DIR variable
        process_output_path = self.stack_id_dir / "mg5" / process
        process_output_path.mkdir(parents=True, exist_ok=True)
        content = content.replace("WORK_DIR", str(process_output_path))
        # Replace PARAM_CARD_DIR variable
        content = content.replace("PARAM_CARD_DIR", str(self.spheno.output_file_path))
        new_mg5_script_path = self.mg5_output_files / f'{process}.txt'

        with open(new_mg5_script_path, "w+") as file:
            file.write(content)
        return new_mg5_script_path, process_output_path

    def run_stack_from_input_slha(self):
        """Runs the HEP stack using the input SLHA file.

        This method executes the HEP Stack processing pipeline with the SLHA file specified in the `self.input_slha` attribute. 

        Returns:
            dict: A dictionary containing the HEP stack data, with keys:
                - "LHS": The input SLHA file as a dictionary (converted using `as_dict()`).
                - "SLHA": The output SLHA file as a dictionary (or None if no output was produced).
                - "HB": The output from HiggsBounds as a dictionary (or None if no output was produced).
                - "HS": The output from HiggsSignals as a dictionary (or None if no output was produced).
                - "MG5": The output from Madgraph as a dictionary (or None if no output was produced).
        """
        
        self.spheno.run(self.input_slha)
        self.output_slha = self.spheno.results
        if self.output_slha is not None:
            self.higgs_bounds.run()
            self.higgs_signals.run()

            # Configure SLHA so that the SPheno output is compatible with Madgraph
            self.input_slha["SPHENOINPUT"][520] = 0  # Effective couplings
            self.input_slha["SPHENOINPUT"][78] = 1  # Write madgraph
            self.input_slha["SPHENOINPUT"][525] = 0  # Contribution to diphoton
            self.input_slha["SPHENOINPUT"][16] = 0  # One loop decays

            self.spheno.run(self.input_slha)
            self.output_slha = self.spheno.results

            mg5_result = {}
            for process in self.mg5_script.keys():
                new_mg5_script_path, process_output_path = self.update_mg5_script(
                    process=process, mg5_script=self.mg5_script[process]
                )
                self.madgraph.run(input_file=new_mg5_script_path)
                mg5_result[process] = self.madgraph.results(process_output_path)

            hep_stack_data = {
                "LHS": self.input_slha.as_dict(),
                "SLHA": self.output_slha.as_dict(),
                "HB": self.higgs_bounds.results,
                "HS": self.higgs_signals.results,
                "MG5": mg5_result,
            }
        else:
            mg5_result = {}
            for process in self.mg5_script.keys():
                mg5_result[process] = None

            hep_stack_data = {
                "LHS": self.input_slha.as_dict(),
                "SLHA": None,
                "HB": None,
                "HS": None,
                "MG5": mg5_result,
            }
        return hep_stack_data


