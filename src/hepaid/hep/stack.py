import os
import shutil
import numpy as np

from typing import Callable, Any, Dict, List
from omegaconf import OmegaConf, DictConfig

from hepaid.hep.read import SLHA, LesHouches
from hepaid.hep.read import HiggsBoundsResults, HiggsSignalsResults
from hepaid.hep.read import read_mg_generation_info
from hepaid.hep.tools import Spheno, HiggsBounds, HiggsSignals
from hepaid.hep.tools import Madgraph
from hepaid.hep.tools import THDMC


HEPSTACK: Dict[str, Callable[..., Any]] = dict()


def register(hep_stack: Callable[..., Any]) -> Callable[..., Any]:
    HEPSTACK[hep_stack.__name__] = hep_stack
    return hep_stack


def input_vector_to_lhs(
    sample: np.ndarray, lhs: LesHouches, model_input: dict
) -> LesHouches:
    """
    Updates a LesHouches's parameter blocks with new values from the sample input vector.

    This function is designed to integrate the parameter values (sample) into an LesHouches object,
    updating specific block parameters with values provided in the `sample` array. Each parameter
    is associated with a block name and an index within that block, as defined in `model_input`.

    Parameters:
    sample (np.ndarray): A numpy array of sampled values to be updated in the LesHouches object.
    lhs (LesHouches): An instance of a LesHouches class representing the LHS file to be modified.
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
        lhs[block_name][index] = value
    return lhs


@register
class SPheno:
    """
    SPheno class handles the manipulation and execution of the SPheno
    tool, managing input operations for Les Houches files, and
    output operations of SLHA/Spc files.

    Attributes:
        hp (DictConfig): Saves the `hep_config` configuration object.
        hp_input (dict): Stores `hep_config.model.input` for easier manipulation.
        scan_dir (str): Stores `hep_config.directories.scan_dir` for easier manipulation.
        sampler_id_dir (str): Directory path specific to the sampler ID.
        block (list): List of block names from model input configuration.
        index (list): List of parameter indices corresponding to each block in 'block'.
        current_lhs (str): Filename for the current Les Houches input file.
        current_spc (str): Filename for the current SPheno spectrum output file.
        spheno (Spheno): SPheno execution object initialized with paths and model details.
        lhs (LesHouches): Initialises a LesHouches object from a reference lhs
                            in `hep_config.directories.reference_lhs`.

    Methods:
        create_dir(run_name: str): Creates directory for the specified run.
        get_lhs(): Returns the Les Houches file handler.
        run_stack_from_lhs(): Executes the SPheno tool with the current LHS file and gathers outputs.
        sample(parameter_point: np.ndarray): Updates LHS with new parameters, runs simulations, and returns outputs.
        __call__(parameter_point: np.ndarray): Callable method to simplify sampling operations.
        close(): Cleans up by removing the sampler ID directory.
    """

    def __init__(
        self,
        sampler_id: int,
        hep_config: DictConfig,
    ):
        """Initialize the HEP tools"""
        self.hp = hep_config
        self.hp_input = hep_config.model.input
        self.scan_dir = self.hp.directories.scan_dir
        self.sampler_id_dir = os.path.join(self.scan_dir, str(sampler_id))

        # Parameter information
        self.block = [self.hp_input[p].block_name for p in self.hp_input.keys()]
        self.index = [self.hp_input[p].block_index for p in self.hp_input.keys()]

        self.current_lhs = "LesHouches.in.Step"
        # self.current_spc = 'Spheno.spc.Step'
        self.current_spc = "SPheno.spc.Step"

        # self.space = Space(self.hp)

        self.spheno = Spheno(
            spheno_dir=self.hp.directories.spheno,
            work_dir=self.sampler_id_dir,
            model_name=self.hp.model.name,
        )

        self.lhs = LesHouches(
            file_dir=self.hp.directories.reference_lhs,
            work_dir=self.sampler_id_dir,
            model=self.hp.model.name,
        )

    def create_dir(self, run_name: str):
        if not (os.path.exists(self.sampler_id_dir)):
            os.makedirs(self.sampler_id_dir)

    def get_lhs(self):
        return self.lhs

    def run_stack_from_lhs(self):
        """
        Run the hep_stack with the LHS file saved in the
        self.lhs property.
        """
        param_card = None
        # Create new lhs file with the parameter values
        self.lhs.new_file(self.current_lhs)
        # Run spheno with the new lhs file
        param_card, spheno_stdout = self.spheno.run(self.current_lhs, self.current_spc)
        if param_card is not None:
            slha = SLHA(param_card, self.sampler_id_dir, model=self.hp.model.name)

            hep_stack_data = {
                "LHE": self.lhs.as_dict(),
                "SLHA": slha.as_dict(),
            }
        else:
            hep_stack_data = {
                "LHE": self.lhs.as_dict(),
                "SLHA": None,
            }
            return hep_stack_data

    def sample(self, parameter_point: np.ndarray) -> Dict[str, float] | None:

        _ = input_vector_to_lhs(parameter_point, self.lhs, self.hp_input)

        hep_stack_data = self.run_stack_from_lhs()

        return hep_stack_data

    def __call__(self, parameter_point: np.ndarray) -> np.ndarray:
        return self.sample(parameter_point)

    def close(self):
        shutil.rmtree(self.sampler_id_dir)


@register
class SPhenoHbHs:
    def __init__(
        self,
        sampler_id: int,
        hep_config: DictConfig,
    ):
        """Initialize the HEP tools"""
        self.hp = hep_config
        self.hp_input = hep_config.model.input
        self.scan_dir = self.hp.directories.scan_dir
        self.sampler_id_dir = os.path.join(self.scan_dir, str(sampler_id))

        # Parameter information
        self.block = [self.hp_input[p].block_name for p in self.hp_input.keys()]
        self.index = [self.hp_input[p].block_index for p in self.hp_input.keys()]

        self.current_lhs = "LesHouches.in.Step"
        # self.current_spc = 'Spheno.spc.Step'
        self.current_spc = "SPheno.spc.Step"

        # self.space = Space(self.hp)

        self.spheno = Spheno(
            spheno_dir=self.hp.directories.spheno,
            work_dir=self.sampler_id_dir,
            model_name=self.hp.model.name,
        )
        self.higgs_bounds = HiggsBounds(
            higgs_bounds_dir=self.hp.directories.higgsbounds,
            work_dir=self.sampler_id_dir,
            neutral_higgs=self.hp.model.neutral_higgs,
            charged_higgs=self.hp.model.charged_higgs,
        )
        self.higgs_signals = HiggsSignals(
            higgs_signals_dir=self.hp.directories.higgssignals,
            work_dir=self.sampler_id_dir,
            neutral_higgs=self.hp.model.neutral_higgs,
            charged_higgs=self.hp.model.charged_higgs,
        )
        self.lhs = LesHouches(
            file_dir=self.hp.directories.reference_lhs,
            work_dir=self.sampler_id_dir,
            model=self.hp.model.name,
        )

    def create_dir(self, run_name: str):
        if not (os.path.exists(self.sampler_id_dir)):
            os.makedirs(self.sampler_id_dir)

    def get_lhs(self):
        return self.lhs

    def run_stack_from_lhs(self):
        """
        Run the hep_stack with the LHS file saved in the
        self.lhs property.
        """
        param_card = None
        # Create new lhs file with the parameter values
        self.lhs.new_file(self.current_lhs)
        # Run spheno with the new lhs file
        param_card, spheno_stdout = self.spheno.run(self.current_lhs, self.current_spc)
        if param_card is not None:
            self.higgs_bounds.run()
            self.higgs_signals.run()
            higgs_signals_results = HiggsSignalsResults(
                self.sampler_id_dir, model=self.hp.model.name
            ).read()
            higgs_bounds_results = HiggsBoundsResults(
                self.sampler_id_dir, model=self.hp.model.name
            ).read()
            slha = SLHA(param_card, self.sampler_id_dir, model=self.hp.model.name)

            hep_stack_data = {
                "LHE": self.lhs.as_dict(),
                "SLHA": slha.as_dict(),
                "HB": higgs_bounds_results,
                "HS": higgs_signals_results,
            }
        else:
            hep_stack_data = {
                "LHE": self.lhs.as_dict(),
                "SLHA": None,
                "HB": None,
                "HS": None,
            }
            return hep_stack_data

    def sample(self, parameter_point: np.ndarray) -> Dict[str, float] | None:

        _ = input_vector_to_lhs(parameter_point, self.lhs, self.hp_input)

        hep_stack_data = self.run_stack_from_lhs()

        return hep_stack_data

    def __call__(self, parameter_point: np.ndarray) -> np.ndarray:
        return self.sample(parameter_point)

    def close(self):
        shutil.rmtree(self.sampler_id_dir)


@register
class SPhenoMg5(SPhenoHbHs):
    """
    Class to run from (LHS, MG5Script) -> SPHENO, HB, HS -> MG5 -> Output Dir.
    """

    def __init__(
        self,
        sampler_id,
        hep_config: DictConfig,
        mg5_script: Dict,
    ):
        super().__init__(sampler_id=sampler_id, hep_config=hep_config)
        self.mg5_script = mg5_script

        self.madgraph = Madgraph(
            madgraph_dir=self.hp.directories.madgraph, work_dir=self.sampler_id_dir
        )

    def update_mg5_script(self, process, mg5_script):
        """
        Takes the template madgraph script and updated with current directories
        to be consistent with the HEPSTACK.
        """
        with open(mg5_script, "r") as file:
            content = file.read()
        # Replace WORK_DIR variable
        self.mg5_work_dir = os.path.join(self.sampler_id_dir, "mg5_{}".format(process))
        content = content.replace("WORK_DIR", self.mg5_work_dir)
        # Replace PARAM_CARD_DIR variable
        new_param_card_dir = os.path.join(
            self.sampler_id_dir,
            "SPheno" + self.hp.model.name + "_output",
            self.current_spc,
        )
        content = content.replace("PARAM_CARD_DIR", new_param_card_dir)
        self.new_mg5_script = os.path.join(self.sampler_id_dir, "mg5_script.txt")
        with open(self.new_mg5_script, "w+") as file:
            file.write(content)

    def run_stack_from_lhs(self):
        """
        Run the hep_stack with the LHS file saved in the
        self.lhs property.
        """
        param_card = None

        self.lhs.new_file(self.current_lhs)
        # Run spheno with the new lhs file
        param_card, spheno_stdout = self.spheno.run(self.current_lhs, self.current_spc)

        if param_card is not None:

            slha = SLHA(param_card, self.sampler_id_dir, model=self.hp.model.name)

            # Configure LHS so that the SPheno output is compatible with Madgraph
            self.lhs["SPHENOINPUT"][520] = 0  # Effective couplings
            self.lhs["SPHENOINPUT"][78] = 1  # Write madgraph
            self.lhs["SPHENOINPUT"][525] = 0  # Contribution to diphoton
            self.lhs["SPHENOINPUT"][16] = 0  # One loop decays

            self.lhs.new_file(self.current_lhs)
            # Run spheno with the new lhs file
            param_card, spheno_stdout = self.spheno.run(
                self.current_lhs, self.current_spc
            )

            # Run heptools if SPheno succeeded
            self.higgs_bounds.run()
            self.higgs_signals.run()

            higgs_signals_results = HiggsSignalsResults(
                self.sampler_id_dir, model=self.hp.model.name
            ).read()
            higgs_bounds_results = HiggsBoundsResults(
                self.sampler_id_dir, model=self.hp.model.name
            ).read()
            # slha = SLHA(
            #    param_card,
            #    self.sampler_id_dir,
            #    model = self.hp.model.name
            #    )

            mg5_result = {}
            for process in self.mg5_script.keys():
                self.update_mg5_script(
                    process=process, mg5_script=self.mg5_script[process]
                )
                self.madgraph.run(input_file=self.new_mg5_script)
                mg5_result[process] = read_mg_generation_info(
                    os.path.join(
                        self.mg5_work_dir, "Events/run_01/run_01_tag_1_banner.txt"
                    )
                )

            hep_stack_data = {
                "LHE": self.lhs.as_dict(),
                "SLHA": slha.as_dict(),
                "HB": higgs_bounds_results,
                "HS": higgs_signals_results,
                "MG5": mg5_result,
            }
        else:
            hep_stack_data = {
                "LHE": self.lhs.as_dict(),
                "SLHA": None,
                "HB": None,
                "HS": None,
                "MG5": None,
            }
        return hep_stack_data


@register
class THDMCExt:
    """THDMC Stack with External calculations to get mu values"""

    def __init__(
        self,
        sampler_id: int,
        hep_config: DictConfig,
    ):
        """Initialize the HEP tools"""
        self.hp = hep_config
        self.hp_input = hep_config.model.input
        self.scan_dir = self.hp.directories.scan_dir
        self.sampler_id_dir = os.path.join(self.scan_dir, str(sampler_id))

        # self.space = Space(self.hp)

        self.thdmc = THDMC(
            self.hp.directories.thdmc,
            self.sampler_id_dir,
            "type3",
        )

    def create_dir(self, run_name: str):
        if not (os.path.exists(self.sampler_id_dir)):
            os.makedirs(self.sampler_id_dir)

    def sample(self, parameter_point: np.ndarray) -> Dict[str, float]:

        parameter_point = np.array(
            [
                parameter_point[0],
                125.0,
                parameter_point[1],
                parameter_point[4],
                parameter_point[2],
                parameter_point[3],
                0.0,
                0.0,
                parameter_point[5],
                999.0,
            ]
        )
        _ = self.thdmc.run(parameter_point)
        slha = self.thdmc.result()

        if slha is not None:
            slha_dict = slha.as_dict()

            # Temporal extra information
            h1TT = np.sqrt(slha["MGUSER"][61][0] ** 2 + slha["MGUSER"][62][0] ** 2)
            h1ZZ = np.sqrt(slha["MGUSER"][161][0] ** 2 + slha["MGUSER"][162][0] ** 2)

            hep_stack_data = {"SLHA": slha_dict, "EXT": {"h1TT": h1TT, "h1ZZ": h1ZZ}}
        else:
            hep_stack_data = {"SLHA": None, "EXT": {"h1TT": None, "h1ZZ": None}}
        return hep_stack_data

    def __call__(self, parameter_point: np.ndarray) -> np.ndarray:
        return self.sample(parameter_point)

    def close(self):
        shutil.rmtree(self.sampler_id_dir)
