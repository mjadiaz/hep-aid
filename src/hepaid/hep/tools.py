import os
import re
import subprocess
from pathlib import Path


import numpy as np

from hepaid.hep.read import SLHA


class SPheno:

    """Manages the execution of SPheno (Supersymmetric Phenomenology) calculations.

    This class facilitates the setup, execution, and output handling of SPheno.

    Attributes:
        spheno_dir (str): The directory containing the SPheno executable.
        output_dir (str): The directory where SPheno output files will be stored.
        model_name (str):  The name of the specific SUSY model to be analyzed.
        output_file_name (str):  The name of the primary output file.
        output_file_path (str):  The full path to the primary output file.
        standard_input_file_name (str): The standard name for SPheno input files.
        model_executable (str): The full path to the SPheno executable for the specified model.

    Methods:
        __init__(self, spheno_dir, output_dir, model_name):
            Initializes the SPheno object with required directory and model information.
        
        run(self, input_file):
            Executes SPheno with the provided input file or SLHA object.
    """
    def __init__(self, heptool_dir: str, output_dir: str , model_name: str):
        """Initializes a new SPheno object.

        Parameters: 
            spheno_dir (str): Path to the directory containing SPheno.
            output_dir (str): Path to the directory where SPheno output should be saved.
            model_name (str): Name of the SUSY model to be used.
        """
        self.heptool_dir = Path(heptool_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.executable = self.heptool_dir / "bin" / f'SPheno{self.model_name}'

        self.output_file_name = f'SPheno.spc.{self.model_name}'
        self.output_file_path = self.output_dir / self.output_file_name

        self.standard_input_file_name = f'LesHouches.in.{self.model_name}'
    
    def run(self, input_file: str | SLHA) -> tuple[str | None, str]:
        """Runs the SPheno calculation.

        Parameters:
            input_file (str or SLHA): Either a path to an input SLHA (LesHouches.in.Model) 
                                        file (str) or an SLHA object containing the SLHA file.

        Returns:
            tuple: A tuple containing:
                - Path: The path to the output file if the run was successful, or None if it failed.
                - str:  The standard output (stdout) from the SPheno process. 
        """
        if isinstance(input_file, str):
            input_file_path = Path(input_file_path)
        elif isinstance(input_file, SLHA):
            input_file_path = self.output_dir / self.standard_input_file_name
            input_file.save(input_file_path)
        else:
            raise TypeError(f"Invalid input_file type: Expected str or SLHA, got {type(input_file).__name__}")

        run = subprocess.run(
            [
                self.executable,
                input_file_path,
                self.output_file_path,
            ],
            capture_output=True,
            text=True,
            cwd=self.output_dir,
        )
        if "Finished" in run.stdout:
            return self.output_file_path, run.stdout
        else:
            return None, run.stdout


class _SPheno:
    def __init__(self, spheno_dir, work_dir, model_name):
        self._spheno_dir = Path(spheno_dir) 
        self._work_dir = Path(work_dir)  
        self._model_name = model_name

    def run(self, in_file_name, out_file_name):
        """Run SPhenoMODEL with input/output files in specific directories.

        Parameters:
        in_file_name (str): Input file's name (located in work_dir/SPhenoMODEL_input/).
        out_file_name (str): Output file's name (created in work_dir/SPhenoMODEL_output/).

        Returns:
            Tuple[Optional[Path], str]: Path to output file if successful, else None. 
                                        SPheno subprocess stdout.
        """

        in_dir = self._work_dir / f"SPheno{self._model_name}_input"
        out_dir = self._work_dir / f"SPheno{self._model_name}_output"

        in_file = in_dir / in_file_name
        out_file = out_dir / out_file_name

        out_dir.mkdir(exist_ok=True)  # Create output dir if not exists

        spheno_executable = self._spheno_dir / "bin" / f"SPheno{self._model_name}"

        run = subprocess.run(
            [spheno_executable, in_file, out_file],
            capture_output=True,
            text=True,
            cwd=self._work_dir, 
        )

        if "Finished" in run.stdout:
            return out_file, run.stdout
        else:
            return None, run.stdout
        
        

class Spheno:
    def __init__(self, spheno_dir, work_dir, model_name):
        self._spheno_dir = spheno_dir
        self._work_dir = work_dir
        self._model_name = model_name
    
    def run(self, input_file_path, output_directory, output_file_name = 'SPheno.spc.model'):
        input_file_path = Path(input_file_path)

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        output_file_path = output_directory / output_file_name
        run = subprocess.run(
            [
                self._spheno_dir + "/bin" + "/SPheno" + self._model_name,
                input_file_path,
                output_file_path,
            ],
            capture_output=True,
            text=True,
            cwd=output_directory,
        )
        if "Finished" in run.stdout:
            return out_file_path, run.stdout
        else:
            return None, run.stdout


    def run(self, in_file_name, out_file_name):
        """
        Run SPhenoMODEL with the input_file_name in work_dir.

        Parameters:
        -----
        in_file_name: str = input file's name located in work_dir/SPhenoMODEL_input/in_file_name
        out_file_name: str = output file's name located in work_dir/SPhenoMODEL_output/out_file_name

        Return:
        ------
        out_file: str = Global path to the output file name if SPheno runs successfully. None if SPheno gets an Error.
        spheno_stdout = stdout of the subprocess that runs SPheno.
        """
        # Reads the input file created with LesHouches.new_file in work_dir
        in_file = os.path.join(
            self._work_dir, "SPheno" + self._model_name + "_input", in_file_name
        )
        # Create and output directory in work_dir/SPhenoMODEL_output
        out_dir = os.path.join(self._work_dir, "SPheno" + self._model_name + "_output")
        if not (os.path.exists(out_dir)):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, out_file_name)

        run = subprocess.run(
            [
                self._spheno_dir + "/bin" + "/SPheno" + self._model_name,
                in_file,
                out_file,
            ],
            capture_output=True,
            text=True,
            cwd=self._work_dir,
        )
        if "Finished" in run.stdout:
            return out_file, run.stdout
        else:
            return None, run.stdout


class Madgraph:
    """
    Basicaly just run madgraph with in the script mode with the scripts created \n
    by the MG5Script class. \n
    Todo:
    - Figure out how to print some output from madgraph.
    """

    def __init__(self, heptool_dir, output_dir):
        self.tool_dir = Path(heptool_dir)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.executable = self.tool_dir / "bin/mg5_aMC"


    def run(self, input_file):
        """
        Run madgraph with an script named MG5Script.txt (created by the MG5Script class) in within work_dir. \n
        Change input_file to change to another script within work_dir.
        """
        subprocess.run(
            [self.executable, input_file],
            capture_output=True,
            text=True,
            cwd=self.output_dir,
        )
        return True


class HiggsBounds:
    def __init__(
        self,
        higgs_bounds_dir,
        work_dir,
        model=None,
        neutral_higgs=None,
        charged_higgs=None,
        output_mode=False,
    ):
        self._dir = higgs_bounds_dir
        self.work_dir = work_dir
        self.model = model
        self.neutral_higgs = neutral_higgs
        self.charged_higgs = charged_higgs
        self.output_mode = output_mode

    def run(self):
        """
        Runs HiggsBounds with the last point calculated by SPheno.
        Returns the path for HiggsBounds_results.dat if SPheno and
        HiggsBounds succeed. If there is an Error in one them it
        returns None and prints the error output.
        """
        run = subprocess.run(
            [
                os.path.join(self._dir, "HiggsBounds"),
                "LandH",
                "effC",
                str(self.neutral_higgs),
                str(self.charged_higgs),
                self.work_dir + "/",
            ],
            capture_output=True,
            text=True,
            cwd=self.work_dir,
        )
        if "finished" in run.stdout:
            if self.output_mode:
                print(run.stdout)
            return os.path.join(self.work_dir, "HiggsBounds_results.dat")
        else:
            if self.output_mode:
                print("HiggsBound not finished!")
                print(run.stdout)
            return None


class HiggsSignals:
    def __init__(
        self,
        higgs_signals_dir,
        work_dir,
        model=None,
        neutral_higgs=None,
        charged_higgs=None,
        output_mode=False,
    ):
        self._dir = higgs_signals_dir
        self.work_dir = work_dir
        self.model = model
        self.neutral_higgs = neutral_higgs
        self.charged_higgs = charged_higgs
        self.output_mode = output_mode

    def run(self):
        """
        Runs HiggsSignals with the last point calculated by SPheno.
        """
        run = subprocess.run(
            [
                os.path.join(self._dir, "HiggsSignals"),
                "latestresults",
                "2",
                "effC",
                str(self.neutral_higgs),
                str(self.charged_higgs),
                self.work_dir + "/",
            ],
            capture_output=True,
            text=True,
            cwd=self.work_dir,
        )
        if not ("Error") in run.stdout:
            if self.output_mode:
                print(run.stdout)
            return os.path.join(self.work_dir, "HiggsSignals_results.dat")
        else:
            if self.output_mode:
                print("HiggsSignals Error")
                print(run.stdout)
            return None


class THDMC:
    """
    Utility class to run 2HDMC programs.
    """

    def __init__(
        self,
        tool_dir: str,
        work_dir: str,
        program_name: str,
    ):
        self.tool_dir = tool_dir
        self.work_dir = work_dir
        self.program_name = program_name
        self.program = os.path.join(tool_dir, program_name)
        self.thdm_dir = os.path.join(self.work_dir, "2HDMC")

    def run(
        self,
        parameters: np.ndarray,
    ) -> bool:
        """
        Parameters are in the standard basis:
        lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7,\
            m12_2, tan_beta = parameters
        """

        if not (os.path.exists(self.thdm_dir)):
            os.makedirs(self.thdm_dir)

        parameters = [str(val) for val in parameters]

        # lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7,\
        #    m12_2, tan_beta = parameters
        try:
            run = subprocess.run(
                [
                    self.program,
                    *parameters,
                    # lambda1,
                    # lambda2,
                    # lambda3,
                    # lambda4,
                    # lambda5,
                    # lambda6,
                    # lambda7,
                    # m12_2,
                    # tan_beta
                ],
                capture_output=True,
                text=True,
                cwd=self.thdm_dir,
            )
            return True
        except:
            return False

    def result(self) -> SLHA:
        """Returns the SLHA file saved in working directory, after .run()"""
        file = os.path.join(self.thdm_dir, "Demo_out.lha")
        if os.path.exists(file):
            slha = SLHA(file=file)
            return slha
        else:
            return None
