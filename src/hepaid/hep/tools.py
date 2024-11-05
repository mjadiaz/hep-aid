"""
This module provides classes and methods for managing and executing various high-energy 
physics tools such as SPheno, Madgraph, HiggsBounds, and HiggsSignals. 

Classes defined within this module facilitate the setup, execution, and analysis of outputs from these tools. 
Each class corresponds to a specific tool, abstracting away the complexities of file management, 
command-line argument construction, and result parsing, thereby simplifying the process of integrating these tools into 
larger computational workflows. Each tool has the form of the `BaseTool` class, with the run and results methods.

Key features include:
- Simplified interface for running computations with minimal boilerplate code.
- Automatic handling of input and output files, including creation of necessary directories.
- Parsing of tool-specific output formats into structured Python objects for further analysis.

This module is designed to be easily extendable, allowing for the addition of support for further tools or 
customised configurations as needed.
"""
import os
import re
import subprocess
from pathlib import Path


import numpy as np

from hepaid.hep.read import SLHA
from hepaid.hep.read import HiggsBoundsResults, HiggsSignalsResults
from hepaid.hep.read import read_mg_generation_info

class BaseTool:
    """
    Base class for a HEP tool supported by this module.
    
    This class provides a common interface for running and retrieving results from various tools 
    like SPheno, Madgraph, HiggsBounds, and HiggsSignals. Subclasses should implement their own logic 
    for the `run` and `results` methods according to the specifics of the tool they represent.
    """

    def __init__(self, heptool_dir: str, output_dir: str):
        """
        Initialises a new instance of the BaseTool class.
        
        Parameters:
            heptool_dir (str): The directory path containing the tool's executables.
            output_dir (str): The directory path where the tool's output will be stored.
        """
        self.heptool_dir = Path(heptool_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *args, **kwargs):
        """
        Executes the tool with the provided arguments.
        
        This method should be overridden by subclasses to implement the actual execution logic of the tool.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the run method.")

    def results(self):
        """
        Retrieves the results produced by the tool.
        
        This method should parse the tool's output and return the results in a structured format. 
        The implementation details depend on the specific output format of the tool.
        If no additional arguments are needed is recommended to use the @property decorator.
        
        Returns:
            Any: The results parsed from the tool's output.
        """
        raise NotImplementedError("Subclasses must implement the results method.")

class SPheno:

    """Manages the execution of SPheno (Supersymmetric Phenomenology) calculations.

    This class facilitates the setup, execution, and output handling of SPheno.

    Attributes:
        heptool_dir (str): The directory containing the SPheno directory (after extracting from zipped file).
        output_dir (str): The directory where SPheno output files will be stored.
        model_name (str):  The name of the specific SUSY model to be analyzed.
        output_file_name (str):  The name of the primary output file.
        output_file_path (str):  The full path to the primary output file.
        standard_input_file_name (str): The standard name for SPheno input files.
        model_executable (str): The full path to the SPheno executable for the specified model.

    Methods:
        __init__(self, heptool_dir, output_dir, model_name):
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
    
    @property
    def results(self):
        if self.output_file_path.exists():
            results_obj = SLHA(self.output_file_path)
            return results_obj
        else:
            return None

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
            input_file_path = Path(input_file)
        elif isinstance(input_file, SLHA):
            input_file_path = self.output_dir / self.standard_input_file_name
            input_file.save(input_file_path)
        else:
            raise TypeError(f"Invalid input_file type: Expected str or SLHA, got {type(input_file).__name__}")

        run_commands =  [
                self.executable,
                input_file_path,
                self.output_file_path,
            ]
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
        self.stdout = run.stdout
        print(self.stdout)



class Madgraph:
    """Manages the execution of Madgraph.

    This class facilitates the setup, execution, and output handling of Madgraph.

    Attributes:
        heptool_dir (str): The directory containing the SPheno executable.
        output_dir (str): The directory where Madgraph output will be stored.
        executable (str): The full path to the Madgraph mg5_aMC executable.
    """
          
    def __init__(self, heptool_dir, output_dir):
        self.heptool_dir = Path(heptool_dir)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.executable = self.heptool_dir / "bin/mg5_aMC"


    
    def results(self, output_path: str):
        """
        Results from MadGraph depends on the output folder which is internal to the script.
        Then, we need to give it as an input in output_path. It assumes a single process with 
        one run.

        Parameters:
            output_path (str): Path to the madgraph output directory
        """
        output_path = Path(output_path)/ "Events/run_01/run_01_tag_1_banner.txt"
        if output_path.exists():
            results_obj = read_mg_generation_info(output_path)
            return results_obj
        else:
            return None

    def run(self, input_file: str):
        """
        Run madgraph with a script in `input_file` (an example script can be created by the MG5Script class) in within work_dir. 

        Parameters:
            input_file (str): Madgraph script.

        Returns:
            bool: True.
        """
        run = subprocess.run(
            [self.executable, input_file],
            capture_output=True,
            text=True,
            cwd=self.output_dir,
        )
        self.stdout = run.stdout


class HiggsBounds:
    """A class for running HiggsBounds.

    Attributes:
        heptool_dir (str): The directory path containing the HiggsBounds executable.
        output_dir (str): The directory path to store the output results.
        neutral_higgs (int): The number of neutral Higgs bosons in the model.
        charged_higgs (int): The number of charged Higgs bosons in the model.
        executable (str): The path to the HiggsBounds executable.
        output_file_path (str): The path where HiggsBounds results will be saved.

    """
    def __init__(
        self,
        heptool_dir: str,
        output_dir: str,
        neutral_higgs: int,
        charged_higgs: int,
    ):
        """Initializes the HiggsBounds object.

        Parameters:
            heptool_dir (str): The directory path to the HiggsBounds build.
            output_dir (str): The directory path to store the output results.
            neutral_higgs (int): The number of neutral Higgs bosons in the model.
            charged_higgs (int): The number of charged Higgs bosons in the model.
        """
        self.heptool_dir = Path(heptool_dir)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.neutral_higgs = neutral_higgs
        self.charged_higgs = charged_higgs

        self.executable = self.heptool_dir / "HiggsBounds"

        self.output_file_path =  self.output_dir / "HiggsBounds_results.dat"

    @property
    def results(self):
        if self.output_file_path.exists():
            results_obj = HiggsBoundsResults(self.output_dir)
            return results_obj.read()
        else:
            return None

    def run(self):
        """Executes HiggsBounds.

        Runs HiggsBounds with the specified parameters and returns the output file path and stdout.
        
        Returns:
            tuple: A tuple containing:
                - output_file_path (Path): Path to the HiggsBounds_results.dat file.
                - stdout (str): The standard output from the HiggsBounds process.

        Raises:
            RuntimeError: If HiggsBounds fails to run successfully (e.g., doesn't produce the expected "finished" output).

        """
        run = subprocess.run(
            [
                self.executable,
                "LandH",
                "effC",
                str(self.neutral_higgs),
                str(self.charged_higgs),
                str(self.output_dir) + "/",
            ],
            capture_output=True,
            text=True,
            cwd=self.output_dir,
        )
        self.stdout = run.stdout

class HiggsSignals:
    """A class for running HiggsSignals.

    Attributes:
        heptool_dir (str): The directory path to the HiggsSignals builds.
        output_dir (str): The directory path to store the output results.
        neutral_higgs (int): The number of neutral Higgs bosons in the model.
        charged_higgs (int): The number of charged Higgs bosons in the model.
        executable (str): The path to the HiggsBounds executable.
        output_file_path (str): The path where HiggsBounds results will be saved.

    """
    def __init__(
        self,
        heptool_dir: str,
        output_dir: str,
        neutral_higgs: int,
        charged_higgs: int,
    ):
        """Initializes the HiggsBounds object.

        Parameters:
            heptool_dir (str): The directory path containing the HiggsBounds executable.
            output_dir (str): The directory path to store the output results.
            neutral_higgs (int): The number of neutral Higgs bosons in the model.
            charged_higgs (int): The number of charged Higgs bosons in the model.
        """
        self.heptool_dir = Path(heptool_dir)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.neutral_higgs = neutral_higgs
        self.charged_higgs = charged_higgs

        self.executable = self.heptool_dir / "HiggsSignals"

        self.output_file_path =  self.output_dir / "HiggsSignals_results.dat"


    @property
    def results(self):
        if self.output_file_path.exists():
            results_obj = HiggsSignalsResults(self.output_dir)
            return results_obj.read()
        else:
            return None

    def run(self):
        """Executes HiggsSignals.

        Runs HiggsSignals with the specified parameters and returns the output file path and stdout.
        
        Returns:
            tuple: A tuple containing:
                - output_file_path (Path): Path to the HiggsSignals_results.dat file.
                - stdout (str): The standard output from the HiggsSignals process.

        Raises:
            RuntimeError: If HiggsSignals fails to run successfully (e.g., doesn't produce the expected "Error" output).

        """
        run = subprocess.run(
            [
                self.executable,
                "latestresults",
                "2",
                "effC",
                str(self.neutral_higgs),
                str(self.charged_higgs),
                str(self.output_dir) + "/",
            ],
            capture_output=True,
            text=True,
            cwd=self.output_dir,
        )
        self.stdout = run.stdout

