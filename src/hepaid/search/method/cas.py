import numpy as np
import torch
import time
import os

from botorch.optim import optimize_acqf

import copy

import pandas as pd


from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from functools import partial

from hepaid.search.method.eci import ExpectedCoverageImprovement
from hepaid.search.models.model_list import get_model_list_gp, get_posterior
from hepaid.search.objective.objective import cas_obj_fn_export, identify_samples_which_satisfy_constraints
from hepaid.search.objective import Objective
from hepaid.search.objective.utils import generate_initial_dataset

from hepaid.search.method.base import Method


tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}


def eci_step(
    resolution: float,
    X: np.ndarray,
    Y: np.ndarray,
    constraints: list,
    bounds: np.ndarray,
):
    """
    Expected Coverage Improvement (ECI) step. Fitting a GP for each objective. 
    Uses the original implementation of Expected Coverage Improvement.
    Adapted from BoTorch implementation of CAS: https://botorch.org/tutorials/constraint_active_search.

    Parameters:
        resolution (float): The hyper-sphere radius for the ECI acquisition function.
        X (np.ndarray): The input data points.
        Y (np.ndarray): The outputs values corresponding to the input data points.
        constraints (list): A list of constraints (given by cas_obj_fn_export()).
        bounds (np.ndarray): The bounds for the optimization variables (given by cas_obj_fn_export()).

    Returns:
        x_next (np.ndarray): The next suggested input point by the ECI acquisition function.
        model_list_gp (list): A list of Gaussian Process models fitted to each objective.
        eci (ExpectedCoverageImprovement): The ECI acquisition function instance.
    """

    model_list_gp = get_model_list_gp(X, Y)

    eci = ExpectedCoverageImprovement(
        model=model_list_gp,
        constraints=constraints,
        punchout_radius=resolution,
        bounds=bounds,
        num_samples=128,
    )

    x_next, _ = optimize_acqf(
        acq_function=eci,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
    )

    # To numpy
    x_next = x_next.cpu().numpy().flatten()
    return x_next, model_list_gp, eci


class CAS(Method):
    """
    Constraint Active Search (CAS) [1], an alternative formulation for the multiobjective design problem as an 
    Active Search Problem,  objectives are treated as constraints with known desired threshold values; 
    the goal is to find diverse samples that satisfy all the thresholds. The thresholds define the satisfactory
    region on the input space, denoted S. Expected Coverage Improvement (ECI), was developed for CAS. It places 
    hyper-spheres in the input data samples to measure diversity by aggregating the volume covered by each data point. 
    The optimal value of ECI corresponds to the sample in the input space that increases the volume covered of the S region
    by the current dataset. 


    [1]: Beyond the Pareto Efficient Frontier: Constraint Active Search for Multiobjective 
    Experimental Design, G. Malkomes, B. Cheng, E.H. Lee, and M. McCourt
    
    Parameters:
        objective (Objective): The Objective to perform the search.
        hyper_parameters (DictConfig | str): Hyperparameters for the CAS strategy.

    Attributes:
        delta_r (float): Rate of change of the resolution parameter.
        model (Any): The probabilistic surrogate model used in the CAS algorithm.
        likelihood (Any): The likelihood for the surrogate (GPs).
        eci (Any): Expected Coverage Improvement instance used in the CAS algorithm.
        metrics (Metrics): Metrics instance to collect and log performance metrics.
        iteration (int): Current iteration number.
    """

    def __init__(
            self, 
            objective: Objective,
            hyper_parameters: DictConfig | str | None = None
        ):
        """
        Initialise the CAS method.

        Parameters:
            objective (Objective): The Objective to perform the search.
            hyper_parameters (DictConfig | str): Hyper parameters for the CAS strategy.
        """
        super().__init__(objective, hyper_parameters)

        self.delta_r = abs(self.hp.resolution.initial - self.hp.resolution.final) / self.hp.resolution.r_decay_steps
        self.model = None
        self.likelihood = None
        self.eci = None

        self.metrics.new_custom_metrics(['r_parameter'])

        
    
    def run(self):
        """
        Executes the CAS search algorithm.

        Returns:
            model: The trained model after the search.
            likelihood: The likelihood associated with the model.
            eci: The Expected Coverage Improvement instance used.
            study: The study object containing the search results.
            metrics: The metrics collected during the search.
        """
        if self.hp.initial_dataset.generate:
            generate_initial_dataset(
                n_workers=self.hp.initial_dataset.n_workers,
                n_points=self.hp.initial_dataset.n_points,
                objective=self.objective,
                parallel=self.hp.parallel,
            )
        
        # Start rich progress logger
        progress = self.metrics.start_progress(description='CAS Search ...')

        with progress:
            for i in progress.track(range(
                self.iteration, self.iteration + self.hp.total_iterations)):

                # Export objective function to CAS required format
                valid, constraints, bounds, train_x, train_y = cas_obj_fn_export(
                    self.objective)

                # Update current Objective Function metrics
                self.metrics.update(self.objective, i)

                # Update resolution parameter cr
                if self.hp.resolution.constant_resolution:
                    cr = self.hp.resolution.value
                else:
                    cr = max(
                        self.hp.resolution.initial - i * self.delta_r,
                        self.hp.resolution.final
                        )

                self.metrics.update_custom({"r_parameter": cr})

                # Get next query by ECI
                x_next, model, eci = eci_step(
                    cr, train_x, train_y, constraints, bounds
                )
                # Sample the x_next configuration
                self.objective.sample(x_next)


                # Log and save
                self.metrics.log(progress)
                self.save_checkpoint(i)

                self.iteration = i

        return self.model, self.likelihood, self.eci, self.metrics


    

