import numpy as np
import time
from functools import partial
from omegaconf import DictConfig

from typing import Callable, Tuple

from pathlib import Path
from hepaid.search.method.base import Method
from hepaid.search.objective import Objective


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def smooth_mask(x, a, eps=2e-3):
    """Returns 0ish for x < a and 1ish for x > a"""
    return sigmoid((x - a) / eps)

def smooth_box_mask(x, a, b, eps=2e-3):
    """Returns 1ish for a < x < b and 0ish otherwise"""
    return smooth_mask(x, a, eps) - smooth_mask(x, b, eps)

def auto_likelihood(result: dict, objectives: dict, mode: str = 'mult') -> float:
    """
    Automatically creates a likelihood value based on the objectives configuration,
    applying sigmoid functions within specified boundaries. Supports two aggregation modes: sum and multiply.

    Parameters:
    -----------
    result : dict
        A dictionary containing the results where keys are parameter names and values.
    objectives : dict
        A dictionary containing the objectives configuration with two keys:
    mode : str, optional
        The aggregation mode for likelihood calculation. Can be either 'sum' or 'mult'. Default is 'mult'.

    Returns:
    --------
    float
        The calculated likelihood value based on the given objectives and mode.

    Raises:
    -------
    AssertionError
        If the first condition in a double constraint is not 'gt'.
        If the first element in a single constraint is not a string.

    Example:
    --------
    result = {'param1': 3, 'param2': 4}
    objectives = {
        'double_constraint': {'param1': [['gt', 2], ['lt', 4]]},
        'single_constraint': {'param2': ['lt', 5]}
    }
    likelihood = auto_likelihood(result, objectives, mode='sum')
    """
    if mode == 'sum':
        likelihood = 0 
    if mode == 'mult':
        likelihood = 1
    if any(value is None for value in result.values()):
        return None
    else:
        for p in objectives["double_constraint"].keys():
            constraint = objectives['double_constraint'][p]
            error_msg = f"First double constraint should be a lower bound. ex: [['gt', 2],['lt', 4]]"
            assert constraint[0][0] == 'gt', error_msg
            lh = smooth_box_mask(result[p], constraint[0][1], constraint[1][1],  1e-3 )
            if mode == 'sum':
                likelihood += lh 
            else:
                likelihood *= lh

        for p in objectives["single_constraint"].keys():
            constraint = objectives['single_constraint'][p]
            error_msg = f"str condition should be in the first element. ex: ['lt', 4]]"
            assert isinstance(constraint[0], str), error_msg
            if constraint[0] == 'lt':
                lh = 1 - smooth_mask(result[p], constraint[1], 1e-3)
            else:
                lh = smooth_mask(result[p], constraint[1], 1e-3)

            if mode == 'sum':
                likelihood += lh 
            else:
                likelihood *= lh
        return likelihood

class MCMCMH(Method):
    """
    MCMC-MH (Markov Chain Monte Carlo - Metropolis-Hastings) sampling policy for search.

    Parameters:
        objective_function (Objective): The objective function instance.
        hyper_parameters(DictConfig): Hyperparameters for the MCMC-MH algorithm.
        metrics(Metrics): Metrics instance. Default is None.
        likelihood(Callable): Custom likelihood function. Default is None.

    Attributes:
        idx_burnin (int): Number of iterations for the burn-in phase. 
        scale (float): Initial scale of the proposal distribution.
        adapt_frequency (int): Frequency of adapting the proposal distribution scale.
        likelihood (Callable): Likelihood function used in the MCMC-MH algorithm.
        accepted (int): Counter for accepted proposals.
        curr_state (np.ndarray): Current state in the parameter space.
        curr_likeli (float): Likelihood of the current state.
        iteration (int): Current iteration number.
    """

    def __init__(self,
                 objective_function: Objective, 
                 hyper_parameters: DictConfig | str | None = None, 
                 likelihood: Callable | None = None,
                 ) -> None:
        """
        Initializes the MCMCMH class.

        Parameters:
            objective_function(Objective): The objective function instance.
            hyper_parameters(DictConfig): Hyperparameters for the MCMC-MH algorithm.
            metrics(Metrics): Metrics for evaluating the performance of the algorithm. Default is None.
            likelihood(Callable): Custom likelihood function. Default is None.
        """
        super().__init__(objective_function, hyper_parameters)

        self.idx_burnin = self.hp.burn_in
        self.scale = self.hp.initial_scale
        self.adapt_frequency = self.hp.adapt_frequency
        
        if likelihood is None:
            self.likelihood = lambda x, add: auto_likelihood(
                result = self.objective_function.sample(x, add=add), 
                objectives = self.objective_function.config.objectives,
                mode = 'mult'
            )
        else:
            self.likelihood = likelihood


        self.accepted = 0

    
    def run(self):
        """
        Executes the MCMC-MH sampling algorithm.

        The method initializes the state, performs the MCMC-MH sampling,
        adapts the proposal distribution scale during the burn-in phase,
        logs the progress, and saves checkpoints.

        Returns:
            bool: True if the algorithm runs successfully.
        """
        # Generate the initial state automatically
        input_size = len(self.objective_function.config.input_space.keys())
        initial_state = np.random.uniform(0,1, size=input_size)
        initial_likelihood = self.likelihood(initial_state, add=False)

        # BSM cases often output None
        while initial_likelihoood == None:
            initial_state = np.random.uniform(0.2,.6,size=input_size)
            initial_likelihoood = self.likelihood(initial_state, add=False)

        # Update current state and likelihood
        self.curr_state = initial_state
        self.curr_likeli = initial_likelihood

        # Start rich progress logger
        progress = self.metrics.start_progress(description='MCMC-MH Sampling ...')
        with progress:
            for i in progress.track(range(
                self.iteration, self.iteration + self.hp.total_iterations)):

                # Update current Objective Function metrics
                self.metrics.update(self.objective_function, i)

                # Perform MH update
                self.curr_state, self.curr_likeli, success = mcmc_updater(
                    curr_state=self.curr_state,
                    curr_likeli=self.curr_likeli,
                    likelihood=partial(self.likelihood, add=True),
                    proposal_distribution=proposal_distribution,
                    scale=self.scale,
                )

                self.accepted += 1 if success else 0

                # Adapt scale during burn-in phase
                if i < self.idx_burnin and (i + 1) % self.adapt_frequency == 0 and self.accepted > 1:
                    acceptance_rate = self.accepted / self.adapt_frequency
                    self.scale = tune(self.scale, acceptance_rate)
                    self.accepted = 0  # Reset the counter
                
                # Log and save
                self.metrics.log(progress)
                self.save_checkpoint(i)

                self.iteration = i
        return True


def mcmc_updater(
        curr_state:np.ndarray, 
        curr_likeli: float, 
        likelihood: Callable, 
        proposal_distribution: Callable, 
        scale: float
    ) -> Tuple[np.ndarray, float, bool]:
    """
    MCMC-MH update. Generate a proposal state using the proposal distribution. 
    Ensure the proposal state is within bounds. Calculate the acceptance criterion.
    Generate a random number between 0 and 1. If acceptance criterion is greater than 
    random number proposal is accepted.

    Parameters:
        curr_state (np.ndarray): Current state in the parameter space.
        curr_likeli (float): Likelihood of the current state.
        likelihood (Callable): Function to compute the likelihood of a given state.
        proposal_distribution (Callable): Function to generate a proposal state from the current state.
        scale (float): Scale parameter for the proposal distribution.

    Returns:
        Tuple[np.ndarray, float, bool]: A tuple containing the new state, its likelihood, and
                                    a boolean indicating whether the proposal was accepted.
    """
    # Generate a proposal state using the proposal distribution
    proposal_state = proposal_distribution(curr_state, stepsize=scale)

    # if is not within bounds
    while not all(0 < element < 1 for element in proposal_state):
        proposal_state = proposal_distribution(curr_state, stepsize=scale)
    
    # Calculate the acceptance criterion
    prop_likeli = likelihood(proposal_state)

    if isinstance(prop_likeli, type(None)):
        # In HEPStack Likelihoods can be None 
        return curr_state, curr_likeli, False
    else:
        accept_crit = prop_likeli / curr_likeli

        # Generate a random number between 0 and 1
        accept_threshold = np.random.uniform(0, 1)

        if accept_crit > accept_threshold:
            return proposal_state, prop_likeli, True
        else:
            return curr_state, curr_likeli, False



def proposal_distribution(x: np.ndarray, stepsize: float = 0.5) -> np.ndarray:
    """
    Generates a  simple proposal state using a normal distribution centered on the 
    current state.

    Parameters:
        x (np.ndarray): Current state in the parameter space.
        stepsize (float): Step size or scale parameter for the proposal distribution. 
                        Default is 0.5.

    Returns:
        np.ndarray: Proposed state generated from the normal distribution.
    """
    prop_dist = np.random.normal(x, stepsize)
    return prop_dist




def tune(scale: float, acceptance: float, target_acceptance: float = 0.234) -> float:
    """
    Adjusts the scale of the proposal distribution based on the acceptance rate.

    This function is borrowed from PyMC3 and is used to adapt the scale to achieve a 
    target acceptance rate.

    Parameters:
        scale (float): Current scale of the proposal distribution.
        acceptance (float): Current acceptance rate of the proposals.
        target_acceptance (float): Target acceptance rate. Default is 0.234.

    Returns:
        float: Adjusted scale for the proposal distribution.
    """
    if acceptance > target_acceptance:
        scale *= 1.1  # Increase scale by 10%
    else:
        scale /= 1.1  # Decrease scale by 10%

    return scale