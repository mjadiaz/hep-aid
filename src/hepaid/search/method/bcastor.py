import torch

import numpy as np
import pandas as pd
import optuna

from functools import partial
from omegaconf import DictConfig
from typing import Any, Tuple, Callable

from hepaid.search.method.eci import ECI
from hepaid.search.objective.objective_fn import obj_fn_export
from hepaid.search.models.model_list import get_model_and_likelihood
from hepaid.search.parallel.modules import run_x_with_pool
from hepaid.search.objective.utils import generate_initial_dataset
from hepaid.search.method.base import Method 
from hepaid.utils import load_config


optuna.logging.set_verbosity(optuna.logging.WARNING)

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

    
def create_optuna_objective(
        trial: optuna.trial.Trial, 
        space_config: dict, 
        eci: Callable
        ) -> np.ndarray:
    """
    Creates the optuna objective function needed for the optimization based on the space_config dictionary.
    Assumes the search space is normalized to [0,1]^d.

    Parameters:
        trial (optuna.trial.Trial): The trial object used by optuna to suggest values for the hyperparameters.
        space_config (dict): Configuration of the input space, specifying the features to be optimized.
        eci (Callable): The Expected Coverage Improvement function.

    Returns:
        float: The result of the ECI function for the suggested trial.
    """
    trials = []
    for feature in space_config.keys():
        trials.append(
            trial.suggest_float(feature, 0 , 1)
        )
  
    point = torch.tensor(trials).float().reshape(
        1,len(space_config)
        ).to(**tkwargs)
    result = eci(point)
    return  result.detach().cpu().numpy()

def optuna_study_tpe(objective: Callable, n_trials: int = 100) -> optuna.study.Study:
    """
    Maximizes the objective function using the Tree-structured Parzen Estimator (TPE) algorithm.

    Parameters:
        objective (Callable): The objective function to be optimized.
        n_trials (int, optional): Number of trials to run the optimization. Default is 100.

    Returns:
        optuna.study.Study: The optuna study object containing the optimization results.
    """
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=20,
            consider_prior=True,
            multivariate=True,
            constant_liar=False
            ),
        direction='maximize')

    study.optimize(objective, n_trials=n_trials)

    return study

def optuna_study_nsgaii(objective, n_trials=100):
    '''Maximise objective with NSGAII'''
    study = optuna.create_study(
        sampler=optuna.samplers.NSGAIISampler(),
        direction='maximize')

    study.optimize(objective, n_trials=n_trials)

    return study

def optuna_study_cmaes(objective, n_trials=100):
    '''Maximise objective with CMAES'''
    study = optuna.create_study(
        sampler=optuna.samplers.CmaEsSampler(),
        direction='maximize')

    study.optimize(objective, n_trials=n_trials)

    return study

def study_to_array(
        study: optuna.study.Study, 
        space_config: dict
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts the optuna study object to numpy arrays.

    Parameters:
        study (optuna.study.Study): The optuna study object containing the optimisation results.
        space_config (dict): Configuration of the input space.

    Returns:
        trials (np.ndarray): Array of trial configurations.
        values (np.ndarray): Array of values corresponding to the trial configurations.
    """
    df = study.trials_dataframe()
    loc_params = [f'params_{p}' for p in space_config.keys()]
    df = df.loc[:,['value'] + loc_params] 
    trials = df.loc[:,loc_params]
    values = df.value
    return trials.to_numpy(), values.to_numpy()

def sort_and_get_n_maximum(
        trials: np.ndarray, 
        values: np.ndarray, 
        n_points: int
        )->np.ndarray:
    """
    Sorts the trials based on their values in descending order and returns the top `n_points` trials.

    Parameters:
        trials (np.ndarray): Array of trial configurations.
        values (np.ndarray): Array of values corresponding to the trial configurations.
        n_points (int): Number of top trials to return.

    Returns:
        np.ndarray: Array of the top `n_points` trial configurations.
    """
    sorted_indices = np.argsort(values)[::-1]
    trials = trials[sorted_indices,:]
    return trials[:n_points]

def rank_based_sampling(
        trials: np.ndarray, 
        values: np.ndarray, 
        n_points: int, 
        alpha: int | float = 2
        ) -> np.ndarray:
    """
    Sorts the trials based on their ECI values. Sample n_points according to the rank-based probabilities,
    described in [1]. The argument alpha is the degree of prioritisation.  

    Parameters:
        trials (np.ndarray): Array of trial configurations.
        values (np.ndarray): Array of values corresponding to the trial configurations.
        n_points (int): Number of points to sample.
        alpha (float, optional): Exponent for the rank-based sampling probabilities. Default is 2.

    Returns:
        np.ndarray: Array of sampled trial configurations.
    """
    sorted_indices = np.argsort(values)[::-1]
    # sort the array by the last column
    n_trials = len(trials)
    trials = sort_and_get_n_maximum(trials, values, n_trials)
    

    priorities = np.power(1/np.arange(1,len(trials)+1), alpha)
    probs = priorities/priorities.sum()

    ids = np.arange(len(trials))
    select = np.random.choice(ids, size=n_points,p=probs, replace=False)
    X = trials[select]

    return X


def bcastor_step(
        hp: DictConfig,
        space_config: dict,
        constraints: list,
        bounds: list,
        train_x: torch.tensor,
        train_y: torch.tensor,
        model: Any,
        delta_r: float,
        step: int 
        ):
    """
    Perform a single step of the bCASTOR algorithm.

    ECI is optimized by the Tree-structured Parzen Estimator (TPE) algorithm. 
    The optimization history, referred to as trials, is sampled using a rank-sampling strategy, 
    where the rank is determined by the ECI value of each trial. This sampling strategy enables
    batch sampling, enhancing the discovery rate of the S region. bCASTOR reinterprets the ratio 
    of the hyper-spheres as a measure of the desired filling of samples, aiming to accumulate as many 
    diverse samples as possible.

    Parameters:
        hp (DictConfig): Hyperparameters for the bCASTOR algorithm.
        space_config (dict): Configuration of the input space.
        constraints (list): List of constraints for the search.
        bounds (list): Bounds for the search space.
        train_x (torch.tensor): Training data inputs.
        train_y (torch.tensor): Training data outputs.
        model (Any): Surrogate model used in the bCASTOR algorithm.
        delta_r (float): Rate of change of the resolution parameter.
        step (int): Current iteration step.

    Returns:
        eci (Callable): The Expected Coverage Improvement instance.
        study (optuna.study.Study): The optuna study object containing the search results.
        X (np.ndarray): The next set of samples to evaluate.
        trials (np.ndarray): The array of trial configurations.
        values (np.ndarray): The ECI values associated with the trials.
        cr (float): The current resolution parameter.
    """
    # Define resolution
    if hp.resolution.constant_resolution:
        cr = hp.resolution.value
    else:
        cr = max(hp.resolution.initial - step*delta_r, hp.resolution.final)

    eci = ECI(
        constraints=constraints,
        punchout_radius= cr,
        bounds=bounds,
        num_samples=hp.eci.num_samples,
        train_x = train_x,
        train_y =train_y
        )

    eci_fn = partial(eci,
        reference_points = train_x,
        surrogate_model = model,
        #scaler=scaler
        )


    # Create optuna study and optimise
    optuna_objective = partial(
        create_optuna_objective,
        space_config = space_config,
        eci=eci_fn
    )
    study = optuna_study_tpe(
        optuna_objective,
        n_trials=hp.batch_sampling.tpe_trials
        )

    trials, values = study_to_array(study, space_config)

    X = rank_based_sampling(
        trials,
        values,
        hp.batch_sampling.rank_samples,
        alpha = hp.batch_sampling.alpha
        )

    probs = eci._estimate_probabilities_of_satisfaction_at_points(
        surrogate_model=model,
        points=train_x
        )
    probs = probs[probs > 0]

    return eci, study, X, trials, values, cr




class bCASTOR(Method):
    """
    bCASTOR [1] is a variant of the Constraint Active Search (CAS) algorithm. 
    CAS is an alternative approach to the multiobjective design problem, formulated 
    as an Active Search Problem. In CAS, objectives are treated as constraints with known 
    desired threshold values, and the goal is to find diverse samples that satisfy all 
    the thresholds. These thresholds define the satisfactory region in the input space, 
    denoted as S.

    To measure diversity, CAS uses Expected Coverage Improvement (ECI), which places hyper-spheres 
    around input data samples and aggregates the volume covered by each data point.
    
    ECI is optimized by the Tree-structured Parzen Estimator (TPE) algorithm. 
    The optimization history, referred to as trials, is sampled using a rank-sampling strategy, 
    where the rank is determined by the ECI value of each trial. This sampling strategy enables
    batch sampling, enhancing the discovery rate of the S region. bCASTOR reinterprets the ratio 
    of the hyper-spheres as a measure of the desired filling of samples, aiming to accumulate as many 
    diverse samples as possible.

    [1]: Bayesian Active Search on Parameter Space: a 95 GeV Spin-0 Resonance in the (B - L) SSM. 
    M. A. Diaz, G. Cerro, S. Dasmahapatra, S. Moretti. https://arxiv.org/abs/2404.18653 
    
    Parameters:
        objective_function (ObjectiveFunction): The ObjectiveFunction to perform the search.
        hyper_parameters (DictConfig | str): Hyperparameters for the bCASTOR strategy.

    Attributes:
        delta_r (float): Rate of change of the resolution parameter.
        model (Any): The model used in the bCASTOR algorithm.
        likelihood (Any): The likelihood function used in the bCASTOR algorithm.
        eci (Any): Expected Coverage Improvement instance used in the bCASTOR algorithm.
        study (Any): The Optuna study object containing the trials.
        metrics (Metrics): Metrics instance to collect and log performance metrics.
        iteration (int): Current iteration number.
    """
    def __init__(self, objective_function, hyper_parameters = None):
        """
        Initialise the bCASTOR method.

        Parameters:
            objective_function (ObjectiveFunction): The ObjectiveFunction to perform the search.
            hyper_parameters (DictConfig | str | None) = None: Hyperparameters for the CAS strategy.
        """
        super().__init__(objective_function, hyper_parameters)
        
        self.delta_r = abs(self.hp.resolution.initial - self.hp.resolution.final) / self.hp.resolution.r_decay_steps
        self.model = None
        self.likelihood = None
        self.eci = None
        self.study = None

        self.metrics.new_custom_metrics(['r_parameter'])

    
    def run(self):
        """
        Executes the bCASTOR search algorithm.

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
                objective_function=self.objective_function,
                parallel=self.hp.parallel,
            )
        
        # Start rich progress logger
        progress = self.metrics.start_progress(description='bCASTOR Search ...')

        with progress:
            for i in progress.track(range(
                self.iteration, self.iteration + self.hp.total_iterations)):

                # Export objective function to CAS required format
                valid, constraints, bounds, train_x, train_y = obj_fn_export(
                    self.objective_function
                    )

                # Update current Objective Function metrics
                self.metrics.update(self.objective_function, i)

                # Initialise model and likelihood and train
                self.model, self.likelihood = get_model_and_likelihood(train_x, train_y)

                # Perform bCASTOR step
                self.eci, self.study, X, trials, values, cr = bcastor_step(
                    self.hp, self.objective_function.input_space_config, 
                    constraints, bounds, train_x, train_y, 
                    self.model, self.delta_r, i)

                # Update custom metrics, bCASTOR handles cr internally
                self.metrics.update_custom({"r_parameter": cr})

                # Evaluate points in objective function
                if self.hp.parallel is False:
                    for x in X:
                        self.objective_function.sample(x, True)
                else:
                    X = self.objective_function.space.inverse_transform(X)
                    results = run_x_with_pool(
                        X,
                        self.hp.batch_sampling.n_evaluation_workers,
                        self.objective_function.function
                        )
                    for sample_dict in results:
                        self.objective_function.add_sample_dict(sample_dict)                


                # Log and save
                self.metrics.log(progress)
                self.save_checkpoint(i)
                
                self.iteration = i

        return self.model, self.likelihood, self.eci, self.study, self.metrics
