import torch
import torch.nn as nn
import numpy as np

from hepaid.search.method.base import Method
from hepaid.search.objective import Objective
from omegaconf import DictConfig
from typing import Callable, List, Optional, Tuple

from hepaid.search.objective.utils import generate_initial_dataset
from hepaid.search.objective.utils import batch_evaluation

from hepaid.search.models.mlp import MLP, train_model
from hepaid.search.models.mlp import reshape_if_1d, pre_process_data



def rejection_sampling(
    model: torch.nn.Module,
    likelihood: Callable,
    num_samples: int,
    input_dim: int,
    x_scaler,
    y_scaler,
    m_factor: float = 2.0,
    extra_random_samples: int = 10,
    max_iterations: int = 10000  # Added max_iterations to prevent infinite loops
) -> np.ndarray:
    """
    Perform rejection sampling to generate new samples based on a model and likelihood function.

    Parameters:
        model (torch.nn.Module): The trained model used for generating predictions.
        likelihood (Callable): A function that computes the likelihood of the model's outputs.
        num_samples (int): Number of samples to generate.
        input_dim (int): Dimension of the input features.
        x_scaler: Scaler used to inverse transform the model's outputs.
        y_scaler: Scaler used to inverse transform the model's outputs.
        m_factor (float): Factor to scale the acceptance probability. Assumes the maximum of the likelihood function will be 1.
        extra_random_samples (int): Number of extra random samples to generate.
        max_iterations (int): Maximum number of iterations to prevent infinite loops.

    Returns:
        np.ndarray: Array of accepted samples.
    """
    model.eval()
    accepted_samples: List[np.ndarray] = []
    iterations = 0

    while len(accepted_samples) < num_samples and iterations < max_iterations:
        iterations += 1

        # Generate random inputs
        inputs = np.random.uniform(0, 1, size=(num_samples, input_dim))
        inputs = torch.tensor(inputs, dtype=torch.float32)

        # Get model predictions
        with torch.no_grad():
            outputs = model(inputs)

        outputs = torch.tensor(y_scaler.inverse_transform(
            reshape_if_1d((outputs.numpy()))))
        likelihoods = likelihood(outputs).numpy()

        # Apply constraints with rejection sampling
        for i in range(num_samples):
            u = np.random.uniform(0, 1)
            if u <= likelihoods[i] / m_factor:  # Using u to simulate acceptance probability
                accepted_samples.append(inputs[i].numpy())
                if len(accepted_samples) >= num_samples:
                    break

    if len(accepted_samples) < num_samples:
        print(f"Warning: Reached maximum iterations ({max_iterations}) before collecting enough samples.")
        needed_samples = num_samples - len(accepted_samples)
        extra_samples = np.random.uniform(0, 1, size=(needed_samples, input_dim))
        accepted_samples.extend(extra_samples)

    x_extra_random = np.random.uniform(size=(extra_random_samples, input_dim))
    accepted_samples = np.array(accepted_samples)

    x_next = np.vstack([accepted_samples, x_extra_random])

    return x_next



class MLScan(Method):
    """
    """

    def __init__(self,
                 objective: Objective,
                 likelihood: Callable,
                 hyper_parameters: DictConfig | str | None = None,
                 ) -> None:
        """

        Parameters:
            objective(Objective): The objective function instance.
            hyper_parameters(DictConfig): Hyperparameters for the MCMC-MH algorithm.
            likelihood(Callable): Custom likelihood function. Default is None.
        """
        super().__init__(objective, hyper_parameters)

        self.likelihood = likelihood

        self.num_inputs = len(self.objective.input_parameters)
        self.num_outputs = len(self.objective.output_parameters)
        self.num_samples = self.hp.num_samples
        self.m_factor = self.hp.m_factor
        self.extra_random_samples = self.hp.extra_random_samples
        self.model =  MLP(
            hyper_parameters=self.hp.get('model_hyperparameters', None)
            )


        self.accepted = 0

        self.train_losses_history = []
        self.val_losses_history = []


    def run(self):
        """

        Returns:
            bool: True if the algorithm runs successfully.
        """
        if self.hp.initial_dataset.generate:
            generate_initial_dataset(
                n_workers=self.hp.n_workers,
                n_points=self.hp.initial_dataset.n_points,
                objective=self.objective,
                parallel=self.hp.parallel,
            )
        # Start rich progress logger
        progress = self.metrics.start_progress(description='MLScan Search ...')


        # Initial preprocessing to stablish the scalers
        initial_preprocess = pre_process_data(self.objective)
        X_train, X_val, y_train, y_val, x_scaler_init, y_scaler_init = initial_preprocess

        with progress:
            for i in progress.track(range(
                self.iteration, self.iteration + self.hp.total_iterations)):

                # Data scaling
                X_train, X_val, y_train, y_val, _, _= pre_process_data(
                    self.objective, x_scaler_init, y_scaler_init
                    )

                # Update current Objective Function metrics
                self.metrics.update(self.objective, i)

                self.model.train(X_train, y_train, X_val, y_val)


                accepted_samples = rejection_sampling(
                    self.model.model,
                    self.likelihood,
                    self.num_samples,
                    self.num_inputs,
                    x_scaler=x_scaler_init,
                    y_scaler=y_scaler_init,
                    m_factor=self.m_factor,
                    extra_random_samples=self.extra_random_samples
                    )

                # Evaluate new points
                batch_evaluation(
                    X=accepted_samples,
                    objective=self.objective,
                    parallel=self.hp.parallel,
                    n_evaluation_workers=self.hp.n_workers,
                    add=True,
                    is_normalised=True
                    )

                # Log and save
                self.metrics.log(progress)
                self.save_checkpoint(i)

                self.iteration = i
                self.train_losses_history.append(self.model.tr_losses)
                self.val_losses_history.append(self.model.val_losses)

        return True
