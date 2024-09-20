import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hepaid.search.method.base import Method
from typing import Callable, List, Optional, Tuple

def reshape_if_1d(array):
    array = np.asarray(array)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array

class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_prob=None):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # No activation or dropout on the output layer
            if i < len(layer_sizes) - 2:  
                layers.append(nn.ReLU())
                if dropout_prob is not None:
                    layers.append(nn.Dropout(dropout_prob))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

    
def pre_process_data(
    obj_fn,
    x_scaler = None,
    y_scaler = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Parameters:
        obj_fn: An object containing the dataset with attributes X and Y.
        x_scaler (Optional[StandardScaler]): Scaler for standardizing the data. If None, a new StandardScaler is created.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]: 
            - X_train: Training features tensor.
            - X_val: Validation features tensor.
            - y_train: Training labels tensor.
            - y_val: Validation labels tensor.
            - x_scaler: The scaler used for standardizing the data.
    """
    # Split and standardize the dataset
    X_train, X_val, y_train, y_val = train_test_split(obj_fn.x, obj_fn.Y, test_size=0.1, random_state=42)
    if x_scaler is None: 
        x_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_val = x_scaler.transform(X_val)
    if y_scaler is None:
        y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    return X_train, X_val, y_train, y_val, x_scaler, y_scaler



def train_model(
    num_epochs: int,
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_loss: float = float('inf'),
    threshold: float = 0.0,
    patience: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Train a model with early stopping and optional metric calculation.

    Parameters:
        num_epochs (int): Number of epochs to train the model.
        model (torch.nn.Module): The model to be trained.
        X_train (torch.Tensor): Training features.
        y_train (torch.Tensor): Training labels.
        X_val (torch.Tensor): Validation features.
        y_val (torch.Tensor): Validation labels.
        criterion (Callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        calculate_metrics (Optional[Callable]): Function to calculate metrics. Should return accuracy, precision, recall, and f1 score.
        best_loss (float): Initial best loss for early stopping.
        threshold (float): Threshold for improvement in validation loss to reset patience counter.
        patience (Optional[int]): Number of epochs to wait for improvement before stopping early.

    Returns:
        Tuple[List[float], List[float]]: Training and validation losses for each epoch.
    """
    patience_counter = 0

    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        # Check for early stopping
        if val_loss < best_loss - threshold:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience is not None:
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, LR: {scheduler.get_last_lr()}')

    return train_losses, val_losses

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
                 objective_function: Objective, 
                 hyper_parameters: DictConfig | str | None = None, 
                 likelihood: Callable | None = None,
                 ) -> None:
        """
        Initializes the MCMCMH class.

        Parameters:
            objective_function(Objective): The objective function instance.
            hyper_parameters(DictConfig): Hyperparameters for the MCMC-MH algorithm.
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
