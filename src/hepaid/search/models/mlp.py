import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hepaid.search.method.base import Method
from hepaid.search.objective import Objective
from typing import Callable, List, Optional, Tuple
from omegaconf import DictConfig

from hepaid.search.objective.utils import generate_initial_dataset
from hepaid.search.objective.utils import batch_evaluation

from hepaid.search.models.base import Model

def reshape_if_1d(array):
    array = np.asarray(array)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


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

class MultiLayerPerceptron(nn.Module):
    def __init__(self, layer_sizes, dropout_prob=None):
        super().__init__()
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


def train_model(
    num_epochs: int,
    model: torch.nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    best_loss: float = float('inf'),
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
        best_loss (float): Initial best loss for early stopping.

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

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

    return train_losses, val_losses



class MLP(Model):
    def __init__(self, hyper_parameters = None):
        super().__init__(hyper_parameters=hyper_parameters)
        # Initialise model
        self.model = MultiLayerPerceptron(
            layer_sizes=self.hp.layer_sizes,
            dropout_prob=self.hp.dropout_prob,
        )
        # Collect config hyper parameters
        self.lr = self.hp.learning_rate
        self.step_size = self.hp.step_size
        self.gamma = self.hp.gamma
        self.num_epochs = self.hp.num_epochs
        self.threshold = self.hp.threshold

        # Initialise loss
        self.criterion = nn.MSELoss()

    def train(self,train_x, train_y, val_x, val_y):

        # Reset optimiser and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            self.step_size,
            self.gamma
            )
        # Train loop
        self.tr_losses, self.val_losses = train_model(
            num_epochs=self.num_epochs,
            model=self.model,
            X_train=train_x,
            y_train=train_y,
            X_val=val_x,
            y_val=val_y,
            criterion=self.criterion,
            optimizer=self.optimizer,
            best_loss=float('inf')
        )


    def predict(self, test_x):
        # Set model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            output = self.model(test_x)
        return output
