import torch
import gpytorch
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior
import numpy as np 

from hepaid.search.models.base import Model
from hepaid.search.objective.transforms import TorchStandardize
tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = train_y.shape[-1]
        self.num_input_dim = train_x.shape[-1]
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([self.num_tasks ])
            )
        self.base_kernel = gpytorch.kernels.MaternKernel(
            nu=2.5,
            batch_shape=torch.Size([self.num_tasks]), 
            ard_num_dims=self.num_input_dim,
            lengthscale_prior=GammaPrior(3.0, 6.0),
            )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=self.base_kernel,
            batch_shape=torch.Size([self.num_tasks]), 
            ard_num_dims=self.num_input_dim,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

def predict(model, likelihood, test_x):
    # Set into eval mode
    model.eval()
    likelihood.eval()
    # Make predictions
    with torch.no_grad():
        observed_pred = likelihood(model(test_x))
    return observed_pred 




def train_gp(training_iterations, learning_rate, likelihood, model, train_x, train_y, 
             patience=10, min_delta=1e-4, T_max=50):
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = np.inf
    early_stopping_counter = 0

    for i in range(training_iterations):
        model.train()
        likelihood.train()

        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        optimizer.step()

        # Step the cosine annealing scheduler
        scheduler.step()

        # Check early stopping condition based on training loss
        if best_loss - loss.item() > min_delta:
            best_loss = loss.item()
            early_stopping_counter = 0  # Reset counter if improvement
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping at iteration {i+1}/{training_iterations} with training loss {loss.item():.4f}")
            break 

class MultiTaskGP(Model):
    def __init__(self, hyper_parameters = None ):
        super().__init__(hyper_parameters=hyper_parameters)

        self.training_iterations = self.hp.training_iterations
        self.learning_rate = self.hp.learning_rate

    def train(self, train_x, train_y):

        self.scaler_y = TorchStandardize()
        self.scaler_y.fit(train_y)
        train_y = self.scaler_y.transform(train_y)



        self.num_tasks = train_y.shape[-1]
        self.num_input_dim = train_x.shape[-1]
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks, 
            noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
        ).to(**tkwargs)
        self.model = BatchIndependentMultitaskGPModel(train_x, train_y, self.likelihood).to(**tkwargs)

        train_gp(
            self.training_iterations, self.learning_rate, self.likelihood, 
            self.model, train_x, train_y
            )
    
    def predict(self, test_x):
        prediction = predict(self.model, self.likelihood, test_x)
        mean = self.scaler_y.inverse_transform(prediction.mean)
        variance = self.scaler_y.inverse_transform(prediction.variance.sqrt())
        return mean, variance
    
def get_model_and_likelihood(train_x, train_y):
    gp = MultiTaksGP(500, 0.02)
    gp.train(train_x, train_y)
    return gp