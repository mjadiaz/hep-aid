from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import torch

def get_and_fit_gp(X, Y):
    """
    Simple method for creating and fitting a Gaussian Process (GP) with one output dimension.
    The function performs the following steps:

    Parameters:
        X (torch.Tensor): Input tensor of shape (n_samples, n_features), where the input space is assumed to be normalized to [0, 1]^d.
        Y (torch.Tensor): Output tensor of shape (n_samples, 1), representing the target values.

    Returns:
        gp (SingleTaskGP): A fitted SingleTaskGP model.
    """

    assert Y.ndim == 2 and Y.shape[-1] == 1
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 1e-3))  # Noise-free
    octf = Standardize(m=Y.shape[-1])
    gp = SingleTaskGP(X, Y, likelihood=likelihood, outcome_transform=octf)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp

def get_model_list_gp(X,Y):
    """
    Creates a list of Gaussian Process (GP) models, one for each output dimension, and combines them into a ModelListGP.

    Parameters:
        X (torch.Tensor): Input tensor of shape (n_samples, n_features), where the input space is assumed to be normalized to [0, 1]^d.
        Y (torch.Tensor): Output tensor of shape (n_samples, n_outputs), representing the target values for each output dimension.

    Returns:
        model_list_gp (ModelListGP): A ModelListGP containing individual GPs for each output dimension.

    """
    gp_models = [get_and_fit_gp(X, Y[:, i : i + 1]) for i in range(Y.shape[-1])]
    model_list_gp = ModelListGP(*gp_models)
    return model_list_gp

def get_posterior(model_list_gp, test_points):
    """
    Computes the posterior mean and confidence bounds for given test points using the List of Gaussian Processes.

    Parameters:
        model_list_gp (gpytorch.models.ExactGP): A list of Gaussian Process models.
        test_points (torch.Tensor): The points at which to compute the posterior.

    Returns:
        mean_prediction_points (torch.Tensor): The posterior mean predictions at the test points.
        lower (torch.Tensor): The lower confidence bounds (2 standard deviations below the mean) at the test points.
        upper (torch.Tensor): The upper confidence bounds (2 standard deviations above the mean) at the test points.
    """
    with torch.no_grad():
        # compute posterior
        posterior = model_list_gp.posterior(test_points)
        # Get upper and lower confidence bounds (2 standard deviations from the mean)
        lower, upper = posterior.mvn.confidence_region()
    mean_prediction_points = posterior.mean
    return mean_prediction_points, lower, upper

def get_model_and_likelihood(train_x, train_y):
    model = get_model_list_gp(train_x, train_y)
    return model, model.likelihood
    
def predict(model, x):
    mean, lower, upper = get_posterior(model, x)
    return mean, lower, upper 

