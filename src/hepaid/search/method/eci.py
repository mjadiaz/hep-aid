'''
Adapted from BoTorch implementation of Constraint Active Search
https://botorch.org/tutorials/constraint_active_search
'''


import os
import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.sampling import sample_hypersphere
from botorch.utils.transforms import t_batch_mode_transform

from hepaid.search.models.multitask_gp import predict
import warnings

warnings.filterwarnings("ignore")


tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}


def smooth_mask(x, a, eps=2e-3):
    """Returns 0ish for x < a and 1ish for x > a"""
    return torch.nn.Sigmoid()((x - a) / eps)


def smooth_box_mask(x, a, b, eps=2e-3):
    """Returns 1ish for a < x < b and 0ish otherwise"""
    return smooth_mask(x, a, eps) - smooth_mask(x, b, eps)


class ECI:
    """
    Modified Implementation of Expected Coverage Improvement.
    The surrogate model and the reference points are external.

    Adapted from BoTorch implementation of CAS: https://botorch.org/tutorials/constraint_active_search.

    Parameters:
        constraints (List[tuple]): Each tuple contains a direction ("gt" or "lt") and a threshold value.
        punchout_radius (float): The radius of the hyper-spheres used for measuring coverage.
        bounds (torch.Tensor): The bounds of the search space.
        train_x (torch.Tensor): Training data inputs.
        train_y (torch.Tensor): Training data outputs.
        num_samples (int, optional): Number of samples for Monte Carlo integration. Default is 512.

    Attributes:
        constraints (List[tuple]): Each tuple contains a direction ("gt" or "lt") and a threshold value.
        punchout_radius (float): The radius of the hyper-spheres used for measuring coverage.
        bounds (torch.Tensor): The bounds of the search space.
        train_x (torch.Tensor): Training data inputs.
        train_y (torch.Tensor): Training data outputs.
        base_points (torch.Tensor): Base points in the training data (train_x).
        train_inputs (torch.Tensor): Training data inputs (train_x).
        ball_of_points (torch.Tensor): Points generated for Monte Carlo integration.
        _thresholds (torch.Tensor): Thresholds extracted from the constraints.
    """
    def __init__(
        self,
        constraints,
        punchout_radius,
        bounds,
        train_x,
        train_y,
        num_samples=512,
    ):
        """
        Initialize Expected Coverage Improvement.

        Parameters:
            constraints (List[tuple]): Each tuple contains a direction ("gt" or "lt") and a threshold value.
            punchout_radius (float): The radius of the hyper-spheres used for measuring coverage.
            bounds (torch.Tensor): The bounds of the search space.
            train_x (torch.Tensor): Training data inputs.
            train_y (torch.Tensor): Training data outputs.
            num_samples (int, optional): Number of samples for Monte Carlo integration. Default is 512.
        """

        # assert len(constraints) == model.num_outputs
        assert all(direction in ("gt", "lt") for direction, _ in constraints)
        assert punchout_radius > 0
        self.constraints = constraints
        self.punchout_radius = punchout_radius
        self.bounds = bounds
        self.train_y = train_y
        self.base_points = train_x
        self.train_inputs = train_x
        self.ball_of_points = self._generate_ball_of_points(
            num_samples=num_samples,
            radius=punchout_radius,
            device=bounds.device,
            dtype=bounds.dtype,
        )
        self._thresholds = torch.tensor(
            [threshold for _, threshold in self.constraints]
        ).to(bounds)
        assert (
            all(ub > lb for lb, ub in self.bounds.T) and len(self.bounds.T) == self.dim
        )

    @property
    def num_outputs(self):
        return self.train_y.shape[-1]

    @property
    def dim(self):
        return self.base_points.shape[-1]

    def _generate_ball_of_points(
        self, num_samples, radius, device=None, dtype=torch.double
    ):
        """
        Creates a ball of points to be used for Monte Carlo integration
        to keep them fix through the ECI estimation.

        Parameters:
            num_samples (int): Number of samples to generate.
            radius (float): Radius of the hyper-spheres.
            device (torch.device, optional): Device for the tensor.
            dtype (torch.dtype, optional): Data type for the tensor.

        Returns:
            torch.Tensor: A tensor of generated points within the hyper-spheres.
        """

        tkwargs = {"device": device, "dtype": dtype}
        z = sample_hypersphere(
            d=self.train_inputs.shape[-1], n=num_samples, qmc=True, **tkwargs
        )
        r = torch.rand(num_samples, 1, **tkwargs) ** (1 / self.dim)
        return radius * r * z

    def _estimate_probabilities_of_satisfaction_at_points(
        self, surrogate_model, points, scaler=None
    ):
        """
        Estimate the probability of satisfying the given constraints.

        Parameters:
            surrogate_model (Any): The surrogate model used for estimation.
            points (torch.Tensor): Points at which to estimate probabilities.
            scaler (Callable, optional): Scaler to transform the outputs.

        Returns:
            torch.Tensor: Probabilities of satisfying the constraints at the given points.
        """
        #posterior = surrogate_model.posterior(X=points)
        mus, sigma2s = surrogate_model.predict(points)
        #mus, sigma2s = posterior.mean, posterior.variance
        if scaler is not None:
            mus = torch.tensor(scaler.inverse_transform(mus.detach())).to(points)
            sigma2s = torch.tensor(scaler.inverse_transform(sigma2s.detach())).to(
                sigma2s
            )
        dist = torch.distributions.normal.Normal(mus, sigma2s)
        norm_cdf = dist.cdf(self._thresholds)
        probs = torch.ones(points.shape[:-1]).to(points)
        for i, (direction, _) in enumerate(self.constraints):
            probs = probs * (
                norm_cdf[..., i] if direction == "lt" else 1 - norm_cdf[..., i]
            )
        return probs

    def __call__(self, X, reference_points, surrogate_model, scaler=None):
        """
        Evaluate Expected Coverage Improvement on the candidate point X with respect to reference points.

        Parameters:
            X (torch.Tensor): Candidate point for evaluation.
            reference_points (torch.Tensor): Reference points for comparison.
            surrogate_model (Any): The surrogate model used for estimation.
            scaler (Callable, optional): Scaler to transform the outputs.

        Returns:
            torch.Tensor: Expected Coverage Improvement value for the candidate point.
        """
        ball_around_X = self.ball_of_points + X
        domain_mask = smooth_box_mask(
            ball_around_X, self.bounds[0, :], self.bounds[1, :]
        ).prod(dim=-1)
        num_points_in_integral = domain_mask.sum(dim=-1)
        # base_point_mask = self._get_base_point_mask(ball_around_X).prod(dim=-1)
        base_point_mask = smooth_mask(
            covar_dist(X, reference_points), self.punchout_radius
        ).prod(dim=-1)
        prob = self._estimate_probabilities_of_satisfaction_at_points(
            surrogate_model, ball_around_X, scaler=scaler
        )
        masked_prob = prob * domain_mask * base_point_mask
        y = masked_prob.sum(dim=-1) / num_points_in_integral
        return y


class ExpectedCoverageImprovement(MCAcquisitionFunction):
    def __init__(
        self,
        model,
        constraints,
        punchout_radius,
        bounds,
        num_samples=128,
        **kwargs,
    ):
        """
        Expected Coverage Improvement (q=1 required, analytic)
        Adapted from BoTorch implementation of CAS: https://botorch.org/tutorials/constraint_active_search.

        Right now, we assume that all the models in the ModelListGP have
        the same training inputs.

        Args:
            model: A ModelListGP object containing models matching the corresponding constraints.
                All models are assumed to have the same training data.
            constraints: List containing 2-tuples with (direction, value), e.g.,
                [('gt', 3), ('lt', 4)]. It is necessary that
                len(constraints) == model.num_outputs.
            punchout_radius: Positive value defining the desired minimum distance between points
            bounds: torch.tensor whose first row is the lower bounds and second row is the upper bounds
            num_samples: Number of samples for MC integration
        """
        super().__init__(model=model, objective=IdentityMCObjective(), **kwargs)
        assert len(constraints) == model.num_outputs
        assert all(direction in ("gt", "lt") for direction, _ in constraints)
        assert punchout_radius > 0
        self.constraints = constraints
        self.punchout_radius = punchout_radius
        self.bounds = bounds
        self.base_points = self.train_inputs
        self.ball_of_points = self._generate_ball_of_points(
            num_samples=num_samples,
            radius=punchout_radius,
            device=bounds.device,
            dtype=bounds.dtype,
        )
        self._thresholds = torch.tensor(
            [threshold for _, threshold in self.constraints]
        ).to(bounds)
        assert (
            all(ub > lb for lb, ub in self.bounds.T) and len(self.bounds.T) == self.dim
        )

    @property
    def num_outputs(self):
        return self.model.num_outputs

    @property
    def dim(self):
        return self.train_inputs.shape[-1]

    @property
    def train_inputs(self):
        return self.model.models[0].train_inputs[0]

    def _generate_ball_of_points(
        self, num_samples, radius, device=None, dtype=torch.double
    ):
        """Creates a ball of points to be used for MC."""
        tkwargs = {"device": device, "dtype": dtype}
        z = sample_hypersphere(d=self.dim, n=num_samples, qmc=True, **tkwargs)
        r = torch.rand(num_samples, 1, **tkwargs) ** (1 / self.dim)
        return radius * r * z

    def _get_base_point_mask(self, X):
        distance_matrix = self.model.models[0].covar_module.base_kernel.covar_dist(
            X, self.base_points
        )
        return smooth_mask(distance_matrix, self.punchout_radius)

    def _estimate_probabilities_of_satisfaction_at_points(self, points):
        """Estimate the probability of satisfying the given constraints."""
        posterior = self.model.posterior(X=points)
        mus, sigma2s = posterior.mean, posterior.variance
        dist = torch.distributions.normal.Normal(mus, sigma2s.sqrt())
        norm_cdf = dist.cdf(self._thresholds)
        probs = torch.ones(points.shape[:-1]).to(points)
        for i, (direction, _) in enumerate(self.constraints):
            probs = probs * (
                norm_cdf[..., i] if direction == "lt" else 1 - norm_cdf[..., i]
            )
        return probs

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Evaluate Expected Improvement on the candidate set X."""
        ball_around_X = self.ball_of_points + X
        domain_mask = smooth_box_mask(
            ball_around_X, self.bounds[0, :], self.bounds[1, :]
        ).prod(dim=-1)
        num_points_in_integral = domain_mask.sum(dim=-1)
        base_point_mask = self._get_base_point_mask(ball_around_X).prod(dim=-1)
        prob = self._estimate_probabilities_of_satisfaction_at_points(ball_around_X)
        masked_prob = prob * domain_mask * base_point_mask
        y = masked_prob.sum(dim=-1) / num_points_in_integral
        return y



def sq_dist(x1, x2, x1_eq_x2=False):
    """Equivalent to the square of `torch.cdist` with p=2."""
    # TODO: use torch squared cdist once implemented: https://github.com/pytorch/pytorch/pull/25799
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment

    # Compute squared distance matrix using quadratic expansion
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        x2, x2_norm, x2_pad = x1, x1_norm, x1_pad
    else:
        x2 = (
            x2 - adjustment
        )  # x1 and x2 should be identical in all dims except -2 at this point
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        res.diagonal(dim1=-2, dim2=-1).fill_(0)

    # Zero out negative values
    return res.clamp_min_(0)


def dist(x1, x2, x1_eq_x2=False):
    """
    Equivalent to `torch.cdist` with p=2, but clamps the minimum element to 1e-15.
    """
    if not x1_eq_x2:
        res = torch.cdist(x1, x2)
        return res.clamp_min(1e-15)
    res = sq_dist(x1, x2, x1_eq_x2=x1_eq_x2)
    return res.clamp_min_(1e-30).sqrt_()


def covar_dist(
    x1: torch.Tensor,
    x2: torch.Tensor,
    diag: bool = False,
    last_dim_is_batch: bool = False,
    square_dist: bool = False,
    **params,
) -> torch.Tensor:
    r"""
    This is a helper method for computing the Euclidean distance between
    all pairs of points in :math:`\mathbf x_1` and :math:`\mathbf x_2`.

    :param x1: First set of data (... x N x D).
    :param x2: Second set of data (... x M x D).
    :param diag: Should the Kernel compute the whole kernel, or just the diag?
        If True, it must be the case that `x1 == x2`. (Default: False.)
    :param last_dim_is_batch: If True, treat the last dimension
        of `x1` and `x2` as another batch dimension.
        (Useful for additive structure over the dimensions). (Default: False.)
    :param square_dist:
        If True, returns the squared distance rather than the standard distance. (Default: False.)
    :return: The kernel matrix or vector. The shape depends on the kernel's evaluation mode:

        * `full_covar`: `... x N x M`
        * `full_covar` with `last_dim_is_batch=True`: `... x K x N x M`
        * `diag`: `... x N`
        * `diag` with `last_dim_is_batch=True`: `... x K x N`
    """
    if last_dim_is_batch:
        x1 = x1.transpose(-1, -2).unsqueeze(-1)
        x2 = x2.transpose(-1, -2).unsqueeze(-1)

    x1_eq_x2 = torch.equal(x1, x2)
    res = None

    if diag:
        # Special case the diagonal because we can return all zeros most of the time.
        if x1_eq_x2:
            return torch.zeros(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        else:
            res = torch.linalg.norm(x1 - x2, dim=-1)  # 2-norm by default
            return res.pow(2) if square_dist else res
    else:
        dist_func = sq_dist if square_dist else dist
        return dist_func(x1, x2, x1_eq_x2)
