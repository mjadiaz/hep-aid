import torch
import gpytorch


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = train_y.shape[-1]
        self.num_input_dim = train_x.shape[-1]
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.num_tasks ]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(batch_shape=torch.Size([self.num_tasks]), 
            ard_num_dims=self.num_input_dim),
            batch_shape=torch.Size([self.num_tasks ]), ard_num_dims=self.num_input_dim
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
    # Test points every 0.02 in [0,1]

    # Make predictions
    with torch.no_grad():

        observed_pred = likelihood(model(test_x))
        # Get mean
        mean = observed_pred.mean
        # Get lower and upper confidence bounds
        lower, upper = observed_pred.confidence_region()
    return observed_pred 


def train_gp(training_iterations, learning_rate,likelihood,model):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
       #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()


class MultiTaksGP:
    def __init__(self, training_iterations, learning_rate):
        self.training_iterations = training_iterations
        self.learning_rate = learning_rate
        
    def fit(self, train_x, train_y):
        self.num_tasks = train_y.shape[-1]
        self.num_input_dim = train_x.shape[-1]
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        )
        self.model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood)

        train_gp(self.training_iterations, self.learning_rate, self.likelihood, self.model)
    
    def predict(self, test_x):
        return predict(self.model, self.likelihood, test_x)