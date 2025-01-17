import torch
from torch import nn

AVAILABLE_DATASETS = ["mnist"]

class MLPMnist(nn.Module):
    """A simple MLP model for MNIST dataset with customizable activation functions."""

    def __init__(self, activation_fn: str = "relu") -> None:
        """Initialize the MLP model.

        Args:
            activation_fn: The activation function to use (relu, sigmoid, tanh, leaky_relu, linear).
        """
        super(MLPMnist, self).__init__()

        # Define available activation functions
        activation_functions = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "linear": nn.Identity(),
        }

        if activation_fn not in activation_functions:
            raise ValueError(f"Unknown activation function: {activation_fn}. Available options are: {list(activation_functions.keys())}")

        self.activation = activation_functions[activation_fn]

        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP model.

        Args:
            x: The input image tensor.

        Returns:
            (torch.Tensor): The output tensor.
        """
        x = x.view(-1, 28*28)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


def create_mlp_model(dataset: str, **kwargs) -> nn.Module:
    """Create a MLP model.

    Args:
        dataset: The dataset to create the model for.
        **kwargs: The arguments to pass to the model constructor.

    Returns:
        The MLP model.

    Raises:
        ValueError: If the dataset is unknown.
    """
    if dataset == "mnist":
        return MLPMnist(**kwargs)
    message = f"Unknown dataset: {dataset}, available: {AVAILABLE_DATASETS}"
    raise ValueError(message)