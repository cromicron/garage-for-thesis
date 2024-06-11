"""Value functions which use PyTorch."""
from garage.torch.value_functions.gaussian_mlp_value_function import (
    GaussianMLPValueFunction)
from garage.torch.value_functions.gamma_mlp_value_function import (
    GammaMLPValueFunction)
from garage.torch.value_functions.value_function import ValueFunction

__all__ = ['ValueFunction', 'GaussianMLPValueFunction', 'GammaMLPValueFunction']
