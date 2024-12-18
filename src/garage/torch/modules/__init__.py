"""PyTorch Modules."""
# yapf: disable
# isort:skip_file
from garage.torch.modules.cnn_module import CNNModule
from garage.torch.modules.gaussian_mlp_module import (
    GaussianMLPIndependentStdModule)  # noqa: E501
from garage.torch.modules.gaussian_mlp_module import (
    GaussianMLPTwoHeadedModule)  # noqa: E501
from garage.torch.modules.gaussian_mlp_module import GaussianMLPModule
from garage.torch.modules.gaussian_gru_module import GaussianGRUModule
from garage.torch.modules.gru_encoder import GRUEncoder
from garage.torch.modules.gamma_mlp_module import GammaMLPModule
from garage.torch.modules.hypernetwork import HyperNetwork
from garage.torch.modules.mlp_module import MLPModule
from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule
# DiscreteCNNModule must go after MLPModule
from garage.torch.modules.discrete_cnn_module import DiscreteCNNModule
# yapf: enable

__all__ = [
    'CNNModule',
    'DiscreteCNNModule',
    'MLPModule',
    'MultiHeadedMLPModule',
    'GaussianMLPModule',
    'GaussianGRUModule',
    'GammaMLPModule',
    'GaussianMLPIndependentStdModule',
    'GaussianMLPTwoHeadedModule',
    'GRUEncoder',
    'HyperNetwork'
]
