import torch
import torch.nn as nn
import numpy as np
#from garage.torch.policies.stochastic_policy import Policy



def reshape_params(parameters: torch.Tensor, input_dim: int, output_dim: int):
    """
    Reshape the parameters tensor into weights and biases for linear layers.

    Args:
    parameters (torch.Tensor): The parameters tensor with shape [batch_size * epoch_length, features],
                               where features include both weights and biases.
    input_dim (int): Number of input features to the layer.
    output_dim (int): Number of output features from the layer.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: weights and biases tensors.
    """
    num_weights = input_dim * output_dim
    weights_params = parameters[..., :num_weights]
    biases = parameters[..., num_weights:]
    weights = weights_params.reshape(*weights_params.shape[:-1], output_dim, input_dim)



    return weights, biases


class Mainnet:
    def __init__(self, input_size, layers, params, activation=None):
        if activation is None:
            self._activation = lambda x : x
        else:
            self._activation = activation
        self._weights = []
        self._biases = []
        input_dim = input_size
        index = 0
        for layer_size in layers:
            n_param_layer = (1+ input_dim) * layer_size
            parameters = params[..., index: index + n_param_layer]
            w, b = reshape_params(parameters, input_dim, layer_size)
            self._weights.append(w)
            self._biases.append(b)
            input_dim = layer_size
            index += n_param_layer


    def forward(self, x):
        for weights, biases in zip(self._weights[: -1], self._biases[: -1]):
            x = torch.matmul(weights, x.unsqueeze(-1)).squeeze() + biases
            x = self._activation(x)
        output = torch.matmul(self._weights[-1], x.unsqueeze(-1)).squeeze() + self._biases[-1]
        return output
class HyperNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        input_size,
        layer_sizes,
        nonlinearity=lambda x: x,
        mainnet_nonlinearity=lambda x: x,
    ):
        super(HyperNetwork, self).__init__()
        self._input_dim = input_dim
        self._input_size = input_size
        self._layer_sizes = layer_sizes
        self._nonlinearity = nonlinearity
        self._mainnet_nonlinearity = mainnet_nonlinearity
        self._n_params = 0
        n_inputs = self._input_size
        for layer_size in layer_sizes:
            n_param_layer = (1 + n_inputs) * layer_size
            self._n_params += n_param_layer
            n_inputs = layer_size
        self._fc = nn.Linear(input_dim, self._n_params)
        nn.init.zeros_(self._fc.weight)

    def forward(self, x):
        p = self._nonlinearity(self._fc(x))
        mainnet = Mainnet(
            self._input_size,
            self._layer_sizes,
            p,
            activation=self._mainnet_nonlinearity
        )
        return mainnet

