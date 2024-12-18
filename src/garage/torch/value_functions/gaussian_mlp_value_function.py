"""A value function based on a GaussianMLP model."""
import torch
from torch import nn

from garage.torch.modules import GaussianMLPModule
from garage.torch.value_functions.value_function import ValueFunction


class GaussianMLPValueFunction(ValueFunction):
    """Gaussian MLP Value Function with a model-based approach.

    This value function model fits input data to a Gaussian distribution
    estimated by a multi-layer perceptron (MLP).

    Args:
        env_spec (EnvSpec): Environment specification detailing observation and action spaces.
        hidden_sizes (list[int]): Output dimensions of dense layers in the MLP for the mean.
            For example, (32, 32) means the MLP has two hidden layers, each with 32 units.
        hidden_nonlinearity (callable): Activation function for intermediate dense layers.
            Should return a torch.Tensor. Set to None for linear activation.
        hidden_w_init (callable): Initializer for the weights of intermediate dense layers.
            Should return a torch.Tensor.
        hidden_b_init (callable): Initializer for the biases of intermediate dense layers.
            Should return a torch.Tensor.
        output_nonlinearity (callable): Activation function for the output layer.
            Should return a torch.Tensor. Set to None for linear activation.
        output_w_init (callable): Initializer for the weights of the output layer.
            Should return a torch.Tensor.
        output_b_init (callable): Initializer for the biases of the output layer.
            Should return a torch.Tensor.
        learn_std (bool): Whether the standard deviation is trainable.
        init_std (float): Initial value for standard deviation (not log-transformed or exponentiated).
        layer_normalization (bool): Whether to apply layer normalization.
        name (str): Name of the value function.
        normalize_inputs (bool): Whether to normalize input observations.
        normalize_outputs (bool): Whether to normalize output values.
        load_weights (bool): Whether to load pre-trained weights from disk.
        weights_dir (str, optional): Directory path for loading and saving model weights.
            Defaults to "saved_models/rl_2_value_funct.pth".

    Attributes:
        module (GaussianMLPModule): The core MLP module for estimating the Gaussian distribution.
        x_mean (float): Mean for input normalization, if applied.
        x_std (float): Standard deviation for input normalization, if applied.
        y_mean (float): Mean for output normalization, if applied.
        y_std (float): Standard deviation for output normalization, if applied.
    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 layer_normalization=False,
                 name='GaussianMLPValueFunction',
                 normalize_inputs=True,
                 normalize_outputs=True,
                 load_weights=False,
                 weights_dir=None,
                 ):
        super(GaussianMLPValueFunction, self).__init__(env_spec, name)

        input_dim = env_spec.observation_space.flat_dim
        output_dim = 1

        self.module = GaussianMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            min_std=None,
            max_std=None,
            std_parameterization='exp',
            layer_normalization=layer_normalization)
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.x_mean = 0
        self.x_std = 1
        self.y_mean = 0
        self.y_std = 1
        if weights_dir is None:
            self.weights_dir = "saved_models/rl_2_value_funct.pth"
        else:
            self.weights_dir = weights_dir
        self.load_weights_from_disc = load_weights
        if load_weights:
            self.load_weights()


    def compute_loss(self, obs, returns):
        r"""Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).

        """
        x = (obs - self.x_mean) / self.x_std
        dist = self.module(x)
        ys = (returns - self.y_mean)/self.y_std
        ll = dist.log_prob(ys.reshape(-1, 1))
        loss = -ll.mean()
        return loss

    # pylint: disable=arguments-differ
    def forward(self, obs):
        r"""Predict value based on paths.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(P, O*)`.

        Returns:
            torch.Tensor: Calculated baselines given observations with
                shape :math:`(P, O*)`.

        """
        x = (obs - self.x_mean) / self.x_std
        return self.module(x).mean.flatten(-2)*self.y_std + self.y_mean

    def save_weights(self):
        params = self.module.state_dict()
        torch.save(params, self.weights_dir)

    def load_weights(self):
        params = torch.load(self.weights_dir)
        self.module.load_state_dict(params)
