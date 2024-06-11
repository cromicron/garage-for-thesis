"""A value function based on a GaussianMLP model."""
import torch
from torch import nn

from garage.torch.modules import GammaMLPModule
from garage.torch.value_functions.value_function import ValueFunction



class GammaMLPValueFunction(ValueFunction):
    """Gamma MLP Value Function with Model.

    It fits the input data to a gamma distribution estimated by
    a MLP.

    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): The name of the value function.

    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=nn.functional.softplus,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_scale=True,
                 init_scale=0.01,
                 layer_normalization=False,
                 name='GammaMLPValueFunction',
                 normalize_inputs=True,
                 normalize_outputs=False,
                 load_weights=False,
                 weights_dir=None,
                 ):
        super(GammaMLPValueFunction, self).__init__(env_spec, name)

        input_dim = env_spec.observation_space.flat_dim
        output_dim = 1

        self.module = GammaMLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_scale=learn_scale,
            init_scale=init_scale,
            min_scale=None,
            max_scale=None,
            scale_parameterization='exp',
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
        ll = dist.log_prob(ys.reshape(-1, 1) +1e-8) # gamma distribution does not allow for exactly 0
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

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
