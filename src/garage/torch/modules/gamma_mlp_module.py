"""GaussianMLPModule."""
import abc

import torch
from torch import nn
from torch.distributions import Gamma
from torch.distributions.independent import Independent
from garage.torch.distributions import TanhNormal
from garage.torch.modules.mlp_module import MLPModule
from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule


class GammaMLPBaseModule(nn.Module):
    """Base of GammaMLPModel.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
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
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_hidden_w_init (callable):  Initializer function for the weight
            of hidden layer (s).
        std_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s).
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation.
            - softplus: the std will be computed as log(1+exp(x)).
        layer_normalization (bool): Bool for using layer normalization or not.
        gamma_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 *,
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=nn.functional.softplus,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_scale=True,
                 init_scale=0.5,
                 min_scale=1e-6,
                 max_scale=None,
                 scale_hidden_sizes=(32, 32),
                 scale_hidden_nonlinearity=torch.tanh,
                 scale_hidden_w_init=nn.init.xavier_uniform_,
                 scale_hidden_b_init=nn.init.zeros_,
                 scale_output_nonlinearity=None,
                 scale_output_w_init=nn.init.xavier_uniform_,
                 scale_parameterization='exp',
                 layer_normalization=False,
                 ):
        super().__init__()

        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = output_dim
        self._learn_scale = learn_scale
        self._scale_hidden_sizes = scale_hidden_sizes
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._scale_hidden_nonlinearity = scale_hidden_nonlinearity
        self._scale_hidden_w_init = scale_hidden_w_init
        self._scale_hidden_b_init = scale_hidden_b_init
        self._scale_output_nonlinearity = scale_output_nonlinearity
        self._scale_output_w_init = scale_output_w_init
        self._scale_parameterization = scale_parameterization
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization

        if self._scale_parameterization not in ('exp', 'softplus'):
            raise NotImplementedError

        self._init_scale = torch.Tensor([init_scale]).log()
        log_scale = torch.Tensor([init_scale] * output_dim).log()
        if self._learn_scale:
            self._log_scale = torch.nn.Parameter(log_scale)
        else:
            self._log_scale = log_scale
            self.register_buffer('log_scale', self._log_scale)

        self._min_scale_param = self._max_scale_param = None
        if min_scale is not None:
            self._min_scale_param = torch.Tensor([min_scale]).log()
            self.register_buffer('min_scale_param', self._min_scale_param)
        if max_scale is not None:
            self._max_scale_param = torch.Tensor([max_scale]).log()
            self.register_buffer('max_scale_param', self._max_scale_param)

    def to(self, *args, **kwargs):
        """Move the module to the specified device.

        Args:
            *args: args to pytorch to function.
            **kwargs: keyword args to pytorch to function.

        """
        super().to(*args, **kwargs)
        buffers = dict(self.named_buffers())
        if not isinstance(self._log_scale, torch.nn.Parameter):
            self._log_scale = buffers['log_scale']
        if "min_scale_param" in buffers:
            self._min_scale_param = buffers['min_scale_param']
        if "max_scale_param" in buffers:
            self._max_scale_param = buffers['max_scale_param']

    @abc.abstractmethod
    def _get_shape_and_log_scale(self, *inputs):
        pass

    def forward(self, *inputs):
        """Forward method.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.distributions.independent.Independent: Independent
                distribution.

        """
        shape, log_scale_uncentered = self._get_shape_and_log_scale(*inputs)

        if self._min_scale_param or self._max_scale_param:
            log_scale_uncentered = log_scale_uncentered.clamp(
                min=(None if self._min_scale_param is None else
                     self._min_scale_param.item()),
                max=(None if self._max_scale_param is None else
                     self._max_scale_param.item()))

        if self._scale_parameterization == 'exp':
            scale = log_scale_uncentered.exp()
        else:
            scale = log_scale_uncentered.exp().exp().add(1.).log()
        dist = Gamma(shape, 1/scale)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist


class GammaMLPModule(GammaMLPBaseModule):
    """GammaMLPModule that shape and scale share the same network.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for shape. For example, (32, 32) means the MLP consists
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
        learn_scale (bool): Is std trainable.
        init_scale (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_scale (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_scale (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        scale_parameterization (str): How the scale should be parametrized. There
            are two options:
            - exp: the logarithm of the scale will be stored, and applied a
               exponential transformation
            - softplus: the scale will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 *,
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=nn.functional.softplus,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_scale=True,
                 init_scale=0.1,
                 min_scale=1e-6,
                 max_scale=None,
                 scale_parameterization='exp',
                 layer_normalization=False,
                 ):
        super().__init__(input_dim=input_dim,
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
                         min_scale=min_scale,
                         max_scale=max_scale,
                         scale_parameterization=scale_parameterization,
                         layer_normalization=layer_normalization,
                         )

        self._shape_module = MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization)

    # pylint: disable=arguments-differ
    def _get_shape_and_log_scale(self, x):
        """Get shape and scale of Gamma distribution given inputs.

        Args:
            x: Input to the module.

        Returns:
            torch.Tensor: The shape of Gamma distribution.
            torch.Tensor: The scale of Gamma distribution.

        """
        shape = self._shape_module(x)

        return shape, self._log_scale

