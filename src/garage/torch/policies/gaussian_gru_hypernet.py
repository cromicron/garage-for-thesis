import torch
import torch.nn as nn
import numpy as np
from garage.torch.policies.stochastic_policy import Policy



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
    num_biases = output_dim

    # Split the parameters into weights and biases
    weights = parameters[:, :num_weights].reshape(-1, output_dim, input_dim)
    biases = parameters[:, num_weights:].reshape(-1, output_dim)

    return weights, biases

def apply_layers(input_tensor: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor):
    """
    Apply a series of linear transformations to the input tensor.

    Args:
    input_tensor (torch.Tensor): Input data tensor with shape [batch_size, epoch_length, input_dim].
    weights (torch.Tensor): Weights for each layer, shaped [batch_size * epoch_length, output_dim, input_dim].
    biases (torch.Tensor): Biases for each layer, shaped [batch_size * epoch_length, output_dim].

    Returns:
    torch.Tensor: The result of applying the linear layers, shape [batch_size, epoch_length, output_dim].
    """
    batch_size, epoch_length, input_dim = input_tensor.shape
    output_dim = weights.shape[1]

    # Reshape input tensor to match the operation
    reshaped_input = input_tensor.reshape(batch_size * epoch_length, input_dim)

    # Apply each linear layer
    outputs = torch.bmm(weights, reshaped_input.unsqueeze(2)).squeeze(2) + biases

    # Reshape output to match original batch structure
    return outputs.reshape(batch_size, epoch_length, output_dim)


class GaussianHyperGru(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        policy_input_dim,
        hidden_dim=32,
        latent_dim=8,
        policy_dim=16,
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
        min_std=None,
        max_std=None,
    ):
        super(GaussianHyperGru, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.policy_input_dim = policy_input_dim
        self.policy_dim = policy_dim
        self.hidden_dim = hidden_dim
        self.hidden_nonlinearity = hidden_nonlinearity
        if output_nonlinearity is None:
            output_nonlinearity = lambda x: x
        self.output_nonlinearity = output_nonlinearity
        if min_std is not None and max_std is not None:
            self.log_min_std = torch.log(torch.tensor(min_std))
            self.log_max_std = torch.log(torch.tensor(max_std))
        else:
            self.log_min_std, self.log_max_std = None, None

        # Define GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.n_weights_hidden = policy_input_dim * policy_dim
        self.n_biases_hidden = policy_dim
        self.n_weights_out = policy_dim * output_dim * 2
        self.n_biases_out = output_dim * 2
        self.n_params_total = sum([
            self.n_weights_hidden,
            self.n_biases_hidden,
            self.n_weights_out,
            self.n_biases_out]
        )
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        self.fc_param = nn.Linear(latent_dim, self.n_params_total)
        nn.init.constant_(self.fc_param.weight, 0)


    def forward(self, x, hidden=None):
        out_rnn, hidden = self.gru(x,hidden)
        task_encoding = self.fc_latent(out_rnn)
        params = self.fc_param(task_encoding)

        w_h = params[..., : self.n_weights_hidden]
        i_biases_hidden = self.n_weights_hidden +self.n_biases_hidden
        b_h = params[..., self.n_weights_hidden: i_biases_hidden]
        i_weights_out = i_biases_hidden + self.n_weights_out
        w_o = params[..., i_biases_hidden: i_weights_out]
        b_o = params[..., i_weights_out:]
        states = x[..., : self.policy_input_dim]

        w_h_mat = w_h.reshape(
            (*w_h.shape[: -1], self.policy_dim, self.policy_input_dim)
        )
        w_o_mat = w_o.reshape(
            (*w_o.shape[: -1], self.output_dim * 2, self.policy_dim)
        )
        out_hidden = self.hidden_nonlinearity(
            torch.matmul(
                w_h_mat, states.unsqueeze(-1)) + b_h.unsqueeze(-1))
        output_mlp = (
            torch.matmul(w_o_mat, out_hidden) + b_o.unsqueeze(-1)
        ).squeeze(-1)
        means_stds = self.output_nonlinearity(output_mlp)
        mean = means_stds[..., : self.output_dim]
        log_std = means_stds[..., self.output_dim: ]

        if self.log_min_std  and self.log_max_std:
            def _softclip(x, x_min, x_max, alpha=2):
                y_scale = (x_max - x_min) / 2
                y_offset = (x_max + x_min) / 2
                x_scale = (2 * alpha) / (x_max - x_min)
                x_offset = (x_max + x_min) / 2
                return (torch.tanh((x - x_offset) * x_scale) * y_scale) + y_offset
            log_std = _softclip(log_std, self.log_min_std, self.log_max_std)
        std = torch.exp(log_std)
        distribution = torch.distributions.Normal(mean, std)

        return distribution, hidden

class GaussianHyperGRUPolicy(Policy):
    def __init__(
        self,
        env_spec,
        hidden_dim=32,
        policy_input_dim=39,
        policy_dim=16,
        name='GaussianHyperGRUPolicy',
        min_std=None,
        max_std=None,
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
        state_include_action=True,
        load_weights=False,
        constraints=0,
    ):
        super().__init__(env_spec=env_spec, name=name)
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._policy_input_dim = policy_input_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._hidden_dim = hidden_dim
        self._policy_dim = policy_dim
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._min_std = min_std
        self._max_std = max_std
        self._state_include_action = state_include_action


        if state_include_action:
            self._input_dim = self._obs_dim + self._action_dim
        else:
            self._input_dim = self._obs_dim


        self._prev_actions = None
        self._prev_hiddens = None
        self._init_hidden = torch.zeros(1, hidden_dim, dtype=torch.float64)

        self._module = GaussianHyperGru(
            input_dim=self._input_dim,
            policy_input_dim=self._policy_input_dim,
            policy_dim=self._policy_dim,
            output_dim=self._action_dim,
            hidden_dim=self._hidden_dim,
            min_std=self._min_std,
            max_std=self._max_std,
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._output_nonlinearity
        )

        self.weights_dir = "saved_models/rl_2_gru.pth"
        if load_weights:
            self.load_weights()

    def reset(self, do_resets=None, dtype=torch.float32):
        if do_resets is None:
            do_resets = np.array([True])

        # Directly fetch the device from the model's existing parameters
        device = next(self.parameters()).device

        # Ensure all tensors are created on the correct device
        if self._prev_actions is None or len(do_resets) != len(
            self._prev_actions):
            self._prev_actions = np.zeros(
                (len(do_resets), self._env_spec.action_space.flat_dim),
            )
            self._prev_hiddens = torch.zeros(
                (1, len(do_resets), self._hidden_dim),
                device=device,
                dtype=dtype,
            )

        # Convert do_resets to a torch tensor and ensure it is on the same device
        do_resets_torch = torch.from_numpy(do_resets.astype(bool)).to(device)

        self._prev_actions[do_resets] = 0.
        self._prev_hiddens = self._prev_hiddens.to(
            device=device, dtype=dtype)
        self._prev_hiddens[:, do_resets_torch] = self._init_hidden.to(
            device=device, dtype=dtype)

    def forward(self, inputs):
        return self._module.forward(inputs,  self._prev_hiddens)

    def get_action(self, observation):
        with torch.no_grad():
            action, info = self.get_actions(observation.reshape(1, 1, -1))
        return action.squeeze(), {k: v.squeeze() for k, v in info.items()}


    def get_actions(self, observations):
        if self._state_include_action:
            assert self._prev_actions is not None
            all_input = np.concatenate([observations, self._prev_actions], axis=-1)
        else:
            all_input = observations
        all_input_tensor = torch.from_numpy(all_input).float()
        dist, hidden_vec = self.forward(all_input_tensor)
        samples = dist.sample().numpy()
        samples = self._env_spec.action_space.unflatten_n(samples)
        prev_actions = self._prev_actions
        self._prev_actions = samples
        self._prev_hiddens = hidden_vec
        info = dict(mean=dist.mean, log_std=torch.log(dist.stddev))
        info = {k: v.detach().numpy() for k, v in info.items()}
        if self._state_include_action:
            info['prev_action'] = np.copy(prev_actions)
        return samples, info

    @property
    def state_info_specs(self):
        if self._state_include_action:
            return [('prev_action', (self._action_dim,))]
        return []

    def clone(self, name):
        new_policy = self.__class__(
            env_spec=self._env_spec,
            hidden_dim=self._hidden_dim,
            policy_input_dim=self._policy_input_dim,
            policy_dim=self._policy_dim,
            name=name,
            min_std=self._min_std,
            max_std=self._max_std,
            output_nonlinearity=self._output_nonlinearity,
            state_include_action=self._state_include_action,
        )
        new_policy.load_parameters(self.get_parameters())
        return new_policy


    def load_parameters(self, new_parameters):
        """Load model parameters from a dictionary object."""
        self._module.load_state_dict(new_parameters)

    def get_parameters(self):
        """Retrieve current model parameters."""
        return self._module.state_dict()

    def save_weights(self):
        """Save the current model parameters to a file."""
        params = self.get_parameters()
        torch.save(params, self.weights_dir)

    def load_weights(self):
        """Load model parameters from the file specified by weights_dir."""
        params = torch.load(self.weights_dir, map_location=torch.device('cpu'))
        self.load_parameters(params)



if __name__ == "__main__":
    obs = torch.load("states.pt").to(device = "cpu", dtype=torch.float32)
    model = GaussianHyperGru(
        hidden_dim=256,
        input_dim=obs.shape[-1],
        output_dim=4,
        policy_input_dim=39)
    model(obs)
