import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from garage.torch.policies.stochastic_policy import Policy
from copy import deepcopy
from typing import OrderedDict
class RL2LSTMCritic(nn.Module):
    def __init__(self, input_dim, lstm_hidden_size=128, hidden_layers=(128, 128), lr=0.001):
        super(RL2LSTMCritic, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size

        # LSTM layer as the first layer to process sequences
        self.lstm = nn.LSTM(input_dim, lstm_hidden_size, batch_first=True)

        # Fully connected layers after LSTM
        self.fc_layers = nn.ModuleList()
        prev_layer_size = lstm_hidden_size  # Input size to the first FC layer is the LSTM's output size
        for layer_size in hidden_layers:
            self.fc_layers.append(nn.Linear(prev_layer_size, layer_size))
            prev_layer_size = layer_size

        # Output layer for value estimation
        self.value = nn.Linear(prev_layer_size, 1)

    def forward(self, x, hidden_state=None):
        # Process input sequence through LSTM
        x, hidden_state = self.lstm(x, hidden_state)

        # Instead of using just the last timestep output, process all timesteps
        # through the fully connected layers.
        # Apply fully connected layers to each timestep
        # This requires reshaping x to [-1, self.lstm_hidden_size] to process all timesteps together
        # and then reshaping back to [batch_size, sequence_length, -1] for the final output
        batch_size, sequence_length, _ = x.shape
        x = x.reshape(-1, self.lstm_hidden_size)
        for layer in self.fc_layers:
            x = F.leaky_relu(layer(x))
        x = x.reshape(batch_size, sequence_length, -1)

        # Compute value for each timestep
        value = self.value(x)
        # No need to squeeze the last dimension here as we want the value per timestep

        return value, hidden_state

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_hidden_size),
                torch.zeros(1, batch_size, self.lstm_hidden_size))


class GaussianLSTMModule(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim = 4,
        lstm_hidden_size=128,
        mlp_layers=(128, 128),
        lr=0.001
    ):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_layers=mlp_layers
        # LSTM layer as the first layer to process sequences
        self.lstm = nn.LSTM(input_dim, lstm_hidden_size, batch_first=True)

        # Fully connected layers after LSTM
        self.fc_layers = nn.ModuleList()
        prev_layer_size = lstm_hidden_size  # Input size to the first FC layer is the LSTM's output size
        for layer_size in mlp_layers:
            self.fc_layers.append(nn.Linear(prev_layer_size, layer_size))
            prev_layer_size = layer_size

        # Output layers for means and stds
        self.means = nn.Linear(prev_layer_size, output_dim)
        self.stds = nn.Linear(prev_layer_size, output_dim)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.hidden = self.init_hidden()

    def forward(self, x):
        # Process input sequence through LSTM
        x, hidden_state = self.lstm(x, self.hidden)
        self.hidden = hidden_state
        # Instead of using just the last timestep output, process all timesteps
        # through the fully connected layers.
        # Apply fully connected layers to each timestep
        # This requires reshaping x to [-1, self.lstm_hidden_size] to process all timesteps together
        # and then reshaping back to [batch_size, sequence_length, -1] for the final output
        batch_size, sequence_length, _ = x.shape
        x = x.reshape(-1, self.lstm_hidden_size)
        for layer in self.fc_layers:
            x = F.leaky_relu(layer(x))
        x = x.reshape(batch_size, sequence_length, -1)

        # Output means and stds
        means = self.means(x)
        means = torch.tanh(means)
        stds = self.stds(x)
        stds = F.softplus(stds) + 1e-6  # Ensure stds are positive

        # Create a normal distribution and sample actions
        dist = torch.distributions.Normal(means, stds)

        return dist, {
            "means": means,
            "stds": stds,
        }

    def init_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.lstm_hidden_size),
                torch.zeros(1, batch_size, self.lstm_hidden_size))

class GaussianLSTMPolicy(Policy):
    def __init__(
        self,
        env_spec,
        lstm_hidden_size=128,
        mlp_layers=(128, 128),
        name='GaussianLSTMPolicy',
        module_params=None,
        state_include_action=True,
        load_weights=False
    ):
        super().__init__(
            env_spec=env_spec,
            name=name,
        )
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_layers = mlp_layers
        self._state_include_action = state_include_action

        if state_include_action:
            self._input_dim = self._obs_dim + self._action_dim
        else:
            self._input_dim = self._obs_dim
        self._module = GaussianLSTMModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            lstm_hidden_size=self.lstm_hidden_size,
            mlp_layers=self.mlp_layers
        )
        if module_params:
            self._module.load_state_dict(module_params)
        self._prev_actions = None
        self._prev_hiddens = None
        self._init_hidden = deepcopy(self._module.hidden)
        self.weights_dir = "saved_models/rl_2_lstm.pth"
        if load_weights:
            self.load_weights()

    def forward(self, x):
        # Process input sequence through LSTM
        return self._module.forward(x)

    def reset(self, do_resets=None):
        """Reset the policy based on terminal states.

        Args:
            do_resets (numpy.ndarray): Bool array indicating which states to reset.
        """
        if do_resets is None:
            do_resets = np.array([True])
        if self._prev_actions is None or len(do_resets) != len(self._prev_actions):
            # Initialize or reinitialize the previous actions and hidden states
            self._prev_actions = np.zeros((len(do_resets), self.action_space.flat_dim))
            # For hidden states in PyTorch, replicate the initial state across the batch as needed
            self._prev_hiddens = tuple(h.repeat(len(do_resets), 1, 1) for h in self._init_hidden)

        # Reset actions
        self._prev_actions[do_resets] = 0.
        # Reset hidden states for LSTM, assuming do_resets affects all states equally
        if np.any(do_resets):
            self._prev_hiddens = tuple(h.repeat(len(do_resets), 1, 1) for h in self._init_hidden)
        self.reset_hidden(batch_size=do_resets.shape[0])

    def reset_hidden(self, batch_size=1):
        self._module.hidden = self._module.init_hidden(batch_size=batch_size)
    def get_action(self, observation):
        action, info = self.get_actions(observation.reshape(1, 1, -1))
        return action.squeeze(), {k: v[0] for k, v in info.items()}

    def get_actions(self, observations):
        if self._state_include_action:
            assert self._prev_actions is not None
            all_input = np.concatenate([observations, self._prev_actions],
                                       axis=-1)
        else:
            all_input = observations
        dist, info =  self.forward(torch.from_numpy(all_input).float())
        info = {k: v.detach().numpy() for k, v in info.items()}
        action = dist.sample().squeeze().numpy()
        self._prev_actions = np.copy(action)
        self._prev_hiddens = tuple(h.clone() for h in self._module.hidden)
        if self._state_include_action:
            info['prev_action'] = np.copy(action)
        return action, info
    @property
    def state_info_specs(self):
        """State info specifcation.

        Returns:
            List[str]: keys and shapes for the information related to the
                policy's state when taking an action.

        """
        if self._state_include_action:
            return [
                ('prev_action', (self._action_dim, )),
            ]

        return []

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec
    def clone(self, name):
        new_policy = self.__class__(
            env_spec=self._env_spec,
            lstm_hidden_size=self.lstm_hidden_size,
            mlp_layers=self.mlp_layers,
            name=name,
            module_params=self._module.state_dict(),
            state_include_action=self._state_include_action,
        )
        return new_policy

    def load_parameters(self, new_parameters: OrderedDict):
        self._module.load_state_dict(new_parameters)

    def get_parameters(self):
        return self._module.state_dict()

    def save_weights(self):
        params = self.get_parameters()
        torch.save(params, self.weights_dir)

    def load_weights(self):
        params = torch.load(self.weights_dir)
        self.load_parameters(params)
