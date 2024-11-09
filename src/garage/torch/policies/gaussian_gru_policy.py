import torch
import numpy as np
from garage.torch.policies.stochastic_policy import Policy
from garage.torch.modules import GaussianGRUModule
import os


class GaussianGRUPolicy(Policy):
    def __init__(
        self,
        env_spec,
        hidden_dim=128,
        name='GaussianGRUPolicy',
        learn_std=True,
        std_share_network=False,
        init_std=1.0,
        min_std=None,
        max_std=None,
        layer_normalization=False,
        output_nonlinearity=None,
        state_include_action=True,
        load_weights=False,
        weights_dir=None,
    ):
        super().__init__(env_spec=env_spec, name=name)
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._hidden_dim = hidden_dim
        self._output_nonlinearity = output_nonlinearity
        self._learn_std = learn_std
        self._std_share_network = std_share_network
        self._init_std = init_std
        self._min_std = min_std
        self._max_std = max_std
        self._layer_normalization = layer_normalization
        self._state_include_action = state_include_action


        if state_include_action:
            self._input_dim = self._obs_dim + self._action_dim
        else:
            self._input_dim = self._obs_dim


        self._prev_actions = None
        self._prev_hiddens = None
        self._init_hidden = torch.zeros(1, hidden_dim, dtype=torch.float64)
        self._module = GaussianGRUModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_dim=self._hidden_dim,
            learn_std=self._learn_std,
            init_std=self._init_std,
            min_std=self._min_std,
            max_std=self._max_std,
            std_share_network=self._std_share_network,
            layer_normalization=self._layer_normalization,
            output_nonlinearity=self._output_nonlinearity
        )
        #
        if weights_dir is None:
            self.weights_dir = f"saved_models/rl_2_gru.pth"
        else:
            self.weights_dir = weights_dir
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

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if isinstance(self._prev_hiddens, torch.Tensor):
            self._prev_hiddens = self._prev_hiddens.to(*args, **kwargs)
        if isinstance(self._init_hidden, torch.Tensor):
            self._init_hidden = self._init_hidden.to(*args, **kwargs)

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

    def eval(self):
        self._module.eval()

    def train(self, mode=True):
        self._module.train(mode=mode)

    @property
    def state_info_specs(self):
        if self._state_include_action:
            return [('prev_action', (self._action_dim,))]
        return []

    def clone(self, name):
        new_policy = self.__class__(
            env_spec=self._env_spec,
            hidden_dim=self._hidden_dim,
            name=name,
            learn_std=self._learn_std,
            std_share_network=self._std_share_network,
            init_std=self._init_std,
            min_std=self._min_std,
            max_std=self._max_std,
            layer_normalization=self._layer_normalization,
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
        os.makedirs(os.path.dirname(self.weights_dir), exist_ok=True)
        torch.save(params, self.weights_dir)

    def load_weights(self):
        """Load model parameters from the file specified by weights_dir."""
        params = torch.load(self.weights_dir, map_location=torch.device('cpu'))
        self.load_parameters(params)
