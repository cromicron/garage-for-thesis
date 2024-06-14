import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Any, Optional
import copy
from torch import Tensor
import os

from garage.torch.policies.stochastic_policy import Policy
from garage.torch.modules import GRUEncoder, HyperNetwork

class FeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size, activation=torch.relu):
        super(FeatureExtractor, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._activation = activation
        self._fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self._activation(self._fc(x))
class GaussianHyperGRUPolicy(Policy):
    def __init__(
        self,
        env_spec,
        hidden_dim=32,
        emb_size_gru=10,
        policy_input_dim=40,
        mainnet_dim=[32, 32],
        input_dim_hyper_latent=25,
        input_dim_policy_net=256,
        output_dim=4,
        name='GaussianHyperGRUPolicy',
        min_std=None,
        max_std=None,
        feature_nonlinearity=torch.relu,
        hyper_input_nonlinearity=torch.tanh,
        policy_nonlinearity=torch.tanh,
        output_nonlinearity=lambda x: x,
        state_include_action=False,
        load_weights=False,
        weights_dir=None,
        n_constraints=0,
    ):

        """

        Args:
            env_spec:
            hidden_dim:
            emb_size_gru:
            policy_input_dim:
            mainnet_dim:
            input_dim_hyper_latent:
            input_dim_policy_net:
            output_dim:
            name:
            min_std:
            max_std:
            feature_nonlinearity:
            hyper_input_nonlinearity:
            policy_nonlinearity:
            output_nonlinearity:
            state_include_action:
            load_weights:
            weights_dir:
            n_constraints:
        """
        super().__init__(env_spec=env_spec, name=name)
        self._env_spec = env_spec
        self._action_dim = self._env_spec.action_space.flat_dim
        self._obs_dim = env_spec.observation_space.flat_dim
        self._policy_input_dim = policy_input_dim
        self._hidden_dim = hidden_dim
        self._emb_size_gru = emb_size_gru
        self._mainnet_dim = copy.copy(mainnet_dim)
        self._mainnet_dim.append(2*output_dim)
        self._input_dim_hyper_latent = input_dim_hyper_latent
        self._input_dim_policy_net = input_dim_policy_net
        self._output_dim = output_dim
        self._min_std = min_std
        self._max_std = max_std
        if min_std is not None and max_std is not None:
            self._log_min_std = torch.log(torch.tensor(min_std))
            self._log_max_std = torch.log(torch.tensor(max_std))
        else:
            self._log_min_std, self._log_max_std = None, None
        self._feature_nonlinearity = feature_nonlinearity
        self._hyper_input_nonlinearity = hyper_input_nonlinearity
        self._policy_nonlinearity = policy_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._state_include_action = state_include_action
        self._n_constraints = n_constraints

        if state_include_action:
            self._input_dim = self._obs_dim + self._action_dim
        else:
            self._input_dim = self._obs_dim

        self._prev_actions = None
        self._prev_hiddens = None
        self._init_hidden = torch.zeros(1, hidden_dim, dtype=torch.float64)

        self._action_extractor = FeatureExtractor(
            self._action_dim, 16, feature_nonlinearity
        )
        self._reward_extractor = FeatureExtractor(1, 16, feature_nonlinearity)
        self._state_extractor = FeatureExtractor(
            self._policy_input_dim,
            32,
            feature_nonlinearity
        )
        if self._n_constraints:
            self._constraint_extractor = FeatureExtractor(
                self._n_constraints, 16, feature_nonlinearity
            )
        input_size_encoder = 64 + self._n_constraints*16
        self._encoder = GRUEncoder(
            input_size=input_size_encoder,
            hidden_size=self._hidden_dim,
            emb_size=self._emb_size_gru,
        )
        self._fc_latent = nn.Linear(
            self._emb_size_gru, self._input_dim_hyper_latent)
        self._fc_policy_input = nn.Linear(
            self._policy_input_dim, self._input_dim_policy_net)
        self._hyper_network = HyperNetwork(
            self._input_dim_hyper_latent,
            self._input_dim_policy_net,
            self._mainnet_dim,
            mainnet_nonlinearity=self._policy_nonlinearity
        )
        if weights_dir is None:
            self.weights_dir = "saved_models/rl_2_gru.pth"
        else:
            self.weights_dir = weights_dir
        if load_weights:
            self.load_weights()

    def create_features(self, states, actions, rewards, constraints=None):
        s = self._state_extractor(states)
        a = self._action_extractor(actions)
        r = self._reward_extractor(rewards)
        features = [s, a, r]
        if constraints is not None:
            c = self._constraint_extractor(constraints)
            features.append(c)
        return torch.cat(features, dim=-1)

    def extract_elements_from_obs(
        self,
        obs,
    ) -> Tuple[
        Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        takes in an observation and returns tuple of torch tensors
        """
        obs_size = 39 + self._n_constraints*6
        state = obs[..., : obs_size]
        action =  obs[..., obs_size: obs_size + self._env_spec.action_space.flat_dim]
        reward_index = -2 - self._n_constraints
        done_index = -1 - self._n_constraints
        reward = obs[..., reward_index].unsqueeze(-1)
        done = obs[..., done_index].unsqueeze(-1)
        state = torch.cat([state, done], axis=-1)
        if self._n_constraints:
            constraints = obs[..., - self._n_constraints]
            if self._n_constraints == 1:
                constraints = constraints.unsqueeze(-1)
        else:
            constraints = None
        return state, action, reward, constraints

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

    def forward(self, state, hidden=None):
        s, a, r, c = self.extract_elements_from_obs(state)
        input_hyper = self.create_features(s, a, r, c)
        latent, hidden = self._encoder(input_hyper, hidden)
        embedding = self._hyper_input_nonlinearity(
            self._fc_latent(latent))
        policy_input = self._hyper_input_nonlinearity(
            self._fc_policy_input(s))
        mainnet = self._hyper_network(embedding)
        x = mainnet.forward(policy_input)
        mean_std = self._output_nonlinearity(x)
        mean = mean_std[..., :self._output_dim]
        log_std = mean_std[..., self._output_dim:]
        if self._log_min_std  and self._log_max_std:
            def _softclip(x, x_min, x_max, alpha=2):
                y_scale = (x_max - x_min) / 2
                y_offset = (x_max + x_min) / 2
                x_scale = (2 * alpha) / (x_max - x_min)
                x_offset = (x_max + x_min) / 2
                return (torch.tanh((x - x_offset) * x_scale) * y_scale) + y_offset
            log_std = _softclip(log_std, self._log_min_std, self._log_max_std)
        std = torch.exp(log_std)
        distribution = torch.distributions.Normal(mean, std)

        return distribution, hidden


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
        device = self._prev_hiddens.device
        dtype = self._prev_hiddens.dtype
        dist, hidden_vec = self.forward(
            torch.tensor(all_input, dtype=dtype, device=device),
            self._prev_hiddens)
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
            emb_size_gru=self._emb_size_gru,
            policy_input_dim=self._policy_input_dim,
            mainnet_dim=self._mainnet_dim[:-1],
            input_dim_hyper_latent=self._input_dim_hyper_latent,
            input_dim_policy_net=self._input_dim_policy_net,
            output_dim=self._output_dim,
            name=self._name,
            min_std=self._min_std,
            max_std=self._max_std,
            feature_nonlinearity=self._feature_nonlinearity,
            hyper_input_nonlinearity=self._hyper_input_nonlinearity,
            policy_nonlinearity=self._policy_nonlinearity,
            output_nonlinearity=self._output_nonlinearity,
            state_include_action=self._state_include_action,
            n_constraints=self._n_constraints,
        )
        new_policy.load_parameters(self.get_parameters())
        return new_policy


    def load_parameters(self, new_parameters):
        """Load model parameters from a dictionary object."""
        self.load_state_dict(new_parameters)

    def get_parameters(self):
        """Retrieve current model parameters."""
        return self.state_dict()

    def save_weights(self):
        """Save the current model parameters to a file."""
        params = self.get_parameters()
        os.makedirs(os.path.dirname(self.weights_dir), exist_ok=True)
        torch.save(params, self.weights_dir)


    def load_weights(self):
        """Load model parameters from the file specified by weights_dir."""
        params = torch.load(self.weights_dir, map_location=torch.device('cpu'))
        self.load_parameters(params)

    def traverse_episodes(self, env, n_episodes=1):
        for _ in range(n_episodes):
            state,info = env.reset()
            done = False
            j = 0
            while not done:
                action, i = self.get_action(state)
                env_step = env.step(action)
                state = env_step.observation
                done = env_step.terminal or env_step.timeout


if __name__ == "__main__":
    import metaworld_constrained as metaworld
    from garage.envs import MetaWorldSetTaskEnv, normalize
    from garage.experiment import MetaWorldTaskSampler, SetTaskSampler
    from garage.torch.algos.rl2 import RL2Env

    constraints = True
    load_state = True
    ml10 = metaworld.ML10()
    env_name = "drawer-close-v2"
    if env_name in ml10.train_classes:
        class_type = "train"
    elif env_name in ml10.test_classes:
        class_type = "test"
    else:
        raise Exception(f"task {env_name} doesn't exist")
    tasks = MetaWorldTaskSampler(
        ml10, class_type,
        lambda env, _: RL2Env(normalize(env, normalize_reward=True), n_constraints=1))


    env_updates = tasks.sample(10)
    for task in env_updates:
        if task._task.env_name == env_name:
            env = task(render_mode="human")
    policy = GaussianHyperGRUPolicy(
        env_spec=env.spec,
        hidden_dim=256,
        policy_input_dim=46,
        input_dim_policy_net=256,
        output_dim=4,
        min_std=0.1,
        max_std=1.5,
        n_constraints=1,
    )
    policy.traverse_episodes(env, n_episodes=5)

