"""Natural Policy Gradient Optimization."""
# pylint: disable=wrong-import-order
# yapf: disable
import collections
import copy

from dowel import logger, tabular
import numpy as np
import tensorflow as tf
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from garage import log_performance, log_multitask_performance, make_optimizer
from garage.np import explained_variance_1d, pad_batch_array
from garage.np.algos import RLAlgorithm
from garage.torch._functions import (
    zero_optim_grads, compute_advantages, flatten_inputs, torch_to_np)
from garage.np. _functions import discount_cumsum
from garage.tf import center_advs, positive_advs
from garage.torch.optimizers import FirstOrderOptimizer

# yapf: enable


class NPO(RLAlgorithm):
    """Natural Policy Gradient Optimization for RL^2.

    This class performs natural policy gradient optimization, tailored for the RL^2
    algorithm, as described in https://arxiv.org/pdf/1611.02779.pdf.

    Args:
        env_spec (EnvSpec): Environment specification detailing observation and action spaces.
        policy (garage.torch.policies.StochasticPolicy): The stochastic policy network for the agent.
        baseline (garage.torch.baselines.Baseline): Baseline network used to fit the value function.
        baseline_const (garage.torch.baselines.Baseline or garage.np.baselines.Baseline or dict, optional):
            Baseline for constraint-specific evaluation, if applicable. Can be a PyTorch or NumPy baseline,
            or a dictionary for handling multiple environment-specific constraints.
        sampler (garage.sampler.Sampler): Sampler instance for generating episodes.
        scope (str): Identifier scope for the algorithm. Required if running multiple algorithms
            simultaneously, each with distinct environments and policies.
        discount (float): Discount factor applied to future rewards, typically between 0 and 1.
        gae_lambda (float): Lambda parameter for generalized advantage estimation (GAE).
        center_adv (bool): Whether to normalize advantages to have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift advantages to be positive. When used with center_adv,
            the advantages will be standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon by disregarding terminal states within an episode.
        pg_loss (str): Loss function type for policy gradient, one of: 'vanilla', 'surrogate',
            or 'surrogate_clip'.
        lr_clip_range (float): Maximum likelihood ratio between policies, constraining the update range (used in PPO).
        max_kl_step (float): Maximum KL divergence between old and new policies, acting as a trust region constraint (used in TRPO).
        optimizer (torch.optim.Optimizer): Optimizer instance used for training, typically from `torch.optim`.
        optimizer_args (dict): Additional arguments for configuring the optimizer.
        policy_ent_coeff (float): Coefficient for policy entropy. A zero value removes entropy regularization.
        use_softplus_entropy (bool): Whether to apply softmax to the entropy estimate, preventing negative entropy values.
        use_neg_logli_entropy (bool): Whether to calculate entropy as the negative log likelihood of actions.
        stop_entropy_gradient (bool): Whether to stop gradients from propagating through entropy terms.
        entropy_method (str): Entropy method, one of: 'max', 'regularized', or 'no_entropy'. 'max' adds dense entropy
            to the reward for each step, while 'regularized' adds mean entropy to the surrogate objective.
            For details, see https://arxiv.org/abs/1805.00909.
        fit_baseline (str): Method for fitting the baseline, either 'before' or 'after'. Currently, only 'before' is supported.
        name (str): Name of the algorithm, typically used for logging and identification.

    Note:
        sane defaults for entropy configuration:
            - entropy_method='max', center_adv=False, stop_gradient=True
              (center_adv normalizes the advantages tensor, which will
              significantly alleviate the effect of entropy. It is also
              recommended to turn off entropy gradient so that the agent
              will focus on high-entropy actions instead of increasing the
              variance of the distribution.)
            - entropy_method='regularized', stop_gradient=False,
              use_neg_logli_entropy=False

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 sampler,
                 lagrangian_start=None,
                 constraint_threshold=0,
                 baseline_const=None,
                 scope=None,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 pg_loss='surrogate',
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args_policy=None,
                 optimizer_args_baseline=None,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 name='NPO',
                 multitask=False,
                 num_tasks=None,
                 task_update_frequency=1,
                 train_task_sampler=None,
                 train_constraint=False,
                 ):
        self._task_update_frequency = task_update_frequency
        self._multitask = multitask
        self._train_task_sampler = train_task_sampler
        self._num_tasks = num_tasks

        self.policy = policy
        self._scope = scope
        self.max_episode_length = env_spec.max_episode_length
        self._env_spec = env_spec
        self._baseline = baseline
        self._lagrangian_start=lagrangian_start
        self._constraint_threshold=constraint_threshold
        self._baseline_const = baseline_const
        self._discount = discount
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._fixed_horizon = fixed_horizon
        self._name = name
        self._name_scope = tf.name_scope(self._name)
        self._old_policy = policy.clone('old_policy')
        self._use_softplus_entropy = use_softplus_entropy
        self._use_neg_logli_entropy = use_neg_logli_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._pg_loss = pg_loss
        if optimizer is None:
            if optimizer_args_policy is None:
                optimizer_args_policy = dict()
            if optimizer_args_baseline is None:
                optimizer_args_baseline = dict()
            optimizer = FirstOrderOptimizer
        optimizer_args_policy["model"] = self.policy
        optimizer_args_policy["name"] = "policy"
        if self._baseline is not None:
            optimizer_args_baseline["model"] = self._baseline
            optimizer_args_baseline["name"] = "value"
            optimizer_args_baseline["load_state"] = self._baseline.load_weights_from_disc
        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          use_neg_logli_entropy,
                                          policy_ent_coeff)

        if pg_loss not in ['vanilla', 'surrogate', 'surrogate_clip']:
            raise ValueError('Invalid pg_loss')
        # move model to gpu if possible before creating optimizer
        self.policy.to(device)
        self._optimizer = make_optimizer(optimizer, **optimizer_args_policy)
        self._optimizer.update_opt(self._policy_loss)
        if self._baseline is not None:
            self._bl_optimizer = make_optimizer(optimizer, **optimizer_args_baseline)
            self._bl_optimizer.update_opt(self._baseline.compute_loss)

        self._lr_clip_range = float(lr_clip_range)
        self._max_kl_step = float(max_kl_step)
        self._policy_ent_coeff = float(policy_ent_coeff)

        self._f_rewards = None
        self._f_returns = None
        self._f_policy_kl = None
        self._f_policy_entropy = None
        self._f_policy_stddev = None
        self._policy_network = None
        self._old_policy_network = None

        self._episode_reward_mean = collections.deque(maxlen=100)

        self._sampler = sampler
        # send policy to cpu for collecting episodes
        self._train_constraint = train_constraint
        if train_constraint:
            if baseline_const and isinstance(baseline_const, torch.nn.Module):
                optimizer_args_bl_const = copy.deepcopy(optimizer_args_baseline)
                optimizer_args_bl_const["model"] = self._baseline_const
                optimizer_args_bl_const["name"] = "value_constraint"
                optimizer_args_bl_const[
                    "load_state"] = self._baseline_const.load_weights_from_disc
                self._bl_optimizer_const = make_optimizer(
                    optimizer, **optimizer_args_bl_const
                )
                self._bl_optimizer_const.update_opt(self._baseline_const.compute_loss)

        self.policy.to("cpu")

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer, which rovides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for i, _ in enumerate(trainer.step_epochs()):
            if not self._multitask:
                trainer.step_path = trainer.obtain_episodes(trainer.step_itr)
            else:
                env_updates = None
                assert self._train_task_sampler is not None
                if (not i % self._task_update_frequency) or (self._task_update_frequency == 1):
                    env_updates = self._train_task_sampler.sample(self._num_tasks)
                trainer.step_path = self.obtain_exact_trajectories(trainer, env_update=env_updates)
            last_return = self._train_once(trainer.step_itr, trainer.step_path)
            trainer.step_itr += 1

        return last_return

    def _train_once(self, itr, episodes):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            episodes (EpisodeBatch): Batch of episodes.

        Returns:
            numpy.float64: Average return.

        """
        # -- Stage: Calculate and pad baselines
        obs = [
            self._baseline.predict({'observations': obs})
            for obs in episodes.observations_list
        ]
        baselines = pad_batch_array(np.concatenate(obs), episodes.lengths,
                                    self.max_episode_length)

        # -- Stage: Run and calculate performance of the algorithm
        if not self._multitask:
            undiscounted_returns = log_performance(itr,
                                                episodes,
                                                discount=self._discount)
        else:
            undiscounted_returns = log_multitask_performance(itr,
                                                             episodes,
                                                             discount=self._discount)
        self._episode_reward_mean.extend(undiscounted_returns)
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self._episode_reward_mean))

        logger.log('Optimizing policy...')
        self._optimize_policy(episodes, baselines)

        return np.mean(undiscounted_returns)

    def _optimize_policy(self, episodes, baselines):
        """Optimize policy.

        Args:
            episodes (EpisodeBatch): Batch of episodes.
            baselines (np.ndarray): Baseline predictions.

        """
        policy_opt_input_values = self._policy_opt_input_values(
            episodes, baselines)
        logger.log('Computing loss before')
        loss_before = self._optimizer.loss(policy_opt_input_values)
        logger.log('Computing KL before')
        policy_kl_before = self._f_policy_kl(*policy_opt_input_values)
        logger.log('Optimizing')
        self._optimizer.optimize(policy_opt_input_values)
        logger.log('Computing KL after')
        policy_kl = self._f_policy_kl(*policy_opt_input_values)
        logger.log('Computing loss after')
        loss_after = self._optimizer.loss(policy_opt_input_values)
        tabular.record('{}/LossBefore'.format(self.policy.name), loss_before)
        tabular.record('{}/LossAfter'.format(self.policy.name), loss_after)
        tabular.record('{}/dLoss'.format(self.policy.name),
                       loss_before - loss_after)
        tabular.record('{}/KLBefore'.format(self.policy.name),
                       policy_kl_before)
        tabular.record('{}/KL'.format(self.policy.name), policy_kl)
        pol_ent = self._f_policy_entropy(*policy_opt_input_values)
        ent = np.sum(pol_ent) / np.sum(episodes.lengths)
        tabular.record('{}/Entropy'.format(self.policy.name), ent)
        pol_std = self._f_policy_stddev(*policy_opt_input_values)
        std = np.sum(pol_std) / np.sum(episodes.lengths)
        tabular.record('{}/StandardDeviation'.format(self.policy.name), std)
        tabular.record('{}/Perplexity'.format(self.policy.name), np.exp(ent))
        returns = self._fit_baseline_with_data(episodes, baselines)

        ev = explained_variance_1d(baselines, returns, episodes.valids)

        tabular.record('{}/ExplainedVariance'.format(self._baseline.name), ev)
        self._old_policy.parameters = self.policy.parameters

    def _build_inputs(self):
        """Build input variables.

        Returns:
            namedtuple: Collection of variables to compute policy loss.
            namedtuple: Collection of variables to do policy optimization.

        """
        observation_space = self._env_spec.observation_space
        action_space = self._env_spec.action_space

        with tf.name_scope('inputs'):
            obs_var = observation_space.to_tf_placeholder(name='obs',
                                                          batch_dims=2)
            action_var = action_space.to_tf_placeholder(name='action',
                                                        batch_dims=2)
            reward_var = tf.compat.v1.placeholder(tf.float32,
                                                  shape=[None, None],
                                                  name='reward')
            valid_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=[None, None],
                                                 name='valid')
            baseline_var = tf.compat.v1.placeholder(tf.float32,
                                                    shape=[None, None],
                                                    name='baseline')

            policy_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name=k)
                for k, shape in self.policy.state_info_specs
            }
            #policy_state_info_vars_list = [
            #    policy_state_info_vars[k] for k in self.policy.state_info_keys
            #]

        augmented_obs_var = obs_var
        #for k in self.policy.state_info_keys:
        #    extra_state_var = policy_state_info_vars[k]
        #    extra_state_var = tf.cast(extra_state_var, tf.float32)
        #    augmented_obs_var = tf.concat([augmented_obs_var, extra_state_var],
        #                                  -1)

        #self._policy_network = self.policy.build(augmented_obs_var,
        #                                         name='policy')
        #self._old_policy_network = self._old_policy.build(augmented_obs_var,
        #                                                  name='policy')

        #policy_loss_inputs = graph_inputs(
        #    'PolicyLossInputs',
        #    action_var=action_var,
        #    reward_var=reward_var,
        #    baseline_var=baseline_var,
        #    valid_var=valid_var,
        #    policy_state_info_vars=policy_state_info_vars,
        #)
        #policy_opt_inputs = graph_inputs(
        #    'PolicyOptInputs',
        #    obs_var=obs_var,
        #    action_var=action_var,
        #    reward_var=reward_var,
        #    baseline_var=baseline_var,
        #    valid_var=valid_var,
        #    policy_state_info_vars_list=policy_state_info_vars_list,
        #)

        #return policy_loss_inputs, policy_opt_inputs

    # pylint: disable=too-many-branches, too-many-statements
    def _policy_loss(
        self,
        states,
        actions,
        rewards,
        baselines,
        valids,
        penalties=None,
        baselines_const=None,
        lambdas=None
    ):
        """Build policy loss and other output tensors.

        Args:
            i (namedtuple): Collection of variables to compute policy loss.
            lambdas: optional if mutliple lambdas

        Returns:
            torch.Tensor: Policy loss.
            torch.Tensor: Mean policy KL divergence.

        """
        batch_size = states.shape[0]
        self.policy.reset(
            do_resets= np.full(shape=batch_size, fill_value=True),
            dtype=torch.float64
        )
        if getattr(self.policy, "is_actor_critic", False):
            # calculate temporal difference loss
            dist, _, value_pred = self.policy.forward(states, get_value=True)
            # bootstrap last pred to be same as previous pred
            next_pred = torch.cat(
                [value_pred[:, 1:], value_pred[:, -1:]], dim=1)
            td_errors = rewards + self._discount * next_pred - value_pred
            critic_loss = (0.5*td_errors**2).mean()
        else:
            dist, _ = self.policy.forward(states)
        policy_entropy = self._entropy(dist, actions)

        if self._maximum_entropy:
                rewards += self._policy_ent_coeff *  policy_entropy


        adv = compute_advantages(self._discount,
                                 self._gae_lambda,
                                 self.max_episode_length,
                                 baselines,
                                 rewards,
                                 ).to(dtype=torch.float64)

        adv = torch.reshape(adv, (-1, self.max_episode_length))

        if self._train_constraint:
            adv_const = compute_advantages(self._discount,
                                 self._gae_lambda,
                                 self.max_episode_length,
                                 baselines_const,
                                 penalties,
                                 ).to(dtype=torch.float64)
            if hasattr(self.policy, "lagrangians"):
                lagrangian = lambdas
            else:
                lagrangian = torch.detach(self.policy.lagrangian).item()
            adv -=  lagrangian * adv_const
            adv /= (1 + lagrangian)
        # Optionally normalize advantages
        eps = 1e-8

        if self._center_adv:
            adv = center_advs(adv, axes=[0], eps=eps)

        if self._positive_adv:
            adv = positive_advs(adv, eps)
        self._old_policy.reset(
            do_resets=np.full(shape=batch_size, fill_value=True),
            dtype=torch.float64
        )
        with torch.no_grad():
            old_policy_dist, _ = self._old_policy.forward(states)

        kl = torch.distributions.kl_divergence(old_policy_dist, dist)
        pol_mean_kl = kl.mean()

        # Calculate vanilla loss

        ll = dist.log_prob(actions).sum(axis=-1)
        vanilla = ll * adv

        # Calculate surrogate loss

        lr = torch.exp(ll - old_policy_dist.log_prob(actions).sum(axis=-1))
        surrogate = lr * adv

        # Finalize objective function

        if self._pg_loss == 'vanilla':
            # VPG uses the vanilla objective
            obj = vanilla  # In PyTorch, just use the variable directly
        elif self._pg_loss == 'surrogate':
            # TRPO uses the standard surrogate objective
            obj = surrogate  # In PyTorch, just use the variable directly
        elif self._pg_loss == 'surrogate_clip':
            # Clip the learning rate
            lr_clip = torch.clamp(lr, min=1 - self._lr_clip_range,
                                  max=1 + self._lr_clip_range)
            surr_clip = lr_clip * adv
            obj = torch.min(surrogate, surr_clip)  # Element-wise minimum

        if self._entropy_regularzied:
            obj += self._policy_ent_coeff * policy_entropy

        # filter only the valid values
        valid_var_bool = valids.bool()
        # Apply the Boolean mask
        obj = obj[valid_var_bool]
        # Maximize E[surrogate objective] by minimizing
        # -E_t[surrogate objective]
        loss = -obj.mean()
        if getattr(self.policy, "is_actor_critic", False):
            loss += critic_loss
        return loss, pol_mean_kl

    def _entropy(self, dist, actions):
        """Build policy entropy tensor.

        Args:
            i (namedtuple): Collection of variables to compute policy loss.

        Returns:
            tf.Tensor: Policy entropy.

        """
        if self._use_neg_logli_entropy:
            policy_entropy = -dist.log_prob(actions)
        else:
            policy_entropy = dist.entropy()

        # This prevents entropy from becoming negative for small policy std
        if self._use_softplus_entropy:
            policy_entropy = torch.nn.functional.softplus(policy_entropy)

        if self._stop_entropy_gradient:
            policy_entropy = policy_entropy.detach()
        return policy_entropy.sum(axis=-1)

    def _fit_baseline_with_data(self, episodes, baselines):
        """Update baselines from samples.

        Args:
            episodes (EpisodeBatch): Batch of episodes.
            baselines (np.ndarray): Baseline predictions.

        Returns:
            np.ndarray: Augment returns.

        """
        policy_opt_input_values = self._policy_opt_input_values(
            episodes, baselines)

        #returns_tensor = discount_cumsum(
        #    policy_opt_input_values[2], self._discount)
        returns_tensor = np.stack([
            discount_cumsum(reward, self._discount)
            for reward in policy_opt_input_values[2]])

        paths = []
        valids = episodes.valids
        observations = episodes.padded_observations

        # Compute returns
        for ret, val, ob in zip(returns_tensor, valids, observations):
            returns = ret[val.astype(bool)]
            obs = ob[val.astype(bool)]
            paths.append(dict(observations=obs, returns=returns))

        # Fit baseline
        zero_optim_grads(self._bl_optimizer.optimizer)
        logger.log('Fitting baseline...')
        xs = np.concatenate([p['observations'] for p in paths])
        if not isinstance(xs, np.ndarray) or len(xs.shape) > 2:
            xs = self._env_spec.observation_space.flatten_n(xs)
        ys = np.concatenate([p['returns'] for p in paths])
        ys = ys.reshape((-1, 1))
        if self._baseline.normalize_inputs:
            self._baseline.x_mean = torch.tensor(
                np.mean(xs, axis=0, keepdims=True),
                dtype=torch.float64,
                device=device
            )
            self._baseline.x_std = torch.tensor(
                np.std(xs, axis=0, keepdims=True) + 1e-8,
                dtype=torch.float64,
                device=device
            )
        if self._baseline.normalize_outputs:
            # recompute normalizing constants for outputs
            self._baseline.y_mean = torch.tensor(
                np.mean(ys, axis=0, keepdims=True),
                dtype=torch.float64,
                device=device
            )
            self._baseline.y_std = torch.tensor(
                np.std(ys, axis=0, keepdims=True) + 1e-8,
                dtype=torch.float64,
                device=device
            )
        x_tensor =  torch.tensor(
                xs, dtype=torch.float64, device=device
            )
        y_tensor = torch.tensor(ys, dtype=torch.float64, device=device)
        with torch.no_grad():
            loss_before = self._baseline.compute_loss(
            x_tensor, y_tensor
        )
        tabular.record('{}/LossBefore'.format("Baseline"), loss_before)
        self._bl_optimizer.optimize(x_tensor, y_tensor)
        with torch.no_grad():
            loss_after = self._baseline.compute_loss(
                x_tensor, y_tensor
            )
        tabular.record('{}/LossAfter'.format("Baseline"), loss_after)
        return returns_tensor

    def _fit_baseline_const(self, episodes):
        """Update baselines from samples.

        Args:
            episodes (EpisodeBatch): Batch of episodes.

        Returns:
            np.ndarray: Augment returns.

        """
        penalties = pad_batch_array(
            episodes.env_infos["constraint"].astype(float),
            episodes.lengths,
            self.max_episode_length
        )

        penalties_tensor = np.stack([
            discount_cumsum(violations, self._discount)
            for violations in penalties])

        paths = []
        valids = episodes.valids
        observations = episodes.padded_observations

        # Compute returns
        for pen, val, ob in zip(penalties_tensor, valids, observations):
            returns = pen[val.astype(bool)]
            obs = ob[val.astype(bool)]
            paths.append(dict(observations=obs, returns=returns))

        # Fit baseline

        logger.log('Fitting constraint baseline...')
        xs = np.concatenate([p['observations'] for p in paths])
        if not isinstance(xs, np.ndarray) or len(xs.shape) > 2:
            xs = self._env_spec.observation_space.flatten_n(xs)
        ys = np.concatenate([p['returns'] for p in paths])
        ys = ys.reshape((-1, 1))

        # If baseline is a torch Module
        if isinstance(self._baseline_const, torch.nn.Module):
            zero_optim_grads(self._bl_optimizer_const.optimizer)
            if self._baseline_const.normalize_inputs:
                self._baseline_const.x_mean = torch.tensor(
                    np.mean(xs, axis=0, keepdims=True),
                    dtype=torch.float64,
                    device=device,
                )
                self._baseline_const.x_std = torch.tensor(
                    np.std(xs, axis=0, keepdims=True) + 1e-8,
                    dtype=torch.float64,
                    device=device,
                )
            if self._baseline_const.normalize_outputs:
                # recompute normalizing constants for outputs
                self._baseline_const.y_mean = torch.tensor(
                    np.mean(ys, axis=0, keepdims=True),
                    dtype=torch.float64,
                    device=device,
                )
                self._baseline_const.y_std = torch.tensor(
                    np.std(ys, axis=0, keepdims=True) + 1e-8,
                    dtype=torch.float64,
                    device=device,
                )
            x_tensor =  torch.tensor(
                xs,
                dtype=torch.float64,
                device=device
                )
            y_tensor = torch.tensor(ys, dtype=torch.float64, device=device)
            with torch.no_grad():
                loss_before = self._baseline_const.compute_loss(
                x_tensor, y_tensor
            )
            tabular.record('{}/LossBefore'.format("BaselineConstraint"), loss_before)
            self._bl_optimizer_const.optimize(x_tensor, y_tensor)
            with torch.no_grad():
                loss_after = self._baseline_const.compute_loss(
                    x_tensor, y_tensor
                )
            tabular.record('{}/LossAfter'.format("BaselineConstraint"), loss_after)
        else:
            if isinstance(self._baseline_const, dict):
                # seperate baselines for envs
                for env_name in self._baseline_const.keys():
                    env_path = [paths[i] for i in range(len(paths)) if
                                episodes.padded_env_infos["task_name"][
                                    i, 0] == env_name]
                    self._baseline_const[env_name].fit(env_path)
            else:
                self._baseline_const.fit(paths)
        return penalties_tensor, penalties

    def _policy_opt_input_values(self, episodes, baselines):
        """Map episode samples to the policy optimizer inputs.

        Args:
            episodes (EpisodeBatch): Batch of episodes.
            baselines (np.ndarray): Baseline predictions.

        Returns:
            list(np.ndarray): Flatten policy optimization input values.

        """
        actions = [
            self._env_spec.action_space.flatten_n(act)
            for act in episodes.actions_list
        ]
        padded_actions = pad_batch_array(np.concatenate(actions),
                                         episodes.lengths,
                                         self.max_episode_length)

        # pylint: disable=unexpected-keyword-arg
        policy_opt_input_values = [
            episodes.padded_observations,
            padded_actions,
            episodes.padded_rewards,
            baselines,
            episodes.valids,
            [],
        ]

        return flatten_inputs(policy_opt_input_values)

    def _check_entropy_configuration(self, entropy_method, center_adv,
                                     stop_entropy_gradient,
                                     use_neg_logli_entropy, policy_ent_coeff):
        """Check entropy configuration.

        Args:
            entropy_method (str): A string from: 'max', 'regularized',
                'no_entropy'. The type of entropy method to use. 'max' adds the
                dense entropy to the reward for each time step. 'regularized'
                adds the mean entropy to the surrogate objective. See
                https://arxiv.org/abs/1805.00909 for more details.
            center_adv (bool): Whether to rescale the advantages
                so that they have mean 0 and standard deviation 1.
            stop_entropy_gradient (bool): Whether to stop the entropy gradient.
            use_neg_logli_entropy (bool): Whether to estimate the entropy as
                the negative log likelihood of the action.
            policy_ent_coeff (float): The coefficient of the policy entropy.
                Setting it to zero would mean no entropy regularization.

        Raises:
            ValueError: If center_adv is True when entropy_method is max.
            ValueError: If stop_gradient is False when entropy_method is max.
            ValueError: If policy_ent_coeff is non-zero when there is
                no entropy method.
            ValueError: If entropy_method is not one of 'max', 'regularized',
                'no_entropy'.

        """
        del use_neg_logli_entropy

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
            self._maximum_entropy = True
            self._entropy_regularzied = False
        elif entropy_method == 'regularized':
            self._maximum_entropy = False
            self._entropy_regularzied = True
        elif entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')
            self._maximum_entropy = False
            self._entropy_regularzied = False
        else:
            raise ValueError('Invalid entropy_method')



    def obtain_exact_trajectories(self, trainer, env_update):
        """Obtain an exact amount of trajs from each env being sampled from.

        Args:
            trainer (Trainer): Experiment trainer, which rovides services
                such as snapshotting and sampler control.

        Returns:
            episodes (EpisodeBatch): Batch of episodes.
        """
        episodes_per_trajectory = trainer._train_args.batch_size
        agent_update = self.policy.get_param_values()
        sampler = trainer._sampler
        episodes = sampler.obtain_exact_episodes(
                              episodes_per_trajectory,
                              agent_update,
                              env_update=env_update)
        trainer._stats.total_env_steps += sum(episodes.lengths)
        return episodes
