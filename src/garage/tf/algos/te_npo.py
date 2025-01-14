"""Natural Policy Optimization with Task Embeddings."""
# pylint: disable=too-many-lines
# yapf: disable
import akro
from dowel import Histogram, logger, tabular
import numpy as np
import scipy.stats
import tensorflow as tf

from garage import InOutSpec, log_performance, log_multitask_performance
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment import deterministic
from garage.np import (discount_cumsum, explained_variance_1d, pad_batch_array,
                       rrse, sliding_window)
from garage.np.algos import RLAlgorithm
from garage.tf import (center_advs, compile_function, compute_advantages,
                       concat_tensor_list, discounted_returns, flatten_inputs,
                       graph_inputs, pad_tensor_dict, positive_advs,
                       stack_tensor_dict_list)
from garage.tf.embeddings import StochasticEncoder
from garage.tf.optimizers import LBFGSOptimizer
from garage.tf.policies import TaskEmbeddingPolicy

# yapf: enable


class TENPO(RLAlgorithm):
    """Natural Policy Optimization with Task Embeddings.

    See https://karolhausman.github.io/pdf/hausman17nips-ws2.pdf for algorithm
    reference.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.TaskEmbeddingPolicy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        sampler (garage.sampler.Sampler): Sampler.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies, as in PPO.
        max_kl_step (float): The maximum KL divergence between old and new
            policies, as in TRPO.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in garage.tf.optimizers.
        optimizer_args (dict): The arguments of the optimizer.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        encoder_ent_coeff (float): The coefficient of the policy encoder
            entropy. Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_ce_gradient (bool): Whether to stop the cross entropy gradient.
        inference (garage.tf.embeddings.StochasticEncoder): A encoder
            that infers the task embedding from state trajectory.
        inference_optimizer (object): The optimizer of the inference. Should be
            an optimizer in garage.tf.optimizers.
        inference_optimizer_args (dict): The arguments of the inference
            optimizer.
        inference_ce_coeff (float): The coefficient of the cross entropy of
            task embeddings inferred from task one-hot and state trajectory.
            This is effectively the coefficient of log-prob of inference.
        name (str): The name of the algorithm.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 sampler,
                 scope=None,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 encoder_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_ce_gradient=False,
                 inference=None,
                 inference_optimizer=None,
                 inference_optimizer_args=None,
                 inference_ce_coeff=0.0,
                 name='NPOTaskEmbedding',
                 task_update_frequency=1,
                 train_task_sampler=None):
        # pylint: disable=too-many-statements
        assert isinstance(policy, TaskEmbeddingPolicy)
        assert isinstance(inference, StochasticEncoder)
        
        self._task_update_frequency = task_update_frequency
        self._train_task_sampler = train_task_sampler

        self.policy = policy
        self._scope = scope
        self.max_episode_length = env_spec.max_episode_length

        self._env_spec = env_spec
        self._baseline = baseline
        self._discount = discount
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._fixed_horizon = fixed_horizon
        self._name = name
        self._name_scope = tf.name_scope(self._name)
        self._old_policy = policy.clone('old_policy')
        self._use_softplus_entropy = use_softplus_entropy
        self._stop_ce_gradient = stop_ce_gradient

        optimizer = optimizer or LBFGSOptimizer
        optimizer_args = optimizer_args or dict()

        inference_opt = inference_optimizer or LBFGSOptimizer
        inference_opt_args = inference_optimizer_args or dict()

        with self._name_scope:
            self._optimizer = optimizer(**optimizer_args)
            self._lr_clip_range = float(lr_clip_range)
            self._max_kl_step = float(max_kl_step)
            self._policy_ent_coeff = float(policy_ent_coeff)

            self._inference = inference
            self._old_inference = inference.clone('old_inference')
            self.inference_ce_coeff = float(inference_ce_coeff)
            self.inference_optimizer = inference_opt(**inference_opt_args)
            self.encoder_ent_coeff = encoder_ent_coeff

        self._f_rewards = None
        self._f_returns = None
        self._f_policy_kl = None
        self._f_policy_entropy = None
        self._f_encoder_kl = None
        self._f_encoder_entropy = None
        self._f_task_entropies = None
        self._f_inference_ce = None
        self._policy_network = None
        self._old_policy_network = None
        self._encoder_network = None
        self._old_encoder_network = None
        self._infer_network = None
        self._old_infer_network = None

        self._sampler = sampler

        self._init_opt()

    def _init_opt(self):
        """Initialize optimizater.

        Raises:
            NotImplementedError: Raise if the policy is recurrent.

        """
        # Input variables
        (pol_loss_inputs, pol_opt_inputs, infer_loss_inputs,
         infer_opt_inputs) = self._build_inputs()

        self._policy_opt_inputs = pol_opt_inputs
        self._inference_opt_inputs = infer_opt_inputs

        # Jointly optimize policy and encoder network
        pol_loss, pol_kl, _ = self._build_policy_loss(pol_loss_inputs)
        self._optimizer.update_opt(loss=pol_loss,
                                   target=self.policy,
                                   leq_constraint=(pol_kl, self._max_kl_step),
                                   inputs=flatten_inputs(
                                       self._policy_opt_inputs),
                                   constraint_name='mean_kl')

        # Optimize inference distribution separately (supervised learning)
        infer_loss, _ = self._build_inference_loss(infer_loss_inputs)
        self.inference_optimizer.update_opt(loss=infer_loss,
                                            target=self._inference,
                                            inputs=flatten_inputs(
                                                self._inference_opt_inputs))

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Trainer is passed to give algorithm
                the access to trainer.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for i, _ in enumerate(trainer.step_epochs()):
            assert self._train_task_sampler is not None
            env_updates = None
            if (not i % self._task_update_frequency) or (self._task_update_frequency == 1):
                num_tasks = self.policy.task_space.flat_dim
                env_ups = self._train_task_sampler.sample(num_tasks)
                envs = [env_up() for env_up in env_ups]
                del self._sampler
                self._sampler = MultiEnvWrapper(envs,
                                                sample_strategy=round_robin_strategy,
                                                mode='vanilla')
            trainer.step_episode = trainer.obtain_episodes(trainer.step_itr, env_update=env_updates)
            last_return = self._train_once(trainer.step_itr,
                                           trainer.step_episode)
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
        undiscounted_returns = log_multitask_performance(itr,
                                               episodes,
                                               discount=self._discount)

        # Calculate baseline predictions
        baselines = []
        start = 0
        for length in episodes.lengths:
            stop = start + length
            baseline = self._baseline.predict(
                dict(observations=episodes.observations[start:stop],
                     tasks=episodes.env_infos['task_onehot'][start:stop],
                     latents=episodes.agent_infos['latent'][start:stop]))
            baselines.append(baseline)
            start = stop
        baselines = pad_batch_array(np.concatenate(baselines),
                                    episodes.lengths, self.max_episode_length)

        # Process trajectories
        embed_eps, embed_ep_infos = self._process_episodes(episodes)

        average_return = np.mean(undiscounted_returns)

        logger.log('Optimizing policy...')
        self._optimize_policy(itr, episodes, baselines, embed_eps,
                              embed_ep_infos)

        return average_return

    def _optimize_policy(self, itr, episodes, baselines, embed_eps,
                         embed_ep_infos):
        """Optimize policy.

        Args:
            itr (int): Iteration number.
            episodes (EpisodeBatch): Batch of episodes.
            baselines (np.ndarray): Baseline predictions.
            embed_eps (np.ndarray): Embedding episodes.
            embed_ep_infos (dict): Embedding distribution information.

        """
        del itr

        policy_opt_input_values = self._policy_opt_input_values(
            episodes, baselines, embed_eps)
        inference_opt_input_values = self._inference_opt_input_values(
            episodes, embed_eps, embed_ep_infos)

        self._train_policy_and_encoder_networks(policy_opt_input_values)
        self._train_inference_network(inference_opt_input_values)

        # paths = samples_data['paths']
        fit_paths = self._evaluate(policy_opt_input_values, episodes,
                                   baselines, embed_ep_infos)
        self._visualize_distribution()

        logger.log('Fitting baseline...')
        self._baseline.fit(fit_paths)

        self._old_policy.parameters = self.policy.parameters
        self._old_policy.encoder.model.parameters = (
            self.policy.encoder.model.parameters)
        self._old_inference.model.parameters = self._inference.model.parameters

    def _process_episodes(self, episodes):
        # pylint: disable=too-many-statements
        """Return processed sample data based on the collected paths.

        Args:
            episodes (EpisodeBatch): Batch of episodes.

        Returns:
            np.ndarray: Embedding episodes.
            dict: Embedding distribution information.
                * mean (list[numpy.ndarray]): Means of the distribution.
                * log_std (list[numpy.ndarray]): Log standard deviations of the
                    distribution.

        """
        max_episode_length = self.max_episode_length

        trajectories = []
        trajectory_infos = []

        for obs in episodes.padded_observations:
            # - Calculate a forward-looking sliding window.
            # - If step_space has shape (n, d), then trajs will have shape
            #   (n, window, d)
            # - The length of the sliding window is determined by the
            #   trajectory inference spec. We smear the last few elements to
            #   preserve the time dimension.
            # - Only observation is used for a single step.
            #   Alternatively, stacked [observation, action] can be used for
            #   in harder tasks.
            obs_flat = self._env_spec.observation_space.flatten_n(obs)
            steps = obs_flat
            window = self._inference.spec.input_space.shape[0]
            traj = sliding_window(steps, window, smear=True)
            traj_flat = self._inference.spec.input_space.flatten_n(traj)
            trajectories.append(traj_flat)

            _, traj_info = self._inference.get_latents(traj_flat)
            trajectory_infos.append(traj_info)

        trajectories = np.stack(trajectories)
        trajectory_infos = stack_tensor_dict_list(
            [pad_tensor_dict(p, max_episode_length) for p in trajectory_infos])

        return trajectories, trajectory_infos

    def _build_inputs(self):
        """Build input variables.

        Returns:
            namedtuple: Collection of variables to compute policy loss.
            namedtuple: Collection of variables to do policy optimization.
            namedtuple: Collection of variables to compute inference loss.
            namedtuple: Collection of variables to do inference optimization.

        """
        # pylint: disable=too-many-statements
        observation_space = self.policy.observation_space
        action_space = self.policy.action_space
        task_space = self.policy.task_space
        latent_space = self.policy.latent_space
        trajectory_space = self._inference.spec.input_space

        with tf.name_scope('inputs'):
            obs_var = observation_space.to_tf_placeholder(name='obs',
                                                          batch_dims=2)
            task_var = tf.compat.v1.placeholder(
                tf.float32,
                shape=[None, None, task_space.flat_dim],
                name='task')
            trajectory_var = tf.compat.v1.placeholder(
                tf.float32, shape=[None, None, trajectory_space.flat_dim])
            latent_var = tf.compat.v1.placeholder(
                tf.float32, shape=[None, None, latent_space.flat_dim])

            action_var = action_space.to_tf_placeholder(name='action',
                                                        batch_dims=2)
            reward_var = tf.compat.v1.placeholder(tf.float32,
                                                  shape=[None, None],
                                                  name='reward')
            baseline_var = tf.compat.v1.placeholder(tf.float32,
                                                    shape=[None, None],
                                                    name='baseline')

            valid_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=[None, None],
                                                 name='valid')

            # Policy state (for RNNs)
            policy_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name=k)
                for k, shape in self.policy.state_info_specs
            }
            policy_state_info_vars_list = [
                policy_state_info_vars[k] for k in self.policy.state_info_keys
            ]

            # Encoder state (for RNNs)
            embed_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name='embed_%s' % k)
                for k, shape in self.policy.encoder.state_info_specs
            }
            embed_state_info_vars_list = [
                embed_state_info_vars[k]
                for k in self.policy.encoder.state_info_keys
            ]

            # Inference distribution state (for RNNs)
            infer_state_info_vars = {
                k: tf.compat.v1.placeholder(tf.float32,
                                            shape=[None] * 2 + list(shape),
                                            name='infer_%s' % k)
                for k, shape in self._inference.state_info_specs
            }
            infer_state_info_vars_list = [
                infer_state_info_vars[k]
                for k in self._inference.state_info_keys
            ]

        extra_obs_var = [
            tf.cast(v, tf.float32) for v in policy_state_info_vars_list
        ]
        # Pylint false alarm
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        augmented_obs_var = tf.concat([obs_var] + extra_obs_var, axis=-1)
        extra_traj_var = [
            tf.cast(v, tf.float32) for v in infer_state_info_vars_list
        ]
        augmented_traj_var = tf.concat([trajectory_var] + extra_traj_var, -1)

        # Policy and encoder network loss and optimizer inputs
        policy_loss_inputs = graph_inputs(
            'PolicyLossInputs',
            augmented_obs_var=augmented_obs_var,
            augmented_traj_var=augmented_traj_var,
            task_var=task_var,
            latent_var=latent_var,
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            valid_var=valid_var)
        policy_opt_inputs = graph_inputs(
            'PolicyOptInputs',
            obs_var=obs_var,
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            trajectory_var=trajectory_var,
            task_var=task_var,
            latent_var=latent_var,
            valid_var=valid_var,
            policy_state_info_vars_list=policy_state_info_vars_list,
            embed_state_info_vars_list=embed_state_info_vars_list,
        )

        # Inference network loss and optimizer inputs
        inference_loss_inputs = graph_inputs('InferenceLossInputs',
                                             latent_var=latent_var,
                                             valid_var=valid_var)
        inference_opt_inputs = graph_inputs(
            'InferenceOptInputs',
            latent_var=latent_var,
            trajectory_var=trajectory_var,
            valid_var=valid_var,
            infer_state_info_vars_list=infer_state_info_vars_list,
        )

        return (policy_loss_inputs, policy_opt_inputs, inference_loss_inputs,
                inference_opt_inputs)

    def _build_policy_loss(self, i):
        """Build policy loss and other output tensors.

        Args:
            i (namedtuple): Collection of variables to compute policy loss.

        Returns:
            tf.Tensor: Policy loss.
            tf.Tensor: Mean policy KL divergence.

        """
        # pylint: disable=too-many-statements
        self._policy_network, self._encoder_network = (self.policy.build(
            i.augmented_obs_var, i.task_var, name='loss_policy'))
        self._old_policy_network, self._old_encoder_network = (
            self._old_policy.build(i.augmented_obs_var,
                                   i.task_var,
                                   name='loss_old_policy'))
        self._infer_network = self._inference.build(i.augmented_traj_var,
                                                    name='loss_infer')
        self._old_infer_network = self._old_inference.build(
            i.augmented_traj_var, name='loss_old_infer')

        pol_dist = self._policy_network.dist
        old_pol_dist = self._old_policy_network.dist

        # Entropy terms
        encoder_entropy, inference_ce, policy_entropy = (
            self._build_entropy_terms(i))

        # Augment the path rewards with entropy terms
        with tf.name_scope('augmented_rewards'):
            rewards = (i.reward_var -
                       (self.inference_ce_coeff * inference_ce) +
                       (self._policy_ent_coeff * policy_entropy))

        with tf.name_scope('policy_loss'):
            with tf.name_scope('advantages'):
                adv = compute_advantages(self._discount,
                                         self._gae_lambda,
                                         self.max_episode_length,
                                         i.baseline_var,
                                         rewards,
                                         name='advantages')
                adv = tf.reshape(adv, [-1, self.max_episode_length])

            # Optionally normalize advantages
            eps = tf.constant(1e-8, dtype=tf.float32)
            if self._center_adv:
                adv = center_advs(adv, axes=[0], eps=eps)

            if self._positive_adv:
                adv = positive_advs(adv, eps)

            # Calculate loss function and KL divergence
            with tf.name_scope('kl'):
                kl = old_pol_dist.kl_divergence(pol_dist)
                pol_mean_kl = tf.reduce_mean(kl)

            ll = pol_dist.log_prob(i.action_var, name='log_likelihood')

            # Calculate surrogate loss
            with tf.name_scope('surr_loss'):
                old_ll = old_pol_dist.log_prob(i.action_var)
                old_ll = tf.stop_gradient(old_ll)
                # Clip early to avoid overflow
                lr = tf.exp(
                    tf.minimum(ll - old_ll, np.log(1 + self._lr_clip_range)))

                surrogate = lr * adv

                surrogate = tf.debugging.check_numerics(surrogate,
                                                        message='surrogate')

            # Finalize objective function
            with tf.name_scope('loss'):
                lr_clip = tf.clip_by_value(lr,
                                           1 - self._lr_clip_range,
                                           1 + self._lr_clip_range,
                                           name='lr_clip')
                surr_clip = lr_clip * adv
                obj = tf.minimum(surrogate, surr_clip, name='surr_obj')
                obj = tf.boolean_mask(obj, i.valid_var)
                # Maximize E[surrogate objective] by minimizing
                # -E_t[surrogate objective]
                loss = -tf.reduce_mean(obj)

                # Encoder entropy bonus
                loss -= self.encoder_ent_coeff * encoder_entropy

            encoder_mean_kl = self._build_encoder_kl()

            # Diagnostic functions
            self._f_policy_kl = tf.compat.v1.get_default_session(
            ).make_callable(pol_mean_kl,
                            feed_list=flatten_inputs(self._policy_opt_inputs))

            self._f_rewards = tf.compat.v1.get_default_session().make_callable(
                rewards, feed_list=flatten_inputs(self._policy_opt_inputs))

            returns = discounted_returns(self._discount,
                                         self.max_episode_length,
                                         rewards,
                                         name='returns')
            self._f_returns = tf.compat.v1.get_default_session().make_callable(
                returns, feed_list=flatten_inputs(self._policy_opt_inputs))

        return loss, pol_mean_kl, encoder_mean_kl

    def _build_entropy_terms(self, i):
        """Build policy entropy tensor.

        Args:
            i (namedtuple): Collection of variables to compute policy loss.

        Returns:
            tf.Tensor: Policy entropy.

        """
        pol_dist = self._policy_network.dist
        infer_dist = self._infer_network.dist
        enc_dist = self._encoder_network.dist
        with tf.name_scope('entropy_terms'):
            # 1. Encoder distribution total entropy
            with tf.name_scope('encoder_entropy'):
                encoder_dist, _, _ = self.policy.encoder.build(
                    i.task_var, name='encoder_entropy').outputs
                encoder_all_task_entropies = -encoder_dist.log_prob(
                    i.latent_var)

                if self._use_softplus_entropy:
                    encoder_entropy = tf.nn.softplus(
                        encoder_all_task_entropies)

                encoder_entropy = tf.reduce_mean(encoder_entropy,
                                                 name='encoder_entropy')
                encoder_entropy = tf.stop_gradient(encoder_entropy)

            # 2. Infernece distribution cross-entropy (log-likelihood)
            with tf.name_scope('inference_ce'):
                # Build inference with trajectory windows

                traj_ll = infer_dist.log_prob(
                    enc_dist.sample(seed=deterministic.get_tf_seed_stream()),
                    name='traj_ll')

                inference_ce_raw = -traj_ll
                inference_ce = tf.clip_by_value(inference_ce_raw, -3, 3)

                if self._use_softplus_entropy:
                    inference_ce = tf.nn.softplus(inference_ce)

                if self._stop_ce_gradient:
                    inference_ce = tf.stop_gradient(inference_ce)

            # 3. Policy path entropies
            with tf.name_scope('policy_entropy'):
                policy_entropy = -pol_dist.log_prob(i.action_var,
                                                    name='policy_log_likeli')

                # This prevents entropy from becoming negative
                # for small policy std
                if self._use_softplus_entropy:
                    policy_entropy = tf.nn.softplus(policy_entropy)

                policy_entropy = tf.stop_gradient(policy_entropy)

        # Diagnostic functions
        self._f_task_entropies = compile_function(
            flatten_inputs(self._policy_opt_inputs),
            encoder_all_task_entropies)
        self._f_encoder_entropy = compile_function(
            flatten_inputs(self._policy_opt_inputs), encoder_entropy)
        self._f_inference_ce = compile_function(
            flatten_inputs(self._policy_opt_inputs),
            tf.reduce_mean(inference_ce * i.valid_var))
        self._f_policy_entropy = compile_function(
            flatten_inputs(self._policy_opt_inputs), policy_entropy)

        return encoder_entropy, inference_ce, policy_entropy

    def _build_encoder_kl(self):
        """Build graph for encoder KL divergence.

        Returns:
            tf.Tensor: Encoder KL divergence.

        """
        dist = self._encoder_network.dist
        old_dist = self._old_encoder_network.dist

        with tf.name_scope('encoder_kl'):
            kl = old_dist.kl_divergence(dist)
            mean_kl = tf.reduce_mean(kl)

            # Diagnostic function
            self._f_encoder_kl = compile_function(
                flatten_inputs(self._policy_opt_inputs), mean_kl)

            return mean_kl

    def _build_inference_loss(self, i):
        """Build loss function for the inference network.

        Args:
            i (namedtuple): Collection of variables to compute inference loss.

        Returns:
            tf.Tensor: Inference loss.

        """
        dist = self._infer_network.dist
        old_dist = self._old_infer_network.dist
        with tf.name_scope('infer_loss'):

            traj_ll = dist.log_prob(i.latent_var, name='traj_ll_2')

            # Calculate loss
            traj_gammas = tf.constant(float(self._discount),
                                      dtype=tf.float32,
                                      shape=[self.max_episode_length])
            # Pylint false alarm
            # pylint: disable=no-value-for-parameter
            traj_discounts = tf.compat.v1.cumprod(traj_gammas,
                                                  exclusive=True,
                                                  name='traj_discounts')
            discount_traj_ll = traj_discounts * traj_ll
            discount_traj_ll = tf.boolean_mask(discount_traj_ll, i.valid_var)

            with tf.name_scope('loss'):
                infer_loss = -tf.reduce_mean(discount_traj_ll,
                                             name='infer_loss')

            with tf.name_scope('kl'):
                # Calculate predicted encoder distributions for each timestep

                # Calculate KL divergence
                kl = old_dist.kl_divergence(dist)
                infer_kl = tf.reduce_mean(kl, name='infer_kl')

            return infer_loss, infer_kl

    def _policy_opt_input_values(self, episodes, baselines, embed_eps):
        """Map episode samples to the policy optimizer inputs.

        Args:
            episodes (EpisodeBatch): Batch of episodes.
            baselines (np.ndarray): Baseline predictions.
            embed_eps (np.ndarray): Embedding episodes.

        Returns:
            list(np.ndarray): Flatten policy optimization input values.

        """
        actions = [
            self._env_spec.action_space.flatten_n(act)
            for act in episodes.actions_list
        ]
        actions = pad_batch_array(np.concatenate(actions), episodes.lengths,
                                  self.max_episode_length)
        tasks = pad_batch_array(episodes.env_infos['task_onehot'],
                                episodes.lengths, self.max_episode_length)
        latents = pad_batch_array(episodes.agent_infos['latent'],
                                  episodes.lengths, self.max_episode_length)

        agent_infos = episodes.padded_agent_infos
        policy_state_info_list = [
            agent_infos[k] for k in self.policy.state_info_keys
        ]
        embed_state_info_list = [
            agent_infos['latent_' + k]
            for k in self.policy.encoder.state_info_keys
        ]
        # pylint: disable=unexpected-keyword-arg
        policy_opt_input_values = self._policy_opt_inputs._replace(
            obs_var=episodes.padded_observations,
            action_var=actions,
            reward_var=episodes.padded_rewards,
            baseline_var=baselines,
            trajectory_var=embed_eps,
            task_var=tasks,
            latent_var=latents,
            valid_var=episodes.valids,
            policy_state_info_vars_list=policy_state_info_list,
            embed_state_info_vars_list=embed_state_info_list,
        )

        return flatten_inputs(policy_opt_input_values)

    def _inference_opt_input_values(self, episodes, embed_eps, embed_ep_infos):
        """Map episode samples to the inference optimizer inputs.

        Args:
            episodes (EpisodeBatch): Batch of episodes.
            embed_eps (np.ndarray): Embedding episodes.
            embed_ep_infos (dict): Embedding distribution information.

        Returns:
            list(np.ndarray): Flatten inference optimization input values.

        """
        latents = pad_batch_array(episodes.agent_infos['latent'],
                                  episodes.lengths, self.max_episode_length)

        infer_state_info_list = [
            embed_ep_infos[k] for k in self._inference.state_info_keys
        ]
        # pylint: disable=unexpected-keyword-arg
        inference_opt_input_values = self._inference_opt_inputs._replace(
            latent_var=latents,
            trajectory_var=embed_eps,
            valid_var=episodes.valids,
            infer_state_info_vars_list=infer_state_info_list,
        )

        return flatten_inputs(inference_opt_input_values)

    def _evaluate(self, policy_opt_input_values, episodes, baselines,
                  embed_ep_infos):
        """Evaluate rewards and everything else.

        Args:
            policy_opt_input_values (list[np.ndarray]): Flattened
                policy optimization input values.
            episodes (EpisodeBatch): Batch of episodes.
            baselines (np.ndarray): Baseline predictions.
            embed_ep_infos (dict): Embedding distribution information.

        Returns:
            dict: Paths for fitting the baseline.

        """
        # pylint: disable=too-many-statements
        fit_paths = []
        valids = episodes.valids
        observations = episodes.padded_observations
        tasks = pad_batch_array(episodes.env_infos['task_onehot'],
                                episodes.lengths, self.max_episode_length)
        latents = pad_batch_array(episodes.agent_infos['latent'],
                                  episodes.lengths, self.max_episode_length)
        baselines_list = []
        for baseline, valid in zip(baselines, valids):
            baselines_list.append(baseline[valid.astype(np.bool)])

        # Augment reward from baselines
        rewards_tensor = self._f_rewards(*policy_opt_input_values)
        returns_tensor = self._f_returns(*policy_opt_input_values)
        returns_tensor = np.squeeze(returns_tensor, -1)

        env_rewards = episodes.rewards
        env_returns = [
            discount_cumsum(rwd, self._discount)
            for rwd in episodes.padded_rewards
        ]
        env_average_discounted_return = np.mean(
            [ret[0] for ret in env_returns])

        # Recompute returns and prepare paths for fitting the baseline
        aug_rewards = []
        aug_returns = []
        for rew, ret, val, task, latent, obs in zip(rewards_tensor,
                                                    returns_tensor, valids,
                                                    tasks, latents,
                                                    observations):
            returns = ret[val.astype(np.bool)]
            task = task[val.astype(np.bool)]
            latent = latent[val.astype(np.bool)]
            obs = obs[val.astype(np.bool)]

            aug_rewards.append(rew[val.astype(np.bool)])
            aug_returns.append(returns)
            fit_paths.append(
                dict(observations=obs,
                     tasks=task,
                     latents=latent,
                     returns=returns))
        aug_rewards = concat_tensor_list(aug_rewards)
        aug_returns = concat_tensor_list(aug_returns)

        # Calculate effect of the entropy terms
        d_rewards = np.mean(aug_rewards - env_rewards)
        tabular.record('{}/EntRewards'.format(self.policy.name), d_rewards)

        aug_average_discounted_return = (np.mean(
            [ret[0] for ret in returns_tensor]))
        d_returns = np.mean(aug_average_discounted_return -
                            env_average_discounted_return)
        tabular.record('{}/EntReturns'.format(self.policy.name), d_returns)

        # Calculate explained variance
        ev = explained_variance_1d(np.concatenate(baselines_list), aug_returns)
        tabular.record('{}/ExplainedVariance'.format(self._baseline.name), ev)

        inference_rmse = (embed_ep_infos['mean'] - latents)**2.
        inference_rmse = np.sqrt(inference_rmse.mean())
        tabular.record('Inference/RMSE', inference_rmse)

        inference_rrse = rrse(latents, embed_ep_infos['mean'])
        tabular.record('Inference/RRSE', inference_rrse)

        embed_ent = self._f_encoder_entropy(*policy_opt_input_values)
        tabular.record('{}/Encoder/Entropy'.format(self.policy.name),
                       embed_ent)

        infer_ce = self._f_inference_ce(*policy_opt_input_values)
        tabular.record('Inference/CrossEntropy', infer_ce)

        pol_ent = self._f_policy_entropy(*policy_opt_input_values)
        pol_ent = np.sum(pol_ent) / np.sum(episodes.lengths)
        tabular.record('{}/Entropy'.format(self.policy.name), pol_ent)

        task_ents = self._f_task_entropies(*policy_opt_input_values)
        tasks = tasks[:, 0, :]
        _, task_indices = np.nonzero(tasks)
        path_lengths = np.sum(valids, axis=1)
        for t in range(self.policy.task_space.flat_dim):
            lengths = path_lengths[task_indices == t]
            completed = lengths < self.max_episode_length
            pct_completed = (np.mean(completed) if completed.size > 0
                else 0)
            mean_lengths = (np.mean(lengths) if lengths.size > 0
                else 0)
            tabular.record('Tasks/EpisodeLength/t={}'.format(t),
                           mean_lengths)
            tabular.record('Tasks/TerminationRate/t={}'.format(t),
                           pct_completed)
            tabular.record('Tasks/Entropy/t={}'.format(t), task_ents[t])

        return fit_paths

    def _visualize_distribution(self):
        """Visualize encoder distribution."""
        num_tasks = self.policy.task_space.flat_dim
        all_tasks = np.eye(num_tasks, num_tasks)
        _, latent_infos = self.policy.encoder.get_latents(all_tasks)

        for task in range(num_tasks):
            for i in range(self.policy.latent_space.flat_dim):
                stds = np.exp(latent_infos['log_std'][task, i])

                norm = scipy.stats.norm(loc=latent_infos['mean'][task, i],
                                        scale=stds)
                samples = norm.rvs(100)
                hist = Histogram(samples)
                tabular.record('Encoder/task={},i={}'.format(task, i), hist)

    def _train_policy_and_encoder_networks(self, policy_opt_input_values):
        """Joint optimization of policy and encoder networks.

        Args:
            policy_opt_input_values (list(np.ndarray)): Flatten policy
                optimization input values.

        Returns:
            float: Policy loss after optimization.

        """
        logger.log('Computing loss before')

        loss_before = self._optimizer.loss(policy_opt_input_values)

        logger.log('Computing KL before')
        policy_kl_before = self._f_policy_kl(*policy_opt_input_values)
        embed_kl_before = self._f_encoder_kl(*policy_opt_input_values)

        logger.log('Optimizing')
        self._optimizer.optimize(policy_opt_input_values)

        logger.log('Computing KL after')
        policy_kl = self._f_policy_kl(*policy_opt_input_values)
        embed_kl = self._f_encoder_kl(*policy_opt_input_values)

        logger.log('Computing loss after')
        loss_after = self._optimizer.loss(policy_opt_input_values)
        tabular.record('{}/LossBefore'.format(self.policy.name), loss_before)
        tabular.record('{}/LossAfter'.format(self.policy.name), loss_after)
        tabular.record('{}/dLoss'.format(self.policy.name),
                       loss_before - loss_after)
        tabular.record('{}/KLBefore'.format(self.policy.name),
                       policy_kl_before)
        tabular.record('{}/KL'.format(self.policy.name), policy_kl)
        tabular.record('{}/Encoder/KLBefore'.format(self.policy.name),
                       embed_kl_before)
        tabular.record('{}/Encoder/KL'.format(self.policy.name), embed_kl)

        return loss_after

    def _train_inference_network(self, inference_opt_input_values):
        """Optimize inference network.

        Args:
            inference_opt_input_values (list(np.ndarray)): Flatten inference
                optimization input values.

        Returns:
            float: Inference loss after optmization.

        """
        logger.log('Optimizing inference network...')
        infer_loss_before = self.inference_optimizer.loss(
            inference_opt_input_values)
        tabular.record('Inference/Loss', infer_loss_before)
        self.inference_optimizer.optimize(inference_opt_input_values)
        infer_loss_after = self.inference_optimizer.loss(
            inference_opt_input_values)
        tabular.record('Inference/dLoss', infer_loss_before - infer_loss_after)

        return infer_loss_after

    @classmethod
    def _get_latent_space(cls, latent_dim):
        """Get latent space given latent length.

        Args:
            latent_dim (int): Length of latent.

        Returns:
            akro.Space: Space of latent.

        """
        latent_lb = np.zeros(latent_dim, )
        latent_up = np.ones(latent_dim, )
        return akro.Box(latent_lb, latent_up)

    @classmethod
    def get_encoder_spec(cls, task_space, latent_dim):
        """Get the embedding spec of the encoder.

        Args:
            task_space (akro.Space): Task spec.
            latent_dim (int): Latent dimension.

        Returns:
            garage.InOutSpec: Encoder spec.

        """
        latent_space = cls._get_latent_space(latent_dim)
        return InOutSpec(task_space, latent_space)

    @classmethod
    def get_infer_spec(cls, env_spec, latent_dim, inference_window_size):
        """Get the embedding spec of the inference.

        Every `inference_window_size` timesteps in the trajectory will be used
        as the inference network input.

        Args:
            env_spec (garage.envs.EnvSpec): Environment spec.
            latent_dim (int): Latent dimension.
            inference_window_size (int): Length of inference window.

        Returns:
            garage.InOutSpec: Inference spec.

        """
        latent_space = cls._get_latent_space(latent_dim)

        obs_lb, obs_ub = env_spec.observation_space.bounds
        obs_lb_flat = env_spec.observation_space.flatten(obs_lb)
        obs_ub_flat = env_spec.observation_space.flatten(obs_ub)
        traj_lb = np.stack([obs_lb_flat] * inference_window_size)
        traj_ub = np.stack([obs_ub_flat] * inference_window_size)
        traj_space = akro.Box(traj_lb, traj_ub)

        return InOutSpec(traj_space, latent_space)

    def __getstate__(self):
        """Parameters to save in snapshot.

        Returns:
            dict: Parameters to save.

        """
        data = self.__dict__.copy()
        del data['_name_scope']
        del data['_inference_opt_inputs']
        del data['_policy_opt_inputs']
        del data['_f_inference_ce']
        del data['_f_task_entropies']
        del data['_f_encoder_entropy']
        del data['_f_encoder_kl']
        del data['_f_policy_entropy']
        del data['_f_policy_kl']
        del data['_f_rewards']
        del data['_f_returns']
        del data['_policy_network']
        del data['_old_policy_network']
        del data['_encoder_network']
        del data['_old_encoder_network']
        del data['_infer_network']
        del data['_old_infer_network']
        return data

    def __setstate__(self, state):
        """Parameters to restore from snapshot.

        Args:
            state (dict): Parameters to restore from.

        """
        self.__dict__ = state
        self._name_scope = tf.name_scope(self._name)
        self._init_opt()
