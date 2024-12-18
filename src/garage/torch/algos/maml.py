"""Model-Agnostic Meta-Learning (MAML) algorithm implementation for RL."""
# yapf: disable
import collections
import copy

from dowel import tabular, logger
import inspect
import numpy as np
import os
import torch

from garage import (_Default, EpisodeBatch, log_multitask_performance,
                    make_optimizer)
from garage.np import discount_cumsum, explained_variance_1d
from garage.torch import update_module_params
from garage.torch._functions import zero_optim_grads
from garage.torch.optimizers import (ConjugateGradientOptimizer,
                                     DifferentiableSGD)
import wandb
# yapf: enable


class MAML:
    """Model-Agnostic Meta-Learning (MAML).

    Args:
        inner_algo (garage.torch.algos.VPG): The inner algorithm used for
            computing loss.
        env (Environment): An environment.
        policy (garage.torch.policies.Policy): Policy.
        sampler (garage.sampler.Sampler): Sampler.
        task_sampler (garage.experiment.TaskSampler): Task sampler.
        meta_optimizer (Union[torch.optim.Optimizer, tuple]):
            Type of optimizer.
            This can be an optimizer type such as `torch.optim.Adam` or a tuple
            of type and dictionary, where dictionary contains arguments to
            initialize the optimizer e.g. `(torch.optim.Adam, {'lr' : 1e-3})`.
        meta_batch_size (int): Number of tasks sampled per batch.
        inner_lr (float): Adaptation learning rate.
        outer_lr (float): Meta policy learning rate.
        num_grad_updates (int): Number of adaptation gradient steps.
        meta_evaluator (MetaEvaluator): A meta evaluator for meta-testing. If
            None, don't do meta-testing.
        evaluate_every_n_epochs (int): Do meta-testing every this epochs.

    """

    def __init__(self,
                 inner_algo,
                 env,
                 policy,
                 sampler,
                 task_sampler,
                 meta_optimizer,
                 meta_batch_size=40,
                 inner_lr=0.1,
                 outer_lr=1e-3,
                 num_grad_updates=1,
                 meta_evaluator=None,
                 evaluate_every_n_epochs=1,
                 w_and_b=False,
                 constraint=False,
                 train_constraint=None,
                 lr_constraint=None,
                 constraint_threshold=None,
                 save_state=False,
                 state_dir=None,
                 validation_evaluator=None,
                 ):
        self._sampler = sampler

        self.max_episode_length = inner_algo.max_episode_length

        self._meta_evaluator = meta_evaluator
        self._policy = policy
        self._env = env
        self._task_sampler = task_sampler
        self._value_function = inner_algo._value_function
        self._value_function_const = inner_algo._value_function_const
        self._num_grad_updates = num_grad_updates
        self._meta_batch_size = meta_batch_size
        self._inner_algo = inner_algo
        self._inner_optimizer = DifferentiableSGD(self._policy, lr=inner_lr)
        self._meta_optimizer = make_optimizer(meta_optimizer,
                                              module=policy,
                                              lr=_Default(outer_lr),
                                              eps=_Default(1e-5))
        self._evaluate_every_n_epochs = evaluate_every_n_epochs
        self._w_and_b=w_and_b
        self._constraint = constraint
        if constraint and train_constraint:
            self._optimizer_lagrangian = torch.optim.Adam(
                [policy.lagrangian], lr=lr_constraint)
            self._constraint_threshold = constraint_threshold
        self._train_constraint = train_constraint
        self._save_state = save_state
        self._state_dir = state_dir
        if save_state:
            assert state_dir is not None, "specify a dir to save model and optimizer"
        self._best_success_rate_test = float('-inf')
        self._best_success_rate_train = float('-inf')
        if self._train_constraint:
            self._lowest_violation_test = float('inf')
            self._lowest_violation_train = float('inf')
        self._validation_evaluator = validation_evaluator
        self._threshold_met_test = False
        self._threshold_met_train = False
    def train(self, trainer):
        """Obtain samples and start training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in trainer.step_epochs():
            all_samples, all_params = self._obtain_samples(trainer)
            last_return = self._train_once(trainer, all_samples, all_params)
            trainer.step_itr += 1

        return last_return

    def _train_once(self, trainer, all_samples, all_params):
        """Train the algorithm once.

        Args:
            trainer (Trainer): The experiment runner.
            all_samples (list[list[_MAMLEpisodeBatch]]): A two
                dimensional list of _MAMLEpisodeBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            all_params (list[dict]): A list of named parameter dictionaries.
                Each dictionary contains key value pair of names (str) and
                parameters (torch.Tensor).

        Returns:
            float: Average return.

        """
        itr = trainer.step_itr
        old_theta = dict(self._policy.named_parameters())

        kl_before = self._compute_kl_constraint(all_samples,
                                                all_params,
                                                set_grad=False)

        meta_objective = self._compute_meta_loss(all_samples, all_params)

        zero_optim_grads(self._meta_optimizer)
        meta_objective.backward()

        self._meta_optimize(all_samples, all_params)

        # Log
        loss_after = self._compute_meta_loss(all_samples,
                                             all_params,
                                             set_grad=False)
        kl_after = self._compute_kl_constraint(all_samples,
                                               all_params,
                                               set_grad=False)

        baselines_list = []
        returns = []
        for task in all_samples:
            for rollout in task:
                baselines_list.append(rollout.baselines[rollout.valids.bool()])
                returns.append([r['returns'] for r in rollout.paths])
        baselines = torch.cat(baselines_list).numpy().flatten()
        ev = explained_variance_1d(baselines, np.array(returns).flatten())

        if self._constraint:
            baselines_const_list = []
            const_violations_pre = []
            const_violations_post = []
            cum_penalties = []
            for task in all_samples:
                for i in range(len(task)):
                    if i < self._num_grad_updates:
                        const_violations_pre.append(task[i].const_violations.sum(axis=-1))
                    else:
                        const_violations_post.append(task[i].const_violations.sum(axis=-1))
                    if self._train_constraint:
                        baselines_const_list.append(
                            task[i].baselines_const[task[i].valids.bool()])
                    cum_penalties.append(
                        [r['penalties'] for r in task[i].paths])
            avg_const_violation_pre = torch.cat(const_violations_pre).mean()
            avg_const_violation_post = torch.cat(const_violations_post).mean()
            if self._train_constraint:
                baselines_const = torch.cat(
                    baselines_const_list).numpy().flatten()
                ev_const = explained_variance_1d(baselines_const, np.array(
                    cum_penalties).flatten())
                self._optimizer_lagrangian.zero_grad()

                # After adaptation, violating constraints should be penalized
                # more than before.
                penalty_pre = 0.5*(avg_const_violation_pre/self.max_episode_length -  self._constraint_threshold)
                penalty_post = 0.5*(avg_const_violation_post / self.max_episode_length - self._constraint_threshold)
                lagrangian_loss = -self.policy.lagrangian*(
                    penalty_pre + penalty_post
                )
                lagrangian_loss.backward()
                self._optimizer_lagrangian.step()
                with torch.no_grad():
                    self.policy.lagrangian.data.clamp_(min=0)
                if wandb.run:
                    wandb.log({"lambda": self.policy.lagrangian.item()}, step=wandb.run.step)

        with torch.no_grad():
            policy_entropy = self._compute_policy_entropy(
                [task_samples[0] for task_samples in all_samples])
            stddev = self._compute_policy_stddev(
                [task_samples[0] for task_samples in all_samples])
            to_log = [itr, all_samples, meta_objective.item(), loss_after.item(),
                kl_before.item(), kl_after.item(),
                policy_entropy.mean().item(),
                stddev.mean().item(), ev]


            if self._train_constraint:
                to_log.extend([ev_const, self.policy.lagrangian.item()])
            average_return = self._log_performance(*to_log)
        success_rate_train = tabular.as_dict[
            "post_adaptation/Average/SuccessRate"]
        if self._train_constraint:
            const_violation_train = tabular.as_dict[
                "post_adaptation/Average/Constraint"]
            if const_violation_train < self._constraint_threshold:
                self._threshold_met_train = True  # Mark that threshold condition is met
                if success_rate_train > self._best_success_rate_train:
                    logger.log(f"new best model. Success Rate: {success_rate_train}, constraint violation: {const_violation_train}")
                    self._best_success_rate_train = success_rate_train
                    self.save_model(itr, "best_model_train")

            elif not self._threshold_met_train:  # Save lowest violation only if threshold has never been met
                if const_violation_train < self._lowest_violation_train:
                    logger.log(
                        f"new best model. Success Rate: {success_rate_train}, constraint violation: {const_violation_train}")
                    self._lowest_violation_train = const_violation_train
                    self.save_model(itr, "low_viol_model_train")

        else:
            if success_rate_train > self._best_success_rate_train:
                logger.log(
                    f"new best model. Success Rate: {success_rate_train}")
                self._best_success_rate_train = success_rate_train
                self.save_model(itr, "best_model_no_constraint_train")


        if self._meta_evaluator and itr % self._evaluate_every_n_epochs == 0:
            if "itr_multiplier" in inspect.signature(
                self._meta_evaluator.evaluate).parameters:
                results = self._meta_evaluator.evaluate(self,
                                                        itr_multiplier=self._evaluate_every_n_epochs)
            else:
                results = self._meta_evaluator.evaluate(self)

            success_rate = results["success_rate"]
            if self._train_constraint:
                constraint_violation = results["constraint_violations"]
                if constraint_violation < self._constraint_threshold:
                    self._threshold_met_test = True  # Mark that threshold condition is met
                    if success_rate > self._best_success_rate_test:
                        logger.log(f"new best model. Success Rate: {success_rate}, constraint violation: {constraint_violation}")
                        self._best_success_rate_test = success_rate
                        self.save_model(itr, "best_model")
                        if self._validation_evaluator is not None:
                            logger.log("Checking Performance on Validation Ens")
                            self._validation_evaluator.evaluate(self, epoch=itr)

                elif not self._threshold_met_test:  # Save lowest violation only if threshold has never been met
                    if constraint_violation < self._lowest_violation_test:
                        logger.log(
                            f"new best model. Success Rate: {success_rate}, constraint violation: {constraint_violation}")
                        self._lowest_violation_test = constraint_violation
                        self.save_model(itr, "low_viol_model")
                        if self._validation_evaluator is not None:
                            logger.log("Checking Performance on Validation Env")
                            self._validation_evaluator.evaluate(self, epoch=itr)
            else:
                if success_rate > self._best_success_rate_test:
                    logger.log(
                        f"new best model. Success Rate: {success_rate}")
                    self._best_success_rate_test = success_rate
                    self.save_model(itr,  "best_model_no_constraint")
                    if self._validation_evaluator is not None:
                        logger.log("Checking Performance on Validation Env")
                        self._validation_evaluator.evaluate(self, epoch=itr)
        update_module_params(self._old_policy, old_theta)
        return average_return

    def save_model(self, itr, model_type):
        if not os.path.exists(self._state_dir):
            os.makedirs(self._state_dir)
        # Save policy parameters
        policy_params_path = f"{self._state_dir}/{model_type}_policy_params_epoch_{itr}.pt"
        torch.save(self.policy.state_dict(), policy_params_path)

        # Save optimizer state
        optimizer_state_path = f"{self._state_dir}/{model_type}_optimizer_state_epoch_{itr}.pt"
        torch.save(self._meta_optimizer.state_dict(), optimizer_state_path)

        # Log to Weights & Biases
        if self._w_and_b:
            wandb.log({
                "model_type": model_type,
                "policy_params": wandb.save(policy_params_path),
                "optimizer_state": wandb.save(optimizer_state_path)
            }, step=itr)


    def _obtain_samples(self, trainer):
        """Obtain samples for each task before and after the fast-adaptation.

        Args:
            trainer (Trainer): A trainer instance to obtain samples.

        Returns:
            tuple: Tuple of (all_samples, all_params).
                all_samples (list[_MAMLEpisodeBatch]): A list of size
                    [meta_batch_size * (num_grad_updates + 1)]
                all_params (list[dict]): A list of named parameter
                    dictionaries.

        """
        tasks = self._task_sampler.sample(self._meta_batch_size)
        all_samples = [[] for _ in range(len(tasks))]
        all_params = []
        theta = dict(self._policy.named_parameters())
        for i, env_up in enumerate(tasks):
            for j in range(self._num_grad_updates + 1):
                episodes = trainer.obtain_episodes(trainer.step_itr,
                                                   env_update=env_up)
                batch_samples = self._process_samples(episodes)
                all_samples[i].append(batch_samples)

                # The last iteration does only sampling but no adapting
                if j < self._num_grad_updates:
                    # A grad need to be kept for the next grad update
                    # Except for the last grad update
                    require_grad = j < self._num_grad_updates - 1
                    self._adapt(batch_samples, set_grad=require_grad)

            all_params.append(dict(self._policy.named_parameters()))
            # Restore to pre-updated policy
            update_module_params(self._policy, theta)

        return all_samples, all_params

    def _adapt(self, batch_samples, set_grad=True):
        """Performs one MAML inner step to update the policy.

        Args:
            batch_samples (_MAMLEpisodeBatch): Samples data for one
                task and one gradient step.
            set_grad (bool): if False, update policy parameters in-place.
                Else, allow taking gradient of functions of updated parameters
                with respect to pre-updated parameters.

        """
        # pylint: disable=protected-access
        loss = self._inner_algo._compute_loss(*batch_samples[1:])

        # Update policy parameters with one SGD step
        self._inner_optimizer.set_grads_none()
        loss.backward(create_graph=set_grad)

        with torch.set_grad_enabled(set_grad):
            self._inner_optimizer.step()

    def _meta_optimize(self, all_samples, all_params):
        if isinstance(self._meta_optimizer, ConjugateGradientOptimizer):
            self._meta_optimizer.step(
                f_loss=lambda: self._compute_meta_loss(
                    all_samples, all_params, set_grad=False),
                f_constraint=lambda: self._compute_kl_constraint(
                    all_samples, all_params))
        else:
            self._meta_optimizer.step(lambda: self._compute_meta_loss(
                all_samples, all_params, set_grad=False))

    def _compute_meta_loss(self, all_samples, all_params, set_grad=True):
        """Compute loss to meta-optimize.

        Args:
            all_samples (list[list[_MAMLEpisodeBatch]]): A two
                dimensional list of _MAMLEpisodeBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            all_params (list[dict]): A list of named parameter dictionaries.
                Each dictionary contains key value pair of names (str) and
                parameters (torch.Tensor).
            set_grad (bool): Whether to enable gradient calculation or not.

        Returns:
            torch.Tensor: Calculated mean value of loss.

        """
        theta = dict(self._policy.named_parameters())
        old_theta = dict(self._old_policy.named_parameters())

        losses = []
        for task_samples, task_params in zip(all_samples, all_params):
            for i in range(self._num_grad_updates):
                require_grad = i < self._num_grad_updates - 1 or set_grad
                self._adapt(task_samples[i], set_grad=require_grad)

            update_module_params(self._old_policy, task_params)
            with torch.set_grad_enabled(set_grad):
                # pylint: disable=protected-access
                last_update = task_samples[-1]
                loss = self._inner_algo._compute_loss(*last_update[1:])
            losses.append(loss)

            update_module_params(self._policy, theta)
            update_module_params(self._old_policy, old_theta)

        return torch.stack(losses).mean()

    def _compute_kl_constraint(self, all_samples, all_params, set_grad=True):
        """Compute KL divergence.

        For each task, compute the KL divergence between the old policy
        distribution and current policy distribution.

        Args:
            all_samples (list[list[_MAMLEpisodeBatch]]): Two
                dimensional list of _MAMLEpisodeBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            all_params (list[dict]): A list of named parameter dictionaries.
                Each dictionary contains key value pair of names (str) and
                parameters (torch.Tensor).
            set_grad (bool): Whether to enable gradient calculation or not.

        Returns:
            torch.Tensor: Calculated mean value of KL divergence.

        """
        theta = dict(self._policy.named_parameters())
        old_theta = dict(self._old_policy.named_parameters())

        kls = []
        for task_samples, task_params in zip(all_samples, all_params):
            for i in range(self._num_grad_updates):
                require_grad = i < self._num_grad_updates - 1 or set_grad
                self._adapt(task_samples[i], set_grad=require_grad)

            update_module_params(self._old_policy, task_params)
            with torch.set_grad_enabled(set_grad):
                # pylint: disable=protected-access
                kl = self._inner_algo._compute_kl_constraint(
                    task_samples[-1].observations)
            kls.append(kl)

            update_module_params(self._policy, theta)
            update_module_params(self._old_policy, old_theta)

        return torch.stack(kls).mean()

    def _compute_policy_entropy(self, task_samples):
        """Compute policy entropy.

        Args:
            task_samples (list[_MAMLEpisodeBatch]): Samples data for
                one task.

        Returns:
            torch.Tensor: Computed entropy value.

        """
        obs = torch.cat([samples.observations for samples in task_samples])
        # pylint: disable=protected-access
        entropies = self._inner_algo._compute_policy_entropy(obs)
        return entropies.mean()

    def _compute_policy_stddev(self, task_samples):
        """Compute policy stddev.

        Args:
            task_samples (list[_MAMLEpisodeBatch]): Samples data for
                one task.

        Returns:
            torch.Tensor: Computed entropy value.

        """
        obs = torch.stack([samples.observations for samples in task_samples])
        # pylint: disable=protected-access
        stddev = self._inner_algo._compute_policy_stddev(obs)
        return stddev.mean()

    @property
    def policy(self):
        """Current policy of the inner algorithm.

        Returns:
            garage.torch.policies.Policy: Current policy of the inner
                algorithm.

        """
        return self._policy

    @property
    def _old_policy(self):
        """Old policy of the inner algorithm.

        Returns:
            garage.torch.policies.Policy: Old policy of the inner algorithm.

        """
        # pylint: disable=protected-access
        return self._inner_algo._old_policy

    def _process_samples(self, episodes):
        """Process sample data based on the collected paths.

        Args:
            episodes (EpisodeBatch): Collected batch of episodes.

        Returns:
            _MAMLEpisodeBatch: Processed samples data.

        """
        paths = episodes.to_list()
        if self._constraint:
            all_constraint_violations = []
        for path in paths:
            path['returns'] = discount_cumsum(
                path['rewards'], self._inner_algo.discount).copy()
            if self._constraint:
                const_violations = path["env_infos"]["constraint"].astype("float")
                all_constraint_violations.append(const_violations)
                path['penalties'] = discount_cumsum(
                    const_violations, self._inner_algo.discount).copy()
        self._value_function.fit(paths)

        obs = torch.Tensor(episodes.padded_observations)
        actions = torch.Tensor(episodes.padded_actions)
        rewards = torch.Tensor(episodes.padded_rewards)
        valids = torch.Tensor(episodes.lengths).int()
        baselines = torch.Tensor(
            [self._value_function.predict(path) for path in paths])
        if self._constraint:

            const_violations = torch.Tensor(np.stack(all_constraint_violations))

            if self._train_constraint:
                self._value_function_const.fit(paths, y_label="penalties")
                baselines_const = torch.clamp(torch.Tensor(
                    [self._value_function_const.predict(path) for path in paths]
                ), min=0)
            else:
                baselines_const = None
            return _MAMLEpisodeBatchConstrained(paths, obs, actions, rewards, valids,
                                     baselines, const_violations, baselines_const)
        return _MAMLEpisodeBatch(paths, obs, actions, rewards, valids,
                                 baselines)

    def _log_performance(
        self,
        itr,
        all_samples,
        loss_before,
        loss_after,
        kl_before,
        kl,
        policy_entropy,
        stddev,
        explained_variance,
        ev_const=None,
        lagrangian=None,
    ):
        """Evaluate performance of this batch.

        Args:
            itr (int): Iteration number.
            all_samples (list[list[_MAMLEpisodeBatch]]): Two
                dimensional list of _MAMLEpisodeBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            loss_before (float): Loss before optimization step.
            loss_after (float): Loss after optimization step.
            kl_before (float): KL divergence before optimization step.
            kl (float): KL divergence after optimization step.
            policy_entropy (float): Policy entropy.
            stddev (float): Policy stddev
            explained_variance (float): explained variance of baseline
            const_violations (float): proportion of constraint-violations.
            ev_const (float): explained variance of constraint-baseline
            lagrangian (float): current lambda

        Returns:
            float: The average return in last epoch cycle.

        """
        tabular.record('Iteration', itr)

        name_map = None
        if hasattr(self._env, 'all_task_names'):
            names = self._env.all_task_names
            name_map = dict(zip(names, names))

        rtns = log_multitask_performance(
            itr,
            EpisodeBatch.from_list(
                env_spec=self._env.spec,
                paths=[
                    path for task_paths in all_samples
                    for path in task_paths[self._num_grad_updates].paths
                ]),
            discount=self._inner_algo.discount,
            name_map=name_map,
            w_b=self._w_and_b,
            super_prefix="post_adaptation/"
        )
        log_multitask_performance(
            itr,
            EpisodeBatch.from_list(
                env_spec=self._env.spec,
                paths=[
                    path for task_paths in all_samples
                    for path in task_paths[self._num_grad_updates-1].paths
                ]),
            discount=self._inner_algo.discount,
            name_map=name_map,
            w_b=self._w_and_b,
            super_prefix="pre_adaptation/"
        )


        with tabular.prefix(self._policy.name + '/'):
            tabular.record('LossBefore', loss_before)
            tabular.record('LossAfter', loss_after)
            tabular.record('dLoss', loss_before - loss_after)
            tabular.record('KLBefore', kl_before)
            tabular.record('KLAfter', kl)
            tabular.record('Entropy', policy_entropy)
            tabular.record('StandardDeviation', stddev)
            if lagrangian is not None:
                tabular.record('lagrangian', lagrangian)
        tabular.record(f'{self._value_function.name}/ExplainedVariance',
                       explained_variance)
        if ev_const is not None:
            tabular.record(f'{self._value_function_const.name}/ExplainedVariance',
                       ev_const)

        return np.mean(rtns)

    def get_exploration_policy(self):
        """Return a policy used before adaptation to a specific task.

        Each time it is retrieved, this policy should only be evaluated in one
        task.

        Returns:
            Policy: The policy used to obtain samples that are later used for
                meta-RL adaptation.

        """
        return copy.deepcopy(self._policy)

    def adapt_policy(self, exploration_policy, exploration_episodes):
        """Adapt the policy by one gradient steps for a task.

        Args:
            exploration_policy (Policy): A policy which was returned from
                get_exploration_policy(), and which generated
                exploration_episodes by interacting with an environment.
                The caller may not use this object after passing it into this
                method.
            exploration_episodes (EpisodeBatch): Episodes with which to adapt,
                generated by exploration_policy exploring the environment.

        Returns:
            Policy: A policy adapted to the task represented by the
                exploration_episodes.

        """
        old_policy, self._policy = self._policy, exploration_policy
        self._inner_algo.policy = exploration_policy
        self._inner_optimizer.module = exploration_policy

        batch_samples = self._process_samples(exploration_episodes)

        self._adapt(batch_samples, set_grad=False)

        self._policy = old_policy
        self._inner_algo.policy = self._inner_optimizer.module = old_policy
        return exploration_policy


class _MAMLEpisodeBatch(
        collections.namedtuple('_MAMLEpisodeBatch', [
            'paths', 'observations', 'actions', 'rewards', 'valids',
            'baselines'
        ])):
    r"""A tuple representing a batch of whole episodes in MAML.

    A :class:`_MAMLEpisodeBatch` represents a batch of whole episodes
    produced from one environment.
    +-----------------------+-------------------------------------------------+
    | Symbol                | Description                                     |
    +=======================+=================================================+
    | :math:`N`             | Episode batch dimension                         |
    +-----------------------+-------------------------------------------------+
    | :math:`T`             | Maximum length of an episode                    |
    +-----------------------+-------------------------------------------------+
    | :math:`S^*`           | Single-step shape of a time-series tensor       |
    +-----------------------+-------------------------------------------------+

    Attributes:
        paths (list[dict[str, np.ndarray or dict[str, np.ndarray]]]):
            Nonflatten original paths from sampler.
        observations (torch.Tensor): A torch tensor of shape
            :math:`(N \bullet T, O^*)` containing the (possibly
            multi-dimensional) observations for all time steps in this batch.
            These must conform to :obj:`env_spec.observation_space`.
        actions (torch.Tensor): A torch tensor of shape
            :math:`(N \bullet T, A^*)` containing the (possibly
            multi-dimensional) actions for all time steps in this batch. These
            must conform to :obj:`env_spec.action_space`.
        rewards (torch.Tensor): A torch tensor of shape
            :math:`(N \bullet T)` containing the rewards for all time
            steps in this batch.
        valids (numpy.ndarray): An integer numpy array of shape :math:`(N, )`
            containing the length of each episode in this batch. This may be
            used to reconstruct the individual episodes.
        baselines (numpy.ndarray): An numpy array of shape
            :math:`(N \bullet T, )` containing the value function estimation
            at all time steps in this batch.

    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """


class _MAMLEpisodeBatchConstrained(
        collections.namedtuple('_MAMLEpisodeBatch', [
            'paths', 'observations', 'actions', 'rewards', 'valids',
            'baselines', 'const_violations', 'baselines_const'
        ])):
    r"""A tuple representing a batch of whole episodes in MAML.

    A :class:`_MAMLEpisodeBatch` represents a batch of whole episodes
    produced from one environment.
    +-----------------------+-------------------------------------------------+
    | Symbol                | Description                                     |
    +=======================+=================================================+
    | :math:`N`             | Episode batch dimension                         |
    +-----------------------+-------------------------------------------------+
    | :math:`T`             | Maximum length of an episode                    |
    +-----------------------+-------------------------------------------------+
    | :math:`S^*`           | Single-step shape of a time-series tensor       |
    +-----------------------+-------------------------------------------------+

    Attributes:
        paths (list[dict[str, np.ndarray or dict[str, np.ndarray]]]):
            Nonflatten original paths from sampler.
        observations (torch.Tensor): A torch tensor of shape
            :math:`(N \bullet T, O^*)` containing the (possibly
            multi-dimensional) observations for all time steps in this batch.
            These must conform to :obj:`env_spec.observation_space`.
        actions (torch.Tensor): A torch tensor of shape
            :math:`(N \bullet T, A^*)` containing the (possibly
            multi-dimensional) actions for all time steps in this batch. These
            must conform to :obj:`env_spec.action_space`.
        rewards (torch.Tensor): A torch tensor of shape
            :math:`(N \bullet T)` containing the rewards for all time
            steps in this batch.
        valids (numpy.ndarray): An integer numpy array of shape :math:`(N, )`
            containing the length of each episode in this batch. This may be
            used to reconstruct the individual episodes.
        baselines (numpy.ndarray): An numpy array of shape
            :math:`(N \bullet T, )` containing the value function estimation
            at all time steps in this batch.
        const_violations (torch.Tensor): A torch tensor of shape
            :math:`(N \bullet T)` containing the constraint violations
             for all time steps in this batch.
        baselines_const (numpy.ndarray): An numpy array of shape
            :math:`(N \bullet T, )` containing the value function
            estimation for penalties at all time steps in this batch.

    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """
