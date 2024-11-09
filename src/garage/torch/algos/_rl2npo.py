"""Natural Policy Gradient Optimization."""
from dowel import logger, tabular
import numpy as np
import torch
from garage.np import explained_variance_1d, pad_batch_array
from garage.torch.algos import NPO
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

class RL2NPO(NPO):
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
    """
    def optimize_policy(self, episodes, save_weights=False):
        """Optimize policy.

        Args:
            episodes (EpisodeBatch): Batch of episodes.

        """
        # Baseline predictions with all zeros for feeding inputs of Tensorflow
        baselines = np.zeros((len(episodes.lengths), max(episodes.lengths)))
        if not getattr(self.policy, 'is_actor_critic', False):
            # if actor critic, actor and critic share loss and opt
            returns = self._fit_baseline_with_data(episodes, baselines)

        with torch.no_grad():
            baselines = self._get_baseline_prediction(episodes)
        ev = explained_variance_1d(
            baselines.cpu().numpy().flatten(),
            returns.flatten(),
            episodes.valids.flatten()
        )
        policy_opt_input_values = self._policy_opt_input_values(
            episodes, baselines)
        if self._train_constraint:
            cum_penalties, penalties = self._fit_baseline_const(episodes)
            with torch.no_grad():
                baselines_const = self._get_baseline_prediction_const(episodes)

                ev_const = explained_variance_1d(
                    baselines_const.cpu().numpy().flatten(),
                    cum_penalties.flatten(),
                    episodes.valids.flatten()
                )

                tabular.record("const_baseline/ExplainedVariance",
                               ev_const)
            policy_opt_input_values.extend([penalties, baselines_const])
            if isinstance(self._baseline_const, dict):
                # ad individual lambdas and arrange them
                env_names_tensor = episodes.padded_env_infos[
                    "task_name"]  # n * 1000 tensor

                # Get the first entry from each row to extract unique environment names (n * 1)
                env_names_list = env_names_tensor[:,
                                 0]  # First column contains the env_name for each row

                # Create a list to store the corresponding Lagrangian multipliers
                lagrangian_values = []

                # Loop over the environment names and get the corresponding Lagrangian from the ParameterDict
                for env_name in env_names_list:
                    env_name_str = env_name.item()  # Convert to a Python string or key if necessary
                    lagrangian_value = self.policy.lagrangians[
                        env_name_str].detach()
                    lagrangian_values.append(lagrangian_value)

                # Stack the Lagrangian values into a tensor (n * 1)
                lagrangian_tensor = torch.stack(lagrangian_values).unsqueeze(1)
                policy_opt_input_values.append(lagrangian_tensor)
        inputs = [torch.tensor(i, dtype=torch.float64, device=device) for i in policy_opt_input_values]
        # Train policy network
        self.policy.train()
        logger.log('Computing loss before')
        self._old_policy.to(device=device, dtype=torch.float64)
        with torch.no_grad():
            loss_before, policy_kl_before = self._policy_loss(*inputs)
        logger.log('Optimizing')

        self._optimizer.optimize(inputs)
        torch.cuda.empty_cache()
        logger.log('Computing loss after')
        with torch.no_grad():
            loss_after, policy_kl = self._policy_loss(*inputs)

        tabular.record('{}/LossBefore'.format(self.policy.name), loss_before)
        tabular.record('{}/LossAfter'.format(self.policy.name), loss_after)
        tabular.record('{}/dLoss'.format(self.policy.name),
                       loss_before - loss_after)
        tabular.record('{}/KLBefore'.format(self.policy.name),
                       policy_kl_before)
        tabular.record('{}/KL'.format(self.policy.name), policy_kl)
        tabular.record(f'{self._baseline.name}/ExplainedVariance',
                       ev)
        self._old_policy.load_parameters(self.policy.get_parameters())
        self.policy.reset()
        self._old_policy.reset()

        if save_weights:
            self.policy.save_weights()
        self._optimizer.save_optimizer_state()
        if not getattr(self.policy, "is_actor_critic", False):
            self._baseline.save_weights()
            self._bl_optimizer.save_optimizer_state()
        if self._train_constraint and isinstance(
            self._baseline_const, torch.nn.Module
        ):
            self._baseline_const.save_weights()





    def _get_baseline_prediction(self, episodes):
        """Get baseline prediction.

        Args:
            episodes (EpisodeBatch): Batch of episodes.

        Returns:
            np.ndarray: Baseline prediction, with shape
                :math:`(N, max_episode_length * episode_per_task)`.

        """
        if getattr(self.policy,"is_actor_critic", False):
            obs_tensor = torch.stack([
                torch.tensor(np_array, dtype=torch.double, device=device) for np_array in episodes.observations_list
            ], dim=0)

            return self.policy.get_value(obs_tensor)

        else:
            obs = [
                self._baseline.forward(torch.tensor(
                    obs,
                    dtype=torch.float32,
                    device=device)).squeeze()
                for obs in episodes.observations_list
            ]
        return pad_batch_array(torch.cat(obs, dim=0), episodes.lengths,
                               self.max_episode_length)

    def _get_baseline_prediction_const(self, episodes):
        """Get baseline prediction.

        Args:
            episodes (EpisodeBatch): Batch of episodes.

        Returns:
            np.ndarray: Baseline prediction, with shape
                :math:`(N, max_episode_length * episode_per_task)`.

        """
        if isinstance(self._baseline_const, torch.nn.Module):
            obs = [
            self._baseline_const.predict(torch.tensor(
                obs,
                dtype=torch.float32,
                device=device)).squeeze()
            for obs in episodes.observations_list
            ]
        else:
            if isinstance(self._baseline_const, dict):
                obs = []
                for env_name in self._baseline_const.keys():
                    env_obs = [episodes.observations_list[i] for i in
                               range(len(episodes.observations_list)) if
                               episodes.padded_env_infos["task_name"][
                                   i, 0] == env_name]

                    obs.extend(
                        [torch.tensor(
                            np.maximum(self._baseline_const[env_name].predict(
                                {"observations": o}), 0),
                            dtype=torch.float32,
                            device=device) for o in env_obs
                        ])
            else:
                obs = [torch.tensor(
                    np.maximum(self._baseline_const.predict({"observations": obs}),0),
                    dtype=torch.float32,
                    device=device) for obs in episodes.observations_list
                       ]

        return pad_batch_array(torch.cat(obs, dim=0), episodes.lengths,
                               self.max_episode_length)
