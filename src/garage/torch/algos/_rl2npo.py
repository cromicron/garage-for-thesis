"""Natural Policy Gradient Optimization."""
from dowel import logger, tabular
import numpy as np
import torch
from garage.np import explained_variance_1d, pad_batch_array
from garage.torch.algos import NPO
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

class RL2NPO(NPO):
    """Natural Policy Gradient Optimization.

    This is specific for RL^2
    (https://arxiv.org/pdf/1611.02779.pdf).

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        baseline_const: If constraints, the basline for const
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
        pg_loss (str): A string from: 'vanilla', 'surrogate',
            'surrogate_clip'. The type of loss functions to use.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies, as in PPO.
        max_kl_step (float): The maximum KL divergence between old and new
            policies, as in TRPO.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in garage.tf.optimizers.
        optimizer_args (dict): The arguments of the optimizer.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        use_neg_logli_entropy (bool): Whether to estimate the entropy as the
            negative log likelihood of the action.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        fit_baseline (str): Either 'before' or 'after'. See above docstring for
            a more detail explanation. Currently it only supports 'before'.
        name (str): The name of the algorithm.

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
