"""Proximal Policy Optimization for RL2."""
from garage.torch.algos import RL2
from torch.optim import Adam
from garage.torch.optimizers import FirstOrderOptimizer


class RL2PPO(RL2):
    class RL2PPO(RL2):
        """Proximal Policy Optimization specific for RL^2.

        See https://arxiv.org/abs/1707.06347 for algorithm reference.

        Args:
            meta_batch_size (int): The number of tasks sampled per meta-iteration.
            task_sampler (MetaWorldTaskSampler): Training Task sampler, providing tasks for each episode.
            env_spec (EnvSpec): Environment specification, detailing observation and action spaces.
            policy (garage.torch.policies.StochasticPolicy): Policy, typically a stochastic neural network policy.
            baseline (garage.np.baselines.Baseline or garage.torch.value_functions.ValueFunction): The baseline model, used to fit the value function.
            sampler (RaySampler or LocalSampler): Sampler, a mechanism to gather trajectories from the environment.
            episodes_per_trial (int): Number of episodes for each trial in the inner algorithm.
            baseline_const (garage.np.baselines.Baseline or None, optional): Baseline model for constraint-specific evaluation, if applicable.
            scope (str, optional): Scope identifier for the algorithm, useful when running multiple algorithms.
            discount (float): Discount factor for future rewards, typically between 0 and 1.
            gae_lambda (float): Lambda used for generalized advantage estimation.
            center_adv (bool): Whether to rescale advantages to have mean 0 and standard deviation 1.
            positive_adv (bool): Whether to shift advantages to be always positive. Used with center_adv to standardize before shifting.
            fixed_horizon (bool): Whether to fix horizon, disregarding terminal states within an episode.
            lr_clip_range (float): The limit on the likelihood ratio between policies, as in PPO, to constrain updates.
            max_kl_step (float): The maximum KL divergence between old and new policies, used for trust region enforcement.
            optimizer_args_policy (dict): Arguments for configuring the policy optimizer.
            optimizer_args_baseline (dict): Arguments for configuring the baseline optimizer.
            policy_ent_coeff (float): Coefficient for policy entropy; setting it to zero removes entropy regularization.
            use_softplus_entropy (bool): Whether to estimate the softmax distribution of the entropy to prevent negative entropy.
            use_neg_logli_entropy (bool): Whether to estimate entropy as the negative log likelihood of the action.
            stop_entropy_gradient (bool): Whether to stop gradient flow through entropy.
            entropy_method (str): Type of entropy method to use: 'max', 'regularized', or 'no_entropy'.
            meta_evaluator (RL2MetaEvaluator, optional): Evaluator for meta-RL algorithms, assessing performance on held-out tasks.
            n_epochs_per_eval (int): If meta_evaluator is provided, evaluation frequency in terms of epochs.
            name (str): Name of the algorithm, defaults to 'PPO'.
            save_weights (bool): Whether to save model weights during training.
            w_and_b (bool): Whether to use Weights & Biases logging.
            render_every_i (int, optional): Frequency of rendering episodes for visualization.
            run_in_episodes (int): Number of episodes to run in each evaluation trial.
            constraint (bool): Whether to apply constraints on the optimization.
            train_constraint (bool, optional): Whether to apply a constraint during training.
            constraint_threshold (float, optional): Threshold for constraint application.
            lr_constraint (float, optional): Learning rate for constraint satisfaction optimization.
            valid_evaluator (RL2MetaEvaluator, optional): Evaluator for validation, if separate from the meta evaluator.
            state_dir (str, optional): Directory to save model states and checkpoints.

        """

    def __init__(self,
                 meta_batch_size,
                 task_sampler,
                 env_spec,
                 policy,
                 baseline,
                 sampler,
                 episodes_per_trial,
                 baseline_const=None,
                 optimizer_args_policy=None,
                 optimizer_args_baseline=None,
                 scope=None,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 meta_evaluator=None,
                 n_epochs_per_eval=10,
                 name='PPO',
                 save_weights=True,
                 w_and_b=False,
                 render_every_i=None,
                 run_in_episodes=0,
                 constraint=False,
                 train_constraint=None,
                 constraint_threshold=None,
                 lr_constraint=None,
                 valid_evaluator=None,
                 state_dir=None
                 ):
        if optimizer_args_policy is None:
            optimizer_args_policy = dict()
            optimizer_args_baseline = dict()
        super().__init__(meta_batch_size=meta_batch_size,
                         task_sampler=task_sampler,
                         env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         sampler=sampler,
                         episodes_per_trial=episodes_per_trial,
                         scope=scope,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         fixed_horizon=fixed_horizon,
                         pg_loss='surrogate_clip',
                         lr_clip_range=lr_clip_range,
                         max_kl_step=max_kl_step,
                         optimizer=FirstOrderOptimizer,
                         optimizer_args_policy=optimizer_args_policy,
                         optimizer_args_baseline=optimizer_args_baseline,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         use_neg_logli_entropy=use_neg_logli_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method,
                         meta_evaluator=meta_evaluator,
                         n_epochs_per_eval=n_epochs_per_eval,
                         name=name,
                         save_weights=save_weights,
                         w_and_b=w_and_b,
                         render_every_i=render_every_i,
                         run_in_episodes=run_in_episodes,
                         constraint=constraint,
                         baseline_const=baseline_const,
                         train_constraint=train_constraint,
                         constraint_threshold=constraint_threshold,
                         lr_constraint=lr_constraint,
                         valid_evaluator=valid_evaluator,
                         state_dir=state_dir,
                         )
