"""Proximal Policy Optimization for RL2."""
from garage.torch.algos import RL2
from torch.optim import Adam
from garage.torch.optimizers import FirstOrderOptimizer


class RL2PPO(RL2):
    """Proximal Policy Optimization specific for RL^2.

    See https://arxiv.org/abs/1707.06347 for algorithm reference.

    Args:
        meta_batch_size (int): Meta batch size.
        task_sampler (TaskSampler): Task sampler.
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        sampler (garage.sampler.Sampler): Sampler.
        episodes_per_trial (int): Used to calculate the max episode length for
            the inner algorithm.
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
        meta_evaluator (garage.experiment.MetaEvaluator): Evaluator for meta-RL
            algorithms.
        n_epochs_per_eval (int): If meta_evaluator is passed, meta-evaluation
            will be performed every `n_epochs_per_eval` epochs.
        name (str): The name of the algorithm.

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
