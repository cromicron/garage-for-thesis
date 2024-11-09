"""
The standard evaluator doesn't allow for parallel processing for
RL2, because episodes are collected sequentially. This evaluator
runs test-environments in parallel and calculates returns the scores
of all episodes except for the first
"""

from dowel import logger, tabular
from typing import Tuple
from garage import EpisodeBatch, log_multitask_performance
from garage.experiment.deterministic import get_seed
from garage.sampler import DefaultWorker, LocalSampler, WorkerFactory


class RL2MetaEvaluator:
    """Evaluates Meta-RL algorithms on test environments.

    This class facilitates evaluation for Meta-RL algorithms by allowing sampling
    from test tasks and logging performance metrics. It splits episodes into
    exploration and adaptation phases to assess the algorithm's performance
    post-adaptation.

    Args:
        sampler (garage.sampler.Sampler): Sampler for generating episodes from tasks.
        task_sampler (TaskSampler): Sampler for test tasks. To validate the effectiveness
            of a meta-learning method, these tasks should differ from the training tasks.
        n_exploration_eps (int): Number of episodes to gather from the exploration
            policy before using the meta algorithm to produce an adapted policy.
        n_test_tasks (int or None): Number of test tasks to sample for evaluation.
            Tasks are sampled without replacement. Defaults to `sampler.n_tasks` if None.
        n_test_episodes (int): Number of episodes to use for each adapted policy.
            The adapted policy should reset (forget previous episodes) when `.reset()` is called.
        prefix (str): Prefix used for logging. Defaults to 'MetaTest', resulting in keys like
            'MetaTest/SuccessRate'. Set to 'MetaTrain' for training logs.
        test_task_names (list[str]): List of task names for testing, ordered consistently
            with the `task_id` in env_info if applicable.
        worker_class (type): Class type of worker used by the Sampler.
        worker_args (dict or None): Additional arguments to pass to the worker.
        start_eval_itr (int): Starting iteration for evaluation, typically set when resuming
            from a checkpoint.
        w_and_b (bool): Whether to log results to Weights & Biases.
        render_examples (bool): Whether to render example episodes during evaluation.

    Raises:
        ValueError: If task_sampler does not provide the required test tasks.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self,
                 *,
                 sampler,
                 task_sampler,
                 n_exploration_eps=10,
                 n_test_tasks=None,
                 n_test_episodes=1,
                 prefix='MetaTest',
                 test_task_names=None,
                 worker_class=DefaultWorker,
                 worker_args=None,
                 start_eval_itr = 0,
                 w_and_b=False,
                 render_examples=False,
                 ):
        self._sampler = sampler
        self._task_sampler = task_sampler
        self._worker_class = worker_class
        if worker_args is None:
            self._worker_args = {}
        else:
            self._worker_args = worker_args
        if n_test_tasks is None:
            n_test_tasks = sampler.n_tasks
        self._n_test_tasks = n_test_tasks
        self._n_test_episodes = n_test_episodes
        self._n_exploration_eps = n_exploration_eps
        self._eval_itr = start_eval_itr
        self._prefix = prefix
        self._test_task_names = test_task_names
        self._test_sampler = None
        self._max_episode_length = None
        self._w_and_b = w_and_b

    def _split_exploration_test(
        self, episodes: EpisodeBatch) -> Tuple[EpisodeBatch]:
        ep_list = episodes.to_list()
        exploration_eps = []
        adaptaion_eps = []
        for i in range(0, len(ep_list), self._n_test_episodes):
            exploration_eps.extend(ep_list[i: i+ self._n_exploration_eps])
            adaptaion_eps.extend(
                ep_list[i+self._n_exploration_eps: i+self._n_test_episodes])
        e = EpisodeBatch.from_list(episodes.env_spec, exploration_eps)
        a = EpisodeBatch.from_list(episodes.env_spec, adaptaion_eps)
        return e, a

    def evaluate(
        self,
        algo,
        test_episodes_per_task=None,
        itr_multiplier=1,
        epoch=None,
        run_in_eps=0,
    ):
        """Evaluate the Meta-RL algorithm on the test tasks.

        Args:
            algo (MetaRLAlgorithm): The algorithm to evaluate.
            test_episodes_per_task (int or None): Number of episodes per task.
                        itr_multiplier: necessary for w and b logging, if not every
            epoch is logged.
            epoch: optionally give epoch directly.
            run_in_eps: Useful if env is normalized, to get mean and std before
                        evaluating. Steps through envs this many episodes and
                        reset hidden afterward

        """
        epoch_log = epoch if epoch is not None else self._eval_itr * itr_multiplier
        if test_episodes_per_task is None:
            test_episodes_per_task = self._n_test_episodes
        logger.log(f'Sampling for adapation and meta-testing ...')
        env_updates = self._task_sampler.sample(self._n_test_tasks)

        policy = getattr(algo, 'exploration_policy', None)
        if policy is None:
            # This field should exist, since self.make_sampler would have
            # failed otherwise.
            policy = algo.policy
        agent_update = policy.get_param_values()
        steps = self._n_test_tasks*test_episodes_per_task*algo.max_episode_length
        if run_in_eps:
            logger.log("Stepping through envs, to get normalization values")
            self._sampler.obtain_samples(
                itr=self._eval_itr,
                num_samples= self._n_test_tasks*run_in_eps*algo.max_episode_length,
                agent_update=agent_update,
                env_update=env_updates)
        episodes = self._sampler.obtain_samples(
            itr=self._eval_itr,
            num_samples=steps,
            agent_update=agent_update,
            env_update=env_updates)
        exploration_episodes, adapted_episodes = self._split_exploration_test(episodes)
        if self._test_task_names is not None:
            name_map = dict(enumerate(self._test_task_names))
        else:
            name_map = None

        with tabular.prefix(self._prefix + "/" + "post_adaptation/" if self._prefix else ""):
            log_multitask_performance(
                epoch_log,
                adapted_episodes,
                getattr(algo, 'discount', 1.0),
                name_map=name_map,
                w_b=self._w_and_b
            )

        with tabular.prefix(self._prefix + "/" + "pre_adaptation/" if self._prefix else ""):
            log_multitask_performance(
                epoch_log,
                exploration_episodes,
                getattr(algo, 'discount', 1.0),
                name_map=name_map,
                w_b=self._w_and_b
            )
        logger.log('Finished meta-testing...')
        self._eval_itr += 1
