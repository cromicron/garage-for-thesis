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

    Args:
        test_task_sampler (TaskSampler): Sampler for test
            tasks. To demonstrate the effectiveness of a meta-learning method,
            these should be different from the training tasks.
        n_test_tasks (int or None): Number of test tasks to sample each time
            evaluation is performed. Note that tasks are sampled "without
            replacement". If None, is set to `test_task_sampler.n_tasks`.
        n_exploration_eps (int): Number of episodes to gather from the
            exploration policy before requesting the meta algorithm to produce
            an adapted policy.
        n_test_episodes (int): Number of episodes to use for each adapted
            policy. The adapted policy should forget previous episodes when
            `.reset()` is called.
        prefix (str): Prefix to use when logging. Defaults to MetaTest. For
            example, this results in logging the key 'MetaTest/SuccessRate'.
            If not set to `MetaTest`, it should probably be set to `MetaTrain`.
        test_task_names (list[str]): List of task names to test. Should be in
            an order consistent with the `task_id` env_info, if that is
            present.
        worker_class (type): Type of worker the Sampler should use.
        worker_args (dict or None): Additional arguments that should be
            passed to the worker.

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
        for i in range(0, len(ep_list), self._n_exploration_eps):
            exploration_eps.append(ep_list[i])
            adaptaion_eps.extend(ep_list[i+1: i+10])
        e = EpisodeBatch.from_list(episodes.env_spec, exploration_eps)
        a = EpisodeBatch.from_list(episodes.env_spec, adaptaion_eps)
        return e, a

    def evaluate(
        self,
        algo,
        test_episodes_per_task=None,
        itr_multiplier=1,
    ):
        """Evaluate the Meta-RL algorithm on the test tasks.

        Args:
            algo (MetaRLAlgorithm): The algorithm to evaluate.
            test_episodes_per_task (int or None): Number of episodes per task.

        """
        if test_episodes_per_task is None:
            test_episodes_per_task = self._n_test_episodes
        adapted_episodes = []
        logger.log(f'Sampling for adapation and meta-testing ...')
        env_updates = self._task_sampler.sample(self._n_test_tasks)

        policy = getattr(algo, 'exploration_policy', None)
        if policy is None:
            # This field should exist, since self.make_sampler would have
            # failed otherwise.
            policy = algo.policy
        agent_update = policy.get_param_values()
        steps = self._n_test_tasks*test_episodes_per_task*algo.max_episode_length
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

        with tabular.prefix(self._prefix + "/" + "post_adaptation" if self._prefix else ""):
            log_multitask_performance(
                self._eval_itr * itr_multiplier,
                adapted_episodes,
                getattr(algo, 'discount', 1.0),
                name_map=name_map,
                w_b=self._w_and_b
            )

        with tabular.prefix(self._prefix + "/" + "pre_adaptation/" if self._prefix else ""):
            log_multitask_performance(
                self._eval_itr * itr_multiplier,
                exploration_episodes,
                getattr(algo, 'discount', 1.0),
                name_map=name_map,
                w_b=self._w_and_b
            )
        logger.log('Finished meta-testing...')
        self._eval_itr += 1
