"""Evaluator which tests Meta-RL algorithms on test environments."""

from dowel import logger, tabular

from garage import EpisodeBatch, log_multitask_performance
from garage.experiment.deterministic import get_seed
from garage.sampler import DefaultWorker, LocalSampler, WorkerFactory
import numpy as np


class MetaEvaluator:
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
                 pre_post_prefixes=None,
                 ):
        self._sampler = sampler
        self._test_task_sampler = task_sampler
        self._worker_class = worker_class
        if worker_args is None:
            self._worker_args = {}
        else:
            self._worker_args = worker_args
        if n_test_tasks is None:
            n_test_tasks = task_sampler.n_tasks
        self._n_test_tasks = n_test_tasks
        self._n_test_episodes = n_test_episodes
        self._n_exploration_eps = n_exploration_eps
        self._eval_itr = start_eval_itr
        self._prefix = prefix
        self._test_task_names = test_task_names
        self._test_sampler = None
        self._max_episode_length = None
        self._w_and_b = w_and_b
        self._render_examples = render_examples
        self._pre_post_prefixes = pre_post_prefixes
        if pre_post_prefixes is not None:
            assert len(pre_post_prefixes) == 2, "specify two prefixes"
    def evaluate(self, algo, test_episodes_per_task=None, itr_multiplier=1, epoch=None):
        """Evaluate the Meta-RL algorithm on the test tasks.

        Args:
            algo (MetaRLAlgorithm): The algorithm to evaluate.
            test_episodes_per_task (int or None): Number of episodes per task.
            itr_multiplier: necessary for w and b logging, if not every
            epoch is logged.
            epoch: optionally give epoch directly.

        """
        if test_episodes_per_task is None:
            test_episodes_per_task = self._n_test_episodes
        adapted_episodes = []
        logger.log(f'Sampling for adapation and meta-testing ...')
        env_updates = self._test_task_sampler.sample(self._n_test_tasks)
        exploration_episodes = []
        for env_up in env_updates:
            policy = algo.get_exploration_policy()
            print("collecting episodes to adapt")
            steps = self._n_exploration_eps * algo.max_episode_length
            eps = self._sampler.obtain_samples(self._eval_itr, steps, policy,
                                                  env_up)

            if self._pre_post_prefixes is not None:
                exploration_episodes.append(eps)
            adapted_policy = algo.adapt_policy(policy, eps)
            print("solving env with adapted policy")
            adapted_eps = self._sampler.obtain_samples(
                self._eval_itr,
                test_episodes_per_task * algo.max_episode_length,
                adapted_policy)
            adapted_episodes.append(adapted_eps)
        logger.log('Finished meta-testing...')

        if self._test_task_names is not None:
            name_map = dict(enumerate(self._test_task_names))
        else:
            name_map = None

        with tabular.prefix(self._prefix + '/' if self._prefix else ''):
            if self._pre_post_prefixes is not None:
                suffix_pre = self._pre_post_prefixes[0]
                suffix_post = self._pre_post_prefixes[1]
                epoch_log = epoch if epoch is not None else self._eval_itr * itr_multiplier
                log_multitask_performance(
                    epoch_log,
                    EpisodeBatch.concatenate(*exploration_episodes),
                    getattr(algo, 'discount', 1.0),
                    name_map=name_map,
                    w_b=self._w_and_b,
                    super_prefix=suffix_pre,
                )
            log_multitask_performance(
                epoch_log,
                EpisodeBatch.concatenate(*adapted_episodes),
                getattr(algo, 'discount', 1.0),
                name_map=name_map,
                w_b=self._w_and_b,
                super_prefix=suffix_post,
            )
        self._eval_itr += 1
        success = []
        const_violations = []
        for ep in adapted_episodes:
            success_rate = np.any(ep.padded_env_infos["success"] == 1,
                                  axis=1).mean()
            const_violation = ep.env_infos["constraint"].mean()
            success.append(success_rate)
            const_violations.append(const_violation)
        success_rate_total = np.array(success).mean()
        const_violation_total = np.array(const_violations).mean()
        return {
            "success_rate": success_rate_total,
            "constraint_violations": const_violation_total
        }
