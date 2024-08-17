"""Evaluator which tests Meta-RL algorithms on test environments."""

from dowel import logger, tabular

from garage import EpisodeBatch, log_multitask_performance
from garage.experiment.deterministic import get_seed
from garage.sampler import DefaultWorker, LocalSampler, WorkerFactory


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
                 test_task_sampler,
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
        self._test_task_sampler = test_task_sampler
        self._worker_class = worker_class
        if worker_args is None:
            self._worker_args = {}
        else:
            self._worker_args = worker_args
        if n_test_tasks is None:
            n_test_tasks = test_task_sampler.n_tasks
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
    def evaluate(self, algo, test_episodes_per_task=None, itr_multiplier=1):
        """Evaluate the Meta-RL algorithm on the test tasks.

        Args:
            algo (MetaRLAlgorithm): The algorithm to evaluate.
            test_episodes_per_task (int or None): Number of episodes per task.

        """
        if test_episodes_per_task is None:
            test_episodes_per_task = self._n_test_episodes
        adapted_episodes = []
        logger.log(f'Sampling for adapation and meta-testing ...')
        env_updates = self._test_task_sampler.sample(self._n_test_tasks)
        if self._test_sampler is None:
            env = env_updates[0]()
            self._max_episode_length = env.spec.max_episode_length
            self._test_sampler = LocalSampler.from_worker_factory(
                WorkerFactory(seed=get_seed(),
                              max_episode_length=self._max_episode_length,
                              n_workers=1,
                              worker_class=self._worker_class,
                              worker_args=self._worker_args),
                agents=algo.get_exploration_policy(),
                envs=env)
        exploration_episodes = []
        for env_up in env_updates:
            policy = algo.get_exploration_policy()
            print("collecting episodes to adapt")
            eps = EpisodeBatch.concatenate(*[
                self._test_sampler.obtain_samples(self._eval_itr, 1, policy,
                                                  env_up)
                for _ in range(self._n_exploration_eps)
            ])
            if self._pre_post_prefixes is not None:
                exploration_episodes.append(eps)
            adapted_policy = algo.adapt_policy(policy, eps)
            print("solving env with adapted policy")
            adapted_eps = self._test_sampler.obtain_samples(
                self._eval_itr,
                test_episodes_per_task * self._max_episode_length,
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
                log_multitask_performance(
                    self._eval_itr * itr_multiplier,
                    EpisodeBatch.concatenate(*exploration_episodes),
                    getattr(algo, 'discount', 1.0),
                    name_map=name_map,
                    w_b=self._w_and_b,
                    super_prefix=suffix_pre,
                )
            log_multitask_performance(
                self._eval_itr * itr_multiplier,
                EpisodeBatch.concatenate(*adapted_episodes),
                getattr(algo, 'discount', 1.0),
                name_map=name_map,
                w_b=self._w_and_b,
                super_prefix=suffix_post,
            )
        self._eval_itr += 1
