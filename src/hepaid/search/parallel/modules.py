import ray
from ray.util.actor_pool import ActorPool
import numpy as np

from typing import Callable, Dict, Any, List
import multiprocessing


@ray.remote
class Worker:
    def __init__(
        self,
        function: Callable[..., Dict[str, Any]],
        ):
        self.fn = function

    def sample(self, x):
        result = self.fn(x)
        return result



def run_x_with_pool(
    X: np.ndarray | list,
    n_workers: int,
    function: Callable[..., Dict[str, Any]],
    ):

    if isinstance(X, np.ndarray):
        X = [X[i] for i in range(len(X))]


    workers = [
        Worker.remote(function)
            for _ in
                range(n_workers)
        ]

    pool = ActorPool(workers)
    # results = pool.map_unordered(
    #     lambda w, x: w.sample.remote(x), X
    #     )

    # Use map_unordered to execute tasks in parallel
    results_iterator = pool.map_unordered(
        lambda w, x: w.sample.remote(x), X
    )

    # Collect results to enforce waiting for all computations
    results = list(results_iterator)

    return results

def run_x_with_mp_pool(
    X: np.ndarray | list,
    n_workers: int,
    function: Callable[..., Dict[str, Any]],
    ):

    if isinstance(X, np.ndarray):
        X = [X[i] for i in range(len(X))]

    # Create a pool of workers
    with multiprocessing.Pool(processes=n_workers) as pool:
        # Distribute the tasks among the workers and collect the results
        results = pool.map(function, X)

    # Return the collected results
    return results
