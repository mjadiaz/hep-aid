import ray
from ray.util.actor_pool import ActorPool
import numpy as np

from typing import Callable, Dict, Any, List

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
    results = pool.map_unordered(
        lambda w, x: w.sample.remote(x), X
        )

    return results


