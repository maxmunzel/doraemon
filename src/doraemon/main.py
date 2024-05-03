import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple, List, Deque, Dict, Tuple
from scipy.optimize import minimize, NonlinearConstraint
from collections import deque
from copy import deepcopy
from scipy.stats import beta
from scipy.special import psi
from math import gamma


class MultivariateBetaDistribution:
    def __init__(
        self,
        alphas: List[float],
        low: List[float],
        high: List[float],
        param_bound: List[float],
        names: List[str],
        seed=42,
    ):
        self.low = np.array(low)
        self.high = np.array(high)
        self.alphas = np.array(alphas)
        self.betas = self.alphas  # Since alpha == beta
        self.param_bound = param_bound
        self.names = names
        assert len(alphas) == len(names) == len(low) == len(high)
        self.random = np.random.default_rng(seed)

    def with_params(self, alphas: List[float]):
        assert len(alphas) == len(self.alphas)
        res = deepcopy(self)
        res.set_params(alphas)
        return res

    def sample(self) -> np.ndarray:
        low = self.low
        high = self.high

        samples = np.array(
            [self.random.beta(a, b) for a, b in zip(self.alphas, self.betas)]
        )

        return low + (high - low) * samples

    def sample_dict(self) -> Tuple[np.ndarray, Dict[str, float]]:
        sample = self.sample()
        return sample, {k: v for k, v in zip(self.names, sample)}

    def likelihood(self, sample: List[float]) -> float:
        high = self.high
        low = self.low
        mask = 1 * (low == high)  # ignore fields with no variance
        sample = (sample - low + mask) / (high - low + mask)
        eps = 0.001
        sample = np.clip(sample, eps, 1 - eps)
        likelihoods = [
            beta.pdf(x, a, b) for x, a, b in zip(sample, self.alphas, self.betas)
        ]
        return np.prod(likelihoods)

    def entropy(self) -> float:
        entropies = [beta.entropy(a, b) for a, b in zip(self.alphas, self.betas)]
        return np.sum(entropies)

    def kl_dist(self, other) -> float:
        # formula from https://math.stackexchange.com/questions/257821/kullback-liebler-divergence#comment564291_257821,
        # verified against numerical integration (at least for our alpha == beta case)

        res = 0
        for f, g in zip(self.get_params(), other.get_params()):
            try:
                res += np.log(
                    (gamma(f + f) * gamma(g) ** 2) / (gamma(g + g) * gamma(f) ** 2)
                )
                res += 2 * (f - g) * (psi(f) - psi(f + f))
            except ArithmeticError:
                res += 1

        return res

    def get_params(self) -> np.ndarray:
        alphas = np.nan_to_num(self.alphas, nan=np.inf)
        alphas = np.clip(alphas, np.ones_like(alphas), self.param_bound)
        return alphas.copy()

    def set_params(self, alphas: List[float]):
        alphas = np.array(alphas).copy()
        alphas = np.nan_to_num(alphas, nan=np.inf)
        alphas = np.clip(alphas, np.ones_like(alphas), self.param_bound)
        self.alphas = alphas
        self.betas = alphas  # Update betas as well since alpha == beta


class Trajectory(NamedTuple):
    params: List[float]
    successful: bool


class Doraemon:
    def __init__(
        self,
        dist: MultivariateBetaDistribution,
        k: int,
        kl_bound: float,
        target_success_rate: float,
    ):
        self.n_traj = 0
        self.k = k
        self.kl_bound = kl_bound
        self.dist = dist
        self.target_success_rate = target_success_rate
        self.buffer: Deque[Trajectory] = deque(maxlen=k)

    def add_trajectory(self, params: List[float], successful: bool):
        self.n_traj += 1
        self.buffer.append(Trajectory(params, successful))

    def param_dict(self) -> Dict[str, float]:
        res = {
            f"doramon_{k}": v for k, v in zip(self.dist.names, self.dist.get_params())
        }
        res["doramon_naive_success_rate"] = self._naive_success_rate()
        res["doramon_estimated_success"] = self._estimate_success(self.dist)
        res["doramon_entropy"] = self.dist.entropy()
        return res

    def _estimate_success(self, cand: MultivariateBetaDistribution) -> float:
        if not self.buffer:
            return 0
        # Importance sampling according to equation (5)
        result = 0
        for t in self.buffer:
            if not t.successful:
                continue
            result += cand.likelihood(t.params) / self.dist.likelihood(t.params)
        return result / len(self.buffer)

    def _find_feasable_dist(self) -> MultivariateBetaDistribution:
        # Equation (6)
        kl_constraint = NonlinearConstraint(
            fun=lambda x: self.dist.kl_dist(self.dist.with_params(x)),
            lb=0,
            ub=self.kl_bound,
        )

        param_bounds = [(1, bound) for bound in self.dist.param_bound]

        res = minimize(
            fun=lambda x: -self._estimate_success(self.dist.with_params(x)),
            x0=self.dist.get_params(),
            constraints=[kl_constraint],
            bounds=param_bounds,
        )
        if (
            res.success
            and np.all(np.isfinite(res.x))
            and np.all(1 <= res.x)
            and np.all(res.x <= self.dist.param_bound)
        ):
            return self.dist.with_params(res.x)
        else:
            return self.dist

    def _naive_success_rate(self) -> float:
        if not self.buffer:
            return 0
        return np.mean([t.successful for t in self.buffer])

    def _find_max_entropy_dist(self) -> MultivariateBetaDistribution:
        # Equation (5)
        kl_constraint = NonlinearConstraint(
            fun=lambda x: self.dist.kl_dist(self.dist.with_params(x)),
            lb=0,
            ub=self.kl_bound,
        )
        success_constraint = NonlinearConstraint(
            fun=lambda x: self._estimate_success(self.dist.with_params(x)),
            lb=self.target_success_rate,
            ub=1,
        )

        param_bounds = [(1, bound) for bound in self.dist.param_bound]

        res = minimize(
            fun=lambda x: -self.dist.with_params(x).entropy(),
            x0=self.dist.get_params(),
            constraints=[success_constraint, kl_constraint],
            bounds=param_bounds,
            method="trust-constr",
        )
        if (
            res.success
            and np.all(np.isfinite(res.x))
            and np.all(1 <= res.x)
            and np.all(res.x <= self.dist.param_bound)
        ):
            return self.dist.with_params(res.x)
        else:
            return self.dist

    def update_dist(self):
        # Implement "Dynamics distribution update" of Algorithm 1 lines 7-14,
        # updating self.dist in-place
        if len(self.buffer) < self.k:
            return

        if self.n_traj % self.k != 0:
            return

        if self._estimate_success(self.dist) < self.target_success_rate:
            cand = self._find_feasable_dist()
            if self._estimate_success(cand) < self.target_success_rate:
                self.dist = cand
                return
            self.dist = cand
        self.dist = self._find_max_entropy_dist()


if __name__ == "__main__":
    dist = MultivariateBetaDistribution([0.5, 2], ["x", "y"])
    dist = MultivariateBetaDistribution(
        [10, 10], names=["x", "y"], low=[-5, -5], high=[5, 5]
    )

    d = Doraemon(dist, 100, 0.1, 0.9)
    params = []
    samples = []
    N = 10000
    for i in range(N):
        sample = d.dist.sample()
        success = all(np.abs(sample) < [5, 1])
        d.add_trajectory(sample, success)
        samples.append(sample.copy())
        params.append(d.dist.get_params())
        d.update_dist()
    params = np.array(params)
    samples = np.array(samples)

    fig, ax = plt.subplots()
    # ax.scatter(np.arange(N), params[:, 0], label="x", alpha=0.1)
    # ax.scatter(np.arange(N), params[:, 1], label="y", alpha=0.1)
    ax.scatter(np.arange(N), samples[:, 0], label="x", alpha=0.1)
    ax.scatter(np.arange(N), samples[:, 1], label="y", alpha=0.1)
    # ax.plot(success, label="success")
    fig.legend()
    plt.show()
