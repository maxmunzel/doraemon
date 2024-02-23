import numpy as np
import scipy
import matplotlib.pyplot as plt
from typing import NamedTuple, List, Deque, Dict, Tuple
from scipy.optimize import minimize
from collections import deque
from copy import deepcopy
from scipy.stats import beta, norm


class MultivariateGaussianDistribution:
    def __init__(self, stds: List[float], names: List[str], means=None, seed=42):
        self.stds = np.array(stds)
        self.names = names
        assert len(stds) == len(names)
        if means is not None:
            self.means = np.array(means)
        else:
            self.means = np.zeros_like(stds)
        self.random = np.random.default_rng(seed)

    def with_params(self, stds: List[float], seed=42):
        stds = np.array(stds)
        assert stds.shape == self.stds.shape
        res = deepcopy(self)
        res.set_params(stds)
        return res

    def sample(self):
        samples = self.random.normal(self.means, self.stds)
        return samples

    def sample_dict(self) -> Tuple[np.ndarray, Dict[str, float]]:
        sample = self.sample()
        return sample, {k: v for k, v in zip(self.names, sample)}

    def likelihood(self, sample):
        likelihoods = norm.pdf(sample, self.means, self.stds)
        return np.prod(likelihoods)

    def entropy(self):
        entropies = norm.entropy(self.means, self.stds)
        return np.sum(entropies)

    def get_params(self):
        return self.stds.copy()

    def set_params(self, stds):
        self.stds = stds.copy()

    def kl_dist(self, other):
        k = len(self.means)
        Sigma0 = np.diag(self.stds**2)
        Sigma1 = np.diag(other.stds**2)

        Sigma1_Inv = np.linalg.inv(Sigma1)

        mu0 = self.means
        mu1 = other.means

        det = np.linalg.det
        tr = np.trace
        ln = np.log

        kl_div = (
            tr(Sigma1_Inv @ Sigma0)
            - k
            + (mu1 - mu0).T @ Sigma1_Inv @ (mu1 - mu0)
            + ln(det(Sigma1) / det(Sigma0))
        )
        kl_div /= 2

        return kl_div


class MultivariateBetaDistribution:
    def __init__(
        self,
        alphas: List[float],
        low: List[float],
        high: List[float],
        names: List[str],
        seed=42,
    ):
        self.low = np.array(low)
        self.high = np.array(high)
        self.alphas = np.array(alphas)
        self.betas = self.alphas  # Since alpha == beta
        self.names = names
        assert len(alphas) == len(names) == len(low) == len(high)
        self.random = np.random.default_rng(seed)

    def with_params(self, alphas: List[float], seed=42):
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
        sample = (sample - low) / (high - low)
        likelihoods = [
            beta.pdf(x, a, b) for x, a, b in zip(sample, self.alphas, self.betas)
        ]
        return np.prod(likelihoods)

    def entropy(self) -> float:
        entropies = [beta.entropy(a, b) for a, b in zip(self.alphas, self.betas)]
        return np.sum(entropies)

    def kl_dist(self, other) -> float:
        # TODO: find an actual way to compute this
        return np.linalg.norm(self.get_params() - other.get_params())

    def get_params(self) -> np.ndarray:
        return self.alphas.copy()

    def set_params(self, alphas: List[float]):
        self.alphas = np.array(alphas)
        self.betas = np.array(alphas)  # Update betas as well since alpha == beta


class Trajectory(NamedTuple):
    params: List[float]
    successful: bool


class Doraemon:
    def __init__(
        self,
        dist: MultivariateGaussianDistribution,
        k: int,
        kl_bound: float,
        target_success_rate: float,
    ):
        self.k = k
        self.kl_bound = kl_bound
        self.dist = dist
        self.target_success_rate = target_success_rate
        self.buffer: Deque[Trajectory] = deque(maxlen=k)

    def add_trajectory(self, params: List[float], successful: bool):
        self.buffer.append(Trajectory(params, successful))

    def _estimate_success(self, cand: MultivariateGaussianDistribution) -> float:
        # Importance sampling according to equation (5)
        result = 0
        for t in self.buffer:
            if not t.successful:
                continue
            result += cand.likelihood(t.params) / self.dist.likelihood(t.params)
        return result / len(self.buffer)

    def _find_feasable_dist(self) -> MultivariateGaussianDistribution:
        # Equation (6)
        constraint = scipy.optimize.NonlinearConstraint(
            fun=lambda x: self.dist.kl_dist(self.dist.with_params(x)),
            lb=0,
            ub=self.kl_bound,
        )
        res = minimize(
            fun=lambda x: -self._estimate_success(self.dist.with_params(x)),
            x0=self.dist.get_params(),
            constraints=[constraint],
        )
        return self.dist.with_params(res.x)

    def _find_max_entropy_dist(self) -> MultivariateGaussianDistribution:
        # Equation (5)
        kl_constraint = scipy.optimize.NonlinearConstraint(
            fun=lambda x: self.dist.kl_dist(self.dist.with_params(x)),
            lb=0,
            ub=self.kl_bound,
        )
        success_constraint = scipy.optimize.NonlinearConstraint(
            fun=lambda x: self._estimate_success(self.dist.with_params(x)),
            lb=self.target_success_rate,
            ub=1,
        )
        res = minimize(
            fun=lambda x: -self.dist.with_params(x).entropy(),
            x0=self.dist.get_params(),
            constraints=[success_constraint, kl_constraint],
        )
        return self.dist.with_params(res.x)

    def update_dist(self):
        # Implement "Dynamics distribution update" of Algorithm 1 lines 7-14,
        # updating self.dist in-place
        if len(self.buffer) < self.k:
            return

        if np.mean([t.successful for t in self.buffer]) < self.target_success_rate:
            cand = self._find_feasable_dist()
            if self._estimate_success(cand) < self.target_success_rate:
                self.dist = cand
                return
            self.dist = cand
        self.dist = self._find_max_entropy_dist()


if __name__ == "__main__":
    dist = MultivariateGaussianDistribution([0.5, 2], ["x", "y"])
    dist = MultivariateBetaDistribution(
        [10, 10], names=["x", "y"], low=[-5, -5], high=[5, 5]
    )

    d = Doraemon(dist, 100, 0.1, 0.9)
    params = []
    samples = []
    N = 1000
    for i in range(N):
        sample = d.dist.sample()
        success = all(np.abs(sample) < [5, 1])
        d.add_trajectory(sample, success)
        samples.append(sample.copy())
        params.append(d.dist.get_params())
        # successes.append(d._estimate_success(d.dist))
        if i > 100 and not i % 100:
            d.update_dist()
    params = np.array(params)
    samples = np.array(samples)

    fig, ax = plt.subplots()
    ax.scatter(np.arange(N), params[:, 0], label="x", alpha=0.1)
    ax.scatter(np.arange(N), params[:, 1], label="y", alpha=0.1)
    # ax.scatter(np.arange(N), samples[:, 0], label="x", alpha=0.1)
    # ax.scatter(np.arange(N), samples[:, 1], label="y", alpha=0.1)
    # ax.plot(success, label="success")
    fig.legend()
    plt.show()
