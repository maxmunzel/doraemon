import marimo

__generated_with = "0.1.76"
app = marimo.App()


@app.cell
def __():
    import numpy as np
    import numpy.random
    import scipy
    import matplotlib.pyplot as plt
    from typing import NamedTuple, Protocol, List, Deque
    from scipy.optimize import minimize, NonlinearConstraint
    from collections import deque
    from copy import deepcopy
    return (
        Deque,
        List,
        NamedTuple,
        NonlinearConstraint,
        Protocol,
        deepcopy,
        deque,
        minimize,
        np,
        numpy,
        plt,
        scipy,
    )


@app.cell
def __(Dict, List, deepcopy, np):
    from scipy.stats import norm


    class MultivariateGaussianDistribution:
        def __init__(
            self, stds: List[float], names: List[str], means=None, seed=42
        ):
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

        def sample_dict(self) -> Dict[str, float]:
            return {k: v for k, v in zip(self.names, self.sample())}

        def likelihood(self, sample):
            likelihoods = norm.pdf(sample, self.means, self.stds)
            return np.prod(likelihoods)

        def entropy(self):
            entropies = norm.entropy(self.means, self.stds)
            return np.sum(entropies)

        def kl_dist(self, other):
            sigma0_squared = self.stds**2
            sigma1_squared = other.stds**2
            mu_diff_squared = (self.means - other.means) ** 2

            kl_div = (
                np.log(other.stds / self.stds)
                + (sigma0_squared + mu_diff_squared) / (2 * sigma1_squared)
                - 0.5
            )
            return np.sum(kl_div)

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
    return MultivariateGaussianDistribution, norm


@app.cell
def __(MultivariateGaussianDistribution):
    p = MultivariateGaussianDistribution(stds=[0.001, 0.001], names=["x", "y"])
    q = p.with_params([0.01, 0.01])
    p.kl_dist(q)
    return p, q


@app.cell
def __(
    Deque,
    List,
    MultivariateGaussianDistribution,
    NamedTuple,
    deque,
    minimize,
    np,
    scipy,
):
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

        def _estimate_success(
            self, cand: MultivariateGaussianDistribution
        ) -> float:
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
                    fun=lambda x: -self._estimate_success(
                        self.dist.with_params(x)
                    ),
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
                constraints=[success_constraint, kl_constraint]
                
            )
            return self.dist.with_params(res.x)
            
        def update_dist(self):
            # Implement "Dynamics distribution update" of Algorithm 1 lines 7-14,
            # updating self.dist in-place
            if len(self.buffer) < self.k:
                return
            
            if (
                np.mean([t.successful for t in self.buffer])
                < self.target_success_rate
            ):
                cand = self._find_feasable_dist()
                if self._estimate_success(cand) < self.target_success_rate:
                    self.dist = cand
                    return
                self.dist = cand
            self.dist = self._find_max_entropy_dist()
            
            

            
    return Doraemon, Trajectory


@app.cell
def __(Doraemon, MultivariateGaussianDistribution):
    dist = MultivariateGaussianDistribution([.5, 2], ["x", "y"])
    d = Doraemon(dist, 100, 0.01, 0.9)
    return d, dist


@app.cell
def __(d, np):
    params = []
    successes = []
    N = 10000
    for i in range(N):
        sample = d.dist.sample()
        success = all(np.abs(sample) < [5, 1])
        d.add_trajectory(sample, success)
        params.append(sample.copy())

        #params.append(d.dist.get_params())
        #successes.append(d._estimate_success(d.dist))
        if i > 100 and not i % 100:
            d.update_dist()
    return N, i, params, sample, success, successes


@app.cell
def __(N, np, params, plt):
    params_ = np.array(params)
    fig, ax = plt.subplots()
    ax.scatter(np.arange(N), params_[:,0], label="x", alpha=.1)
    ax.scatter(np.arange(N), params_[:,1], label="y", alpha=.1)
    #ax.plot(success, label="success")
    fig.legend()

    return ax, fig, params_


@app.cell
def __(params_):
    params_.shape
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
