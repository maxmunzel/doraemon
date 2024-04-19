import numpy as np
import pytest
from .main import Doraemon, MultivariateBetaDistribution


@pytest.fixture
def beta_distribution():
    # Assuming the MultivariateBetaDistribution class has been modified to include a 'param_bound' attribute
    dist = MultivariateBetaDistribution(
        alphas=[10, 10],
        names=["x", "y"],
        low=[-5, -5],
        high=[5, 5],
        param_bound=[10, 10],
    )
    return dist


def test_out_of_bounds_params():
    """Dist params may be a little bit away from the limits due to floating point errors. We need to handle this gracefully."""
    dist = MultivariateBetaDistribution(
        alphas=[10, 10],
        names=["x", "y"],
        low=[5, -5],
        high=[5, 5],
        param_bound=[10, 10],
    )
    d = Doraemon(dist=dist, k=100, kl_bound=0.1, target_success_rate=0.9)

    for i in range(301):
        sample = dist.sample()
        d.add_trajectory(sample, 0 == i % 2)
        d.update_dist()
        L = dist.likelihood(sample)
        assert np.isfinite(L)
        assert np.all(np.isfinite(dist.sample()))
        assert np.all(np.isfinite(dist.get_params()))
        params = dist.get_params()
        if i % 6 == 0:
            params[0] = 0.9
        elif i % 6 == 1:
            params[0] = 11
        elif i % 6 == 2:
            params[0] == 0
        elif i % 6 == 3:
            params[0] == np.inf
        elif i % 6 == 4:
            params[0] == -np.inf
        elif i % 6 == 5:
            params[0] == np.nan

        dist.set_params(params)


def test_out_of_bounds_likelihood():
    """Samples may be a little bit away from the distribution due to floating point errors. We need to handle this gracefully."""
    dist = MultivariateBetaDistribution(
        alphas=[10, 10],
        names=["x", "y"],
        low=[5, -5],
        high=[5, 5],
        param_bound=[10, 10],
    )
    d = Doraemon(dist=dist, k=100, kl_bound=0.1, target_success_rate=0.9)

    for i, eps in enumerate(np.linspace(0, 0.1, 1000)):
        sample = [5 + eps, -5 - eps]
        d.add_trajectory(sample, 0 == i % 2)
        d.update_dist()
        L = dist.likelihood(sample)
        assert np.isfinite(L)
        assert np.all(np.isfinite(dist.sample()))
        assert np.all(np.isfinite(dist.get_params()))


def test_equal_bounds():
    """Setting low=high should be supported for convenience"""
    dist = MultivariateBetaDistribution(
        alphas=[10, 10],
        names=["x", "y"],
        low=[5, -5],
        high=[5, 5],
        param_bound=[10, 10],
    )
    x, y = dist.sample()
    assert abs(x - 5) < 0.001

    L = dist.likelihood(np.array([5, 0]))
    assert np.isfinite(L)


def test_doraemon_param_dict(beta_distribution):
    d = Doraemon(dist=beta_distribution, k=100, kl_bound=0.1, target_success_rate=0.9)
    res = d.param_dict()
    assert res
    for k, v in res.items():
        assert type(k) is str
        float(v)


def test_alphas_stay_above_one():
    dist = MultivariateBetaDistribution(
        alphas=[2],
        names=["x"],
        low=[-5],
        high=[5],
        param_bound=[10],
        seed=42,
    )
    d = Doraemon(
        dist=dist,
        k=10,
        kl_bound=0.1,
        target_success_rate=0.2,
    )
    N = 400
    start_entropy = d.dist.entropy()
    for i in range(N):
        sample = d.dist.sample()
        d.add_trajectory(list(sample), True)
        d.update_dist()
        assert np.all(d.dist.get_params() >= 1)
    assert d.dist.entropy() > start_entropy


def test_doraemon_updates_increasing_entropy():
    # test if the entropy grows in the face of success
    dist = MultivariateBetaDistribution(
        alphas=[10, 10],
        names=["x", "y"],
        low=[-5, -5],
        high=[5, 5],
        param_bound=[10, 10],
        seed=42,
    )
    d = Doraemon(
        dist=dist,
        k=100,
        kl_bound=0.1,
        target_success_rate=0.9,
    )
    N = 200
    start_entropy = d.dist.entropy()
    for i in range(N):
        sample = d.dist.sample()
        d.add_trajectory(list(sample), True)  # everything is successful
        if i == 199:
            pass
            # breakpoint()
        old_params = d.dist.get_params()
        d.update_dist()
        assert dist.with_params(old_params).kl_dist(d.dist) <= d.kl_bound * 1.01
        assert np.all(d.dist.get_params() <= d.dist.param_bound)
        assert np.all(d.dist.get_params() >= 1)
    assert d.dist.entropy() > start_entropy


def test_doraemon_updates_decreasing_entropy():
    # test if the entropy shrinks in the face of failure
    dist = MultivariateBetaDistribution(
        alphas=[2, 2],
        names=["x", "y"],
        low=[-5, -5],
        high=[5, 5],
        param_bound=[3, 3],
        seed=42,
    )
    d = Doraemon(
        dist=dist,
        k=100,
        kl_bound=0.1,
        target_success_rate=0.9,
    )
    N = 200
    start_entropy = d.dist.entropy()
    for i in range(N):
        sample = d.dist.sample()
        success = max(abs(sample)) < 3
        d.add_trajectory(list(sample), success)  # everything fails
        old_params = d.dist.get_params()
        d.update_dist()
        assert dist.with_params(old_params).kl_dist(d.dist) <= d.kl_bound * 1.01
        assert np.all(d.dist.get_params() <= d.dist.param_bound)
        assert np.all(d.dist.get_params() >= 1)
    assert d.dist.entropy() < start_entropy
