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


def test_doraemon_param_dict(beta_distribution):
    d = Doraemon(dist=beta_distribution, k=100, kl_bound=0.1, target_success_rate=0.9)
    res = d.param_dict()
    assert res
    for k, v in res.items():
        assert type(k) is str
        float(v)


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
    assert d.dist.entropy() < start_entropy
