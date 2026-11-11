import math

import torch

from moe_eviction.models.train_learned_beta import _differentiable_cache_loss


def _terms(deliberation, switch, load, commitment):
    return {
        "deliberation": torch.tensor(deliberation),
        "switch": torch.tensor(switch),
        "load": torch.tensor(load),
        "commitment": torch.tensor(commitment),
    }


def test_differentiable_cache_loss_weighted_sum() -> None:
    lm = torch.tensor(2.0)
    soft = _terms(0.5, 0.1, 0.4, 0.25)
    router_kl = torch.tensor(0.2)
    weights = {"deliberation": 0.1, "switch": 0.3, "load": 0.05, "commit": 0.2, "router": 0.5}
    got = _differentiable_cache_loss(lm, soft, router_kl, weights)
    expected = 2.0 + 0.1 * 0.5 + 0.3 * 0.1 + 0.05 * 0.4 + 0.2 * 0.25 + 0.5 * 0.2
    assert math.isclose(float(got), expected, rel_tol=1e-6)


def test_zero_weights_recover_lm_loss() -> None:
    lm = torch.tensor(3.14)
    soft = _terms(1.0, 1.0, 1.0, 1.0)
    weights = {"deliberation": 0.0, "switch": 0.0, "load": 0.0, "commit": 0.0, "router": 0.0}
    got = _differentiable_cache_loss(lm, soft, torch.tensor(9.9), weights)
    assert math.isclose(float(got), 3.14, rel_tol=1e-6)


def test_loss_is_differentiable_through_costs() -> None:
    lm = torch.tensor(1.0, requires_grad=True)
    beta = torch.tensor(0.4, requires_grad=True)
    # deliberation = beta, commitment = beta*(1-beta) — both functions of a trainable beta.
    soft = {
        "deliberation": beta,
        "switch": beta * 2.0,
        "load": beta * 0.0,
        "commitment": beta * (1.0 - beta),
    }
    weights = {"deliberation": 0.1, "switch": 0.3, "load": 0.0, "commit": 0.2, "router": 0.0}
    loss = _differentiable_cache_loss(lm, soft, torch.tensor(0.0), weights)
    loss.backward()
    # d/dbeta [0.1*beta + 0.3*2beta + 0.2*beta(1-beta)] at beta=0.4 = 0.1+0.6+0.2*(1-0.8)=0.74
    assert beta.grad is not None
    assert math.isclose(float(beta.grad), 0.74, rel_tol=1e-5)
    assert math.isclose(float(lm.grad), 1.0, rel_tol=1e-6)
