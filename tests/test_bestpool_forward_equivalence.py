"""Equivalence oracle for the best-of-pool soft-forward speed reorg (precondition 0a).

``BestOfPoolBetaController._forward_soft`` was reorganized to precompute the gate once over all
tokens, batch the keep/fresh expert dispatch across the batch dimension, add a whole-layer fast path
for unrestricted layers, and hoist the attention-mask host sync out of the loop. None of that may
change the objective. This test runs the pre-reorg oracle (``_forward_soft_reference``) and the new
``_forward_soft`` on two twin controllers that share the same (frozen, forward-only) heads and
identical initial state, and asserts they agree within tolerance on:

- the blended output tensor for every layer call;
- every soft-loss term (beta / switch / deliberation / commitment / mu_coverage / router KL);
- the controller bookkeeping (controller_calls, beta_calls, events, cache_load_cost, dwell).

Exercised shapes cover a restricted layer with a padded (vanilla) row and an unrestricted layer
(the fast path). Pure CPU, no GPU / model download.
"""

from __future__ import annotations

import torch

from moe_eviction.models.learned_beta import (
    BestOfPoolBetaController,
    BetaHeadConfig,
    LearnedTerminationHead,
    PolicyHeadConfig,
    PolicyOverOptionsHead,
    policy_feature_dim,
)


class _IdentityGate(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class _AffineExpert(torch.nn.Module):
    def __init__(self, num_experts: int, expert_idx: int) -> None:
        super().__init__()
        torch.manual_seed(100 + expert_idx)
        self.lin = torch.nn.Linear(num_experts, num_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lin(hidden_states)


class _TinyMoe(torch.nn.Module):
    def __init__(self, num_experts: int = 4, top_k: int = 1) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = False
        self.gate = _IdentityGate()
        self.experts = torch.nn.ModuleList(
            [_AffineExpert(num_experts, i) for i in range(num_experts)]
        )


def _heads(num_experts: int, num_layers: int):
    torch.manual_seed(7)
    beta = LearnedTerminationHead(
        num_experts=num_experts, top_k=1, hidden_size=num_experts, num_layers=num_layers,
        config=BetaHeadConfig(feature_mode="henderson"),
    )
    policy = PolicyOverOptionsHead(
        num_experts=num_experts, num_layers=num_layers, feature_dim=policy_feature_dim(),
        config=PolicyHeadConfig(hidden_dim=8, shared_across_layers=False),
    )
    beta.eval()
    policy.eval()
    return beta, policy


def _controller(beta, policy, mask, *, num_layers, restricted, num_experts=4):
    return BestOfPoolBetaController(
        attention_mask=mask,
        num_layers=num_layers, capacity=2, top_k=1,
        termination_head=beta, mode="soft",
        policy_head=policy, restricted_layers=frozenset(restricted),
    )


def _terms(ctrl):
    device = torch.device("cpu")
    soft = ctrl.soft_loss_terms(device)
    out = {k: float(v) for k, v in soft.items()}
    out["router_kl"] = float(ctrl.router_kl_loss(device))
    out["controller_calls"] = ctrl.controller_calls
    out["beta_calls"] = ctrl.beta_calls
    out["events"] = ctrl.events
    out["cache_load_cost"] = float(ctrl.cache_load_cost)
    ctrl.finalize()
    out["mean_option_length"] = float(ctrl.mean_option_length)
    return out


def test_reorg_matches_reference_within_tolerance() -> None:
    num_experts, num_layers = 4, 3
    batch, seq = 2, 6
    block = _TinyMoe(num_experts=num_experts)
    beta, policy = _heads(num_experts, num_layers)

    # A padded tail on the second sequence exercises the vanilla row inside a restricted layer.
    mask = torch.ones(batch, seq, dtype=torch.bool)
    mask[1, seq - 2 :] = False

    torch.manual_seed(0)
    hiddens = [torch.randn(batch, seq, num_experts) for _ in range(num_layers)]
    # layer 0 restricted (pooled), layers 1 and 2 unrestricted (fast path).
    restricted = {0}

    ref = _controller(beta, policy, mask, num_layers=num_layers, restricted=restricted)
    new = _controller(beta, policy, mask, num_layers=num_layers, restricted=restricted)

    for layer in range(num_layers):
        out_ref, _ = ref._forward_soft_reference(block, layer, hiddens[layer].clone())
        out_new, _ = new._forward_soft(block, layer, hiddens[layer].clone())
        assert torch.allclose(out_ref, out_new, atol=1e-6, rtol=1e-5), (
            f"layer {layer} output mismatch: max |d| = "
            f"{(out_ref - out_new).abs().max().item():.3e}"
        )

    ref_terms = _terms(ref)
    new_terms = _terms(new)
    assert set(ref_terms) == set(new_terms)
    for key in ref_terms:
        assert abs(ref_terms[key] - new_terms[key]) <= 1e-5 + 1e-5 * abs(ref_terms[key]), (
            f"term {key} mismatch: ref={ref_terms[key]!r} new={new_terms[key]!r}"
        )


def test_reorg_matches_reference_multi_restricted() -> None:
    # Two restricted layers, no padding, larger batch -- stresses the batched keep/fresh dispatch.
    num_experts, num_layers = 6, 2
    batch, seq = 3, 5
    block = _TinyMoe(num_experts=num_experts)
    beta, policy = _heads(num_experts, num_layers)
    mask = torch.ones(batch, seq, dtype=torch.bool)
    torch.manual_seed(3)
    hiddens = [torch.randn(batch, seq, num_experts) for _ in range(num_layers)]
    restricted = {0, 1}

    ref = _controller(beta, policy, mask, num_layers=num_layers, restricted=restricted,
                      num_experts=num_experts)
    new = _controller(beta, policy, mask, num_layers=num_layers, restricted=restricted,
                      num_experts=num_experts)
    for layer in range(num_layers):
        out_ref, _ = ref._forward_soft_reference(block, layer, hiddens[layer].clone())
        out_new, _ = new._forward_soft(block, layer, hiddens[layer].clone())
        assert torch.allclose(out_ref, out_new, atol=1e-6, rtol=1e-5)

    ref_terms, new_terms = _terms(ref), _terms(new)
    for key in ref_terms:
        assert abs(ref_terms[key] - new_terms[key]) <= 1e-5 + 1e-5 * abs(ref_terms[key])
