import torch
import torch.nn.functional as F
from torch import nn

from moe_eviction.models.lora_experts import (
    LoRAExperts,
    LoRAExpertsConfig,
    attach_lora_experts,
    load_lora_state,
    lora_state_dict,
)


class TinyExperts(nn.Module):
    """Stand-in mirroring the installed OlmoeExperts forward (no model download)."""

    def __init__(self, num_experts: int, hidden: int, inter: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden
        self.intermediate_dim = inter
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, inter))
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for raw in expert_hit:
            expert_idx = int(raw[0])
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current = self.act_fn(gate) * up
            current = F.linear(current, self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
        return final_hidden_states


class TinyBlock(nn.Module):
    def __init__(self, experts: nn.Module) -> None:
        super().__init__()
        self.gate = nn.Linear(experts.hidden_dim, experts.num_experts)
        self.experts = experts


class TinyModel(nn.Module):
    def __init__(self, num_experts: int, hidden: int, inter: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TinyBlock(TinyExperts(num_experts, hidden, inter)) for _ in range(2)]
        )


def _inputs(num_experts: int, hidden: int, top_k: int, tokens: int = 5):
    torch.manual_seed(1)
    hidden_states = torch.randn(tokens, hidden)
    top_k_index = torch.stack(
        [torch.randperm(num_experts)[:top_k] for _ in range(tokens)]
    )
    top_k_weights = torch.softmax(torch.randn(tokens, top_k), dim=-1)
    return hidden_states, top_k_index, top_k_weights


def test_zero_init_lora_matches_base() -> None:
    torch.manual_seed(0)
    base = TinyExperts(num_experts=4, hidden=6, inter=8)
    inputs = _inputs(4, 6, top_k=2)
    expected = base(*inputs)
    lora = LoRAExperts(base, LoRAExpertsConfig(enabled=True, rank=2, alpha=4.0))
    got = lora(*inputs)
    assert torch.allclose(got, expected, atol=1e-6)


def test_nonzero_lora_changes_output_and_grads_flow() -> None:
    torch.manual_seed(0)
    base = TinyExperts(num_experts=4, hidden=6, inter=8)
    inputs = _inputs(4, 6, top_k=2)
    lora = LoRAExperts(base, LoRAExpertsConfig(enabled=True, rank=2, alpha=4.0))
    with torch.no_grad():
        lora.b_gate_up.normal_(std=0.1)
        lora.b_down.normal_(std=0.1)
    out = lora(*inputs)
    assert not torch.allclose(out, base(*inputs), atol=1e-5)
    out.sum().backward()
    # Adapter factors receive gradient; frozen base does not.
    assert lora.b_gate_up.grad is not None and lora.b_gate_up.grad.abs().sum() > 0
    assert lora.b_down.grad is not None
    assert base.gate_up_proj.grad is None
    assert base.down_proj.grad is None


def test_only_adapter_params_trainable() -> None:
    base = TinyExperts(num_experts=3, hidden=4, inter=4)
    lora = LoRAExperts(base, LoRAExpertsConfig(enabled=True, rank=2, alpha=2.0))
    trainable = {name for name, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"a_gate_up", "b_gate_up", "a_down", "b_down"}
    assert all(not p.requires_grad for n, p in lora.named_parameters() if n.startswith("base."))
    assert len(lora.adapter_parameters()) == 4


def test_targets_subset_only_adapts_gate_up() -> None:
    base = TinyExperts(num_experts=3, hidden=4, inter=4)
    lora = LoRAExperts(base, LoRAExpertsConfig(enabled=True, rank=2, targets=("gate_up",)))
    names = {name for name, _ in lora.named_parameters() if not name.startswith("base.")}
    assert names == {"a_gate_up", "b_gate_up"}


def test_attach_and_state_roundtrip() -> None:
    torch.manual_seed(0)
    model = TinyModel(num_experts=4, hidden=6, inter=8)
    attached = attach_lora_experts(model, LoRAExpertsConfig(enabled=True, rank=2, alpha=4.0))
    assert len(attached) == 2
    assert all(isinstance(block.experts, LoRAExperts) for block in model.blocks)
    # Perturb adapters, snapshot, zero them, reload, and confirm exact recovery.
    with torch.no_grad():
        for module in attached:
            module.b_gate_up.normal_(std=0.1)
    state = lora_state_dict(model)
    assert state and all(".base." not in key for key in state)
    with torch.no_grad():
        for module in attached:
            module.b_gate_up.zero_()
    loaded = load_lora_state(model, state)
    assert len(loaded) == len(state)
    module_names = {id(module): name for name, module in model.named_modules()}
    for module in attached:
        key = module_names[id(module)]
        assert torch.allclose(module.b_gate_up, state[f"{key}.b_gate_up"].to(module.b_gate_up))
