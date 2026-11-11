"""LoRA adapters over OLMoE's stacked expert FFNs.

The installed ``OlmoeExperts`` stores all experts as 3D ``nn.Parameter`` tensors
(``gate_up_proj[E, 2*inter, hidden]``, ``down_proj[E, hidden, inter]``) and applies expert
``e`` with ``F.linear(state, gate_up_proj[e])`` / ``F.linear(h, down_proj[e])`` inside a
per-expert loop. PEFT targets ``nn.Linear`` modules, so it cannot adapt these stacked
parameters; this module adds a small custom per-expert low-rank adapter instead.

``LoRAExperts`` wraps a frozen base experts module and adds, per expert ``e``:

    gate_up += scaling * (state @ A_gu[e]^T) @ B_gu[e]^T
    down    += scaling * (h     @ A_d[e]^T)  @ B_d[e]^T

with ``scaling = alpha / rank`` and ``B`` zero-initialized, so a freshly attached adapter
reproduces the base output exactly. Only the adapter factors carry gradients; the base expert
weights stay frozen. This is the co-adaptation lever for joint cache-pressured training — it
lets the expert representations move so that committing to an option becomes cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LoRAExpertsConfig:
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    targets: tuple[str, ...] = ("gate_up", "down")
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("lora.rank must be positive")
        for target in self.targets:
            if target not in {"gate_up", "down"}:
                raise ValueError(f"unsupported lora target: {target}")


def _is_olmoe_experts(module: nn.Module) -> bool:
    return all(hasattr(module, attribute) for attribute in ("gate_up_proj", "down_proj", "act_fn"))


class LoRAExperts(nn.Module):
    """Frozen ``OlmoeExperts`` plus trainable per-expert low-rank adapters."""

    def __init__(self, base: nn.Module, config: LoRAExpertsConfig) -> None:
        super().__init__()
        if not _is_olmoe_experts(base):
            raise TypeError("LoRAExperts expects an OlmoeExperts-like module")
        self.base = base
        self.config = config
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        num_experts, two_inter, hidden = base.gate_up_proj.shape
        _, _hidden, inter = base.down_proj.shape
        self.num_experts = int(num_experts)
        self.scaling = config.alpha / config.rank
        rank = config.rank
        ref = base.gate_up_proj
        factory = {"device": ref.device, "dtype": ref.dtype}

        self.adapt_gate_up = "gate_up" in config.targets
        self.adapt_down = "down" in config.targets

        if self.adapt_gate_up:
            self.a_gate_up = nn.Parameter(torch.empty(num_experts, rank, hidden, **factory))
            self.b_gate_up = nn.Parameter(torch.zeros(num_experts, two_inter, rank, **factory))
            nn.init.normal_(self.a_gate_up, std=config.init_std)
        if self.adapt_down:
            self.a_down = nn.Parameter(torch.empty(num_experts, rank, inter, **factory))
            self.b_down = nn.Parameter(torch.zeros(num_experts, hidden, rank, **factory))
            nn.init.normal_(self.a_down, std=config.init_std)

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("base.")
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Mirrors OlmoeExperts.forward, adding the LoRA delta to each expert's projections.
        base = self.base
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for raw_index in expert_hit:
            expert_idx = int(raw_index[0])
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            gate_up = F.linear(current_state, base.gate_up_proj[expert_idx])
            if self.adapt_gate_up:
                gate_up = gate_up + self.scaling * F.linear(
                    F.linear(current_state, self.a_gate_up[expert_idx]),
                    self.b_gate_up[expert_idx],
                )
            gate, up = gate_up.chunk(2, dim=-1)
            current_hidden_states = base.act_fn(gate) * up

            down = F.linear(current_hidden_states, base.down_proj[expert_idx])
            if self.adapt_down:
                down = down + self.scaling * F.linear(
                    F.linear(current_hidden_states, self.a_down[expert_idx]),
                    self.b_down[expert_idx],
                )
            down = down * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, down.to(final_hidden_states.dtype))

        return final_hidden_states


def _sparse_blocks_with_experts(model: nn.Module) -> list[nn.Module]:
    blocks: list[nn.Module] = []
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None or isinstance(experts, LoRAExperts):
            continue
        if _is_olmoe_experts(experts):
            blocks.append(module)
    return blocks


def attach_lora_experts(model: nn.Module, config: LoRAExpertsConfig) -> list[LoRAExperts]:
    """Replace each sparse block's ``.experts`` with a ``LoRAExperts`` wrapper. Returns them."""
    attached: list[LoRAExperts] = []
    for block in _sparse_blocks_with_experts(model):
        wrapped = LoRAExperts(block.experts, config)
        block.experts = wrapped
        attached.append(wrapped)
    if not attached:
        raise RuntimeError("no OLMoE expert modules found to attach LoRA adapters")
    return attached


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect adapter tensors (excluding frozen base weights) keyed by module path."""
    state: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, LoRAExperts):
            for param_name, parameter in module.named_parameters():
                if param_name.startswith("base."):
                    continue
                state[f"{module_name}.{param_name}"] = parameter.detach().cpu()
    return state


def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> tuple[str, ...]:
    """Load adapter tensors produced by :func:`lora_state_dict` into attached LoRAExperts."""
    model_params = dict(model.named_parameters())
    loaded: list[str] = []
    with torch.no_grad():
        for name, value in state.items():
            if name not in model_params:
                raise KeyError(f"adapter state has unknown parameter: {name}")
            parameter = model_params[name]
            if tuple(parameter.shape) != tuple(value.shape):
                raise ValueError(
                    f"shape mismatch for {name}: checkpoint {tuple(value.shape)} "
                    f"model {tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            loaded.append(name)
    return tuple(sorted(loaded))
