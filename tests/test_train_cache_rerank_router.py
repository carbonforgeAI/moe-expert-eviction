from __future__ import annotations

import random

import torch

from moe_eviction.models.train_cache_rerank_router import (
    _select_reference_layers,
    _select_training_batch,
)


def test_select_training_batch_returns_indices_and_aligned_labels() -> None:
    random.seed(0)
    encoded = {
        "input_ids": torch.arange(24).reshape(6, 4),
        "attention_mask": torch.ones((6, 4), dtype=torch.long),
    }
    labels = encoded["input_ids"] + 100

    batch, batch_labels, indices = _select_training_batch(encoded, labels, batch_size=3)

    assert indices.shape == (3,)
    assert batch["input_ids"].shape == (3, 4)
    assert torch.equal(batch["input_ids"], encoded["input_ids"].index_select(0, indices))
    assert torch.equal(batch_labels, batch["input_ids"] + 100)


def test_select_reference_layers_selects_matching_flat_router_rows() -> None:
    flat_reference = torch.arange(24).reshape(6, 4)
    selected = _select_reference_layers(
        (flat_reference,),
        torch.tensor([0, 2]),
        total_batch_size=3,
        sequence_length=2,
    )

    expected = flat_reference.reshape(3, 2, 4).index_select(0, torch.tensor([0, 2]))
    assert torch.equal(selected[0], expected.reshape(4, 4))


def test_select_reference_layers_selects_matching_3d_router_rows() -> None:
    reference = torch.arange(24).reshape(3, 2, 4)

    selected = _select_reference_layers(
        (reference,),
        torch.tensor([1, 2]),
        total_batch_size=3,
        sequence_length=2,
    )

    assert torch.equal(selected[0], reference.index_select(0, torch.tensor([1, 2])))
