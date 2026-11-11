"""Pure (CPU, no model) tests for the MMLU accuracy vs switch-rate scoring helpers."""

from __future__ import annotations

import torch

from moe_eviction.models.evaluate_switch_accuracy import (
    format_prompt,
    letter_token_ids,
    score_answer,
    switch_rate,
)


class _FakeTokenizer:
    """Maps ' A'->[101], ' B'->[102], ... so letter_token_ids is deterministic without a model."""

    def encode(self, text: str, add_special_tokens: bool = False):
        letter = text.strip()[:1]
        return [100 + (ord(letter) - ord("A") + 1)]


def test_letter_token_ids_uses_leading_space_first_token() -> None:
    assert letter_token_ids(_FakeTokenizer(), 4) == [101, 102, 103, 104]


def test_score_answer_picks_highest_logit_choice() -> None:
    # vocab of 8; candidate letter tokens are ids [101,102,103,104] -> but here use small ids.
    logits = torch.zeros(10)
    logits[5] = 3.0  # candidate index 2 (id 5) is the argmax among candidates
    assert score_answer(logits, [3, 4, 5, 6]) == 2
    logits[6] = 9.0
    assert score_answer(logits, [3, 4, 5, 6]) == 3


def test_score_answer_ignores_non_candidate_tokens() -> None:
    logits = torch.zeros(10)
    logits[0] = 100.0  # a non-candidate token with a huge logit must not win
    logits[4] = 1.0
    assert score_answer(logits, [3, 4, 5, 6]) == 1


def test_switch_rate_is_inverse_dwell() -> None:
    assert switch_rate(4.0) == 0.25
    assert switch_rate(1.0) == 1.0
    assert switch_rate(None) is None
    assert switch_rate(0.0) is None


def test_format_prompt_lists_choices_and_answer_cue() -> None:
    p = format_prompt("2+2=?", ["3", "4", "5", "6"])
    assert "A. 3" in p and "D. 6" in p
    assert p.rstrip().endswith("Answer:")
