from __future__ import annotations

import sys
import types
from pathlib import Path

from moe_eviction.models.collect_trace import _extract_prompt_from_record, _load_prompts


def test_load_prompts_from_file_ignores_blank_lines(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(" first prompt \n\nsecond prompt\n", encoding="utf-8")

    prompts = _load_prompts({"trace": {"prompt_file": str(prompt_file)}})

    assert prompts == ["first prompt", "second prompt"]


def test_extract_prompt_from_conversation_record() -> None:
    record = {
        "conversations": [
            {"from": "human", "value": "Explain cache locality."},
            {"from": "assistant", "value": "Cache locality is reuse over time."},
        ]
    }

    prompt = _extract_prompt_from_record(record, {"conversation_field": "conversations"})

    assert prompt == "Explain cache locality.\nCache locality is reuse over time."


def test_extract_prompt_from_instruction_and_input_record() -> None:
    prompt = _extract_prompt_from_record(
        {"instruction": "Summarize this.", "input": "MoE routers select experts."},
        {},
    )

    assert prompt == "Summarize this.\nMoE routers select experts."


def test_load_prompts_from_dataset_module(monkeypatch: object) -> None:
    calls: dict[str, object] = {}

    class FakeDataset(list[dict[str, str]]):
        def shuffle(self, *, seed: int, buffer_size: int | None = None) -> FakeDataset:
            calls["seed"] = seed
            calls["buffer_size"] = buffer_size
            return self

    def load_dataset(**kwargs: object) -> FakeDataset:
        calls.update(kwargs)
        return FakeDataset(
            [
                {"text": " first "},
                {"text": ""},
                {"text": "second"},
                {"text": "third"},
            ]
        )

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)

    prompts = _load_prompts(
        {
            "trace": {
                "max_sequences": 3,
                "prompt_dataset": {
                    "path": "fake/dataset",
                    "split": "train[:10]",
                    "text_field": "text",
                    "shuffle": True,
                    "seed": 7,
                    "shuffle_buffer_size": 128,
                },
            }
        }
    )

    assert prompts == ["first", "second"]
    assert calls["path"] == "fake/dataset"
    assert calls["split"] == "train[:10]"
    assert calls["streaming"] is False
    assert calls["seed"] == 7
    assert calls["buffer_size"] == 128


def test_load_prompts_from_dataset_skips_records(monkeypatch: object) -> None:
    class FakeDataset(list[dict[str, str]]):
        pass

    def load_dataset(**_kwargs: object) -> FakeDataset:
        return FakeDataset(
            [
                {"text": "train zero"},
                {"text": "train one"},
                {"text": "heldout zero"},
                {"text": "heldout one"},
                {"text": "unused"},
            ]
        )

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)

    prompts = _load_prompts(
        {
            "trace": {
                "prompt_dataset": {
                    "path": "fake/dataset",
                    "text_field": "text",
                    "skip_records": 2,
                    "max_records": 2,
                },
            }
        }
    )

    assert prompts == ["heldout zero", "heldout one"]
