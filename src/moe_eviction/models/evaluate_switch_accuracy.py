"""MMLU/MMMLU accuracy vs switch-rate under the frozen best-of-pool controller.

Commensurability with Shen & Henderson, "Temporally Extended Mixture-of-Experts Models"
(arXiv:2604.20156): they report **downstream accuracy retention vs expert switch rate** (switch rate
>50% -> <5% while retaining up to 90% of base accuracy on MATH/MMLU/MMMLU, on gpt-oss-20b + LoRA +
self-distillation). This module produces the analogous curve for our *frozen* OLMoE-1B-7B under
the learned beta+mu best-of-pool controller and the freeze baselines.

Metric mechanics. MMLU-style questions are scored by the log-prob the model assigns to the answer
*letter* (A/B/C/D) at the position right after "Answer:" -- a single-token continuation, so one
forward per question. The controller's gate hooks run during that forward, so the answer-position
logits already reflect the cache-restricted routing. Switch rate = fraction of tokens starting a
new option (= 1 / mean dwell); accuracy retention = controller acc / vanilla acc. MATH (generative,
needs answer extraction + equivalence) is intentionally out of scope here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .evaluate_router_state import _load_router_state
from .evaluate_working_set import (
    FreezePoolWindowController,
    LearnedBetaPoolController,
    _attach,
    _attach_learned,
    _dtype,
    _gate_modules,
    load_learned_beta_checkpoint,
)
from .learned_beta import load_policy_head

_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


@dataclass(frozen=True, slots=True)
class AccuracyRow:
    policy: str
    accuracy: float
    n_questions: int
    n_correct: int
    total_tokens: int
    transfers: int
    transfer_rate: float
    # switch_rate = fraction of tokens that start a new option (= 1 / mean_dwell); the
    # Shen-Henderson axis. None for vanilla. mean_dwell/window_size/beta_threshold are policy tags.
    switch_rate: float | None = None
    mean_dwell: float | None = None
    window_size: int | None = None
    beta_threshold: float | None = None


# --------------------------------------------------------------------------- pure, testable helpers
def letter_token_ids(tokenizer: Any, n_choices: int) -> list[int]:
    """First token id of each answer letter as it appears after 'Answer:' (i.e. ' A', ' B', ...)."""
    ids: list[int] = []
    for letter in _LETTERS[:n_choices]:
        toks = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if not toks:
            toks = tokenizer.encode(letter, add_special_tokens=False)
        ids.append(int(toks[0]))
    return ids


def score_answer(logits_row: torch.Tensor, candidate_ids: list[int]) -> int:
    """Index of the highest-logit candidate letter at the answer position (argmax over choices)."""
    cand = torch.tensor(candidate_ids, device=logits_row.device)
    return int(torch.argmax(logits_row.index_select(0, cand)))


def format_prompt(question: str, choices: list[str]) -> str:
    lines = [
        "The following is a multiple choice question (with answer).",
        "",
        question.strip(),
    ]
    for letter, choice in zip(_LETTERS, choices, strict=False):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def switch_rate(mean_dwell: float | None) -> float | None:
    if mean_dwell is None or mean_dwell <= 0:
        return None
    return 1.0 / mean_dwell


# --------------------------------------------------------------------------- dataset + batching
def _load_questions(eval_config: dict[str, Any]) -> list[tuple[str, list[str], int]]:
    ds_cfg = eval_config.get("task_dataset", {"path": "cais/mmlu", "name": "all", "split": "test"})
    kwargs = {k: v for k, v in ds_cfg.items() if k not in {"question_field", "choices_field",
                                                            "answer_field", "max_questions"}}
    dataset = load_dataset(**kwargs)
    qf = ds_cfg.get("question_field", "question")
    cf = ds_cfg.get("choices_field", "choices")
    af = ds_cfg.get("answer_field", "answer")
    limit = int(ds_cfg.get("max_questions", eval_config.get("max_questions", 500)))
    out: list[tuple[str, list[str], int]] = []
    for row in dataset:
        choices = list(row[cf])
        answer = int(row[af])
        out.append((str(row[qf]), [str(c) for c in choices], answer))
        if len(out) >= limit:
            break
    return out


def _batched(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --------------------------------------------------------------------------- eval core
def _run_policy(
    *,
    label: str,
    model: torch.nn.Module,
    tokenizer: Any,
    questions: list[tuple[str, list[str], int]],
    candidate_ids: list[int],
    device: torch.device,
    batch_size: int,
    max_length: int,
    make_controller: Any | None,
    attach: Any | None,
    window_size: int | None = None,
    beta_threshold: float | None = None,
) -> AccuracyRow:
    n_correct = 0
    total_tokens = 0
    transfers = 0
    option_lengths: list[int] = []
    for batch in _batched(questions, batch_size):
        prompts = [format_prompt(q, c) for q, c, _ in batch]
        golds = [a for _, _, a in batch]
        enc = tokenizer(
            prompts, return_tensors="pt", padding="max_length", truncation=True,
            max_length=max_length,
        ).to(device)
        controller = make_controller(enc["attention_mask"]) if make_controller else None
        handles = attach(model, controller) if (attach and controller) else []
        try:
            with torch.no_grad():
                out = model(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    use_cache=False,
                )
            logits = out.logits
            last = enc["attention_mask"].sum(dim=1) - 1
            for b in range(logits.shape[0]):
                pred = score_answer(logits[b, int(last[b])], candidate_ids[: len(batch[b][1])])
                n_correct += int(pred == golds[b])
        finally:
            for handle in handles:
                handle.remove()
        if controller is not None:
            if hasattr(controller, "finalize"):
                controller.finalize()  # close open options so their dwell is counted (learned)
            transfers += int(controller.transfers)
            total_tokens += int(enc["attention_mask"].sum())
            option_lengths.extend(getattr(controller, "option_lengths", []))
    n = len(questions)
    mean_dwell: float | None
    if window_size is not None:
        mean_dwell = float(window_size)
    elif option_lengths:
        mean_dwell = sum(option_lengths) / len(option_lengths)
    else:
        mean_dwell = None
    return AccuracyRow(
        policy=label,
        accuracy=n_correct / n if n else 0.0,
        n_questions=n,
        n_correct=n_correct,
        total_tokens=total_tokens,
        transfers=transfers,
        transfer_rate=transfers / total_tokens if total_tokens else 0.0,
        switch_rate=switch_rate(mean_dwell),
        mean_dwell=mean_dwell,
        window_size=window_size,
        beta_threshold=beta_threshold,
    )


def run(
    config: dict[str, Any],
    output_dir: Path,
    router_state_path: Path | None = None,
    learned_beta_state: Path | None = None,
    policy_head_state: Path | None = None,
) -> None:
    seed = int(config.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    model_config = config["model"]
    eval_config = config.get("eval", {})
    model_id = model_config["id"]
    revision = model_config.get("revision")
    device = torch.device(model_config.get("device", "cuda"))
    dtype = _dtype(model_config.get("dtype", "bfloat16"))
    batch_size = int(eval_config.get("batch_size", 8))
    max_length = int(eval_config.get("max_length", 384))
    restrict_layers_cfg = eval_config.get("restrict_layers")
    freeze_capacity = int(eval_config.get("freeze_capacity", 32))
    freeze_durations = tuple(int(w) for w in eval_config.get("freeze_durations", []))
    freeze_pool_selection = str(eval_config.get("freeze_pool_selection", "oracle"))
    learned_beta_thresholds = tuple(
        float(t) for t in eval_config.get("learned_beta_thresholds", [])
    )
    learned_beta_capacity = int(eval_config.get("learned_beta_capacity", 32))
    preload_router_state = bool(eval_config.get("preload_router_state", False))

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    questions = _load_questions(eval_config)
    candidate_ids = letter_token_ids(tokenizer, 4)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    if router_state_path is not None and preload_router_state:
        _load_router_state(model, router_state_path)

    top_k = int(getattr(model.config, "num_experts_per_tok", 8))
    num_layers = len(_gate_modules(model))
    selective = (
        frozenset(int(x) for x in restrict_layers_cfg)
        if restrict_layers_cfg is not None
        else frozenset(range(3, num_layers))
    )
    num_experts = int(getattr(model.config, "num_experts", 64))

    rows: list[AccuracyRow] = []
    # Vanilla base accuracy (no controller, free routing).
    rows.append(_run_policy(
        label="vanilla", model=model, tokenizer=tokenizer, questions=questions,
        candidate_ids=candidate_ids, device=device, batch_size=batch_size, max_length=max_length,
        make_controller=None, attach=None,
    ))

    # Freeze baselines (fixed-k dwell): oracle ceiling / causal fair baseline / trivial predictors.
    for window in freeze_durations:
        def _mk_freeze(mask, window=window):
            return FreezePoolWindowController(
                attention_mask=mask, num_layers=num_layers, capacity=freeze_capacity,
                top_k=top_k, window_size=window, restricted_layers=selective,
                pool_selection=freeze_pool_selection,
            )
        rows.append(_run_policy(
            label=f"freeze_{freeze_pool_selection}", model=model, tokenizer=tokenizer,
            questions=questions, candidate_ids=candidate_ids, device=device,
            batch_size=batch_size, max_length=max_length, make_controller=_mk_freeze,
            attach=_attach, window_size=window,
        ))

    # Learned beta+mu across termination thresholds.
    if learned_beta_thresholds and learned_beta_state is not None and policy_head_state is not None:
        beta_head = load_learned_beta_checkpoint(Path(learned_beta_state), device)
        policy_head = load_policy_head(Path(policy_head_state), device)
        for threshold in learned_beta_thresholds:
            def _mk_learned(mask, threshold=threshold):
                return LearnedBetaPoolController(
                    attention_mask=mask, num_layers=num_layers, num_experts=num_experts,
                    capacity=learned_beta_capacity, top_k=top_k, beta_head=beta_head,
                    policy_head=policy_head, threshold=threshold, restricted_layers=selective,
                )
            rows.append(_run_policy(
                label="learned_beta_pool", model=model, tokenizer=tokenizer, questions=questions,
                candidate_ids=candidate_ids, device=device, batch_size=batch_size,
                max_length=max_length, make_controller=_mk_learned, attach=_attach_learned,
                beta_threshold=threshold,
            ))

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "accuracy_eval.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row)) + "\n")
    base_acc = next((r.accuracy for r in rows if r.policy == "vanilla"), float("nan"))
    metadata = {
        "seed": seed, "model_id": model_id, "revision": revision, "num_layers": num_layers,
        "top_k": top_k, "num_experts": num_experts, "n_questions": len(questions),
        "base_accuracy": base_acc, "task_dataset": eval_config.get("task_dataset"),
        "freeze_pool_selection": freeze_pool_selection, "max_length": max_length,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for row in rows:
        ret = row.accuracy / base_acc if base_acc else float("nan")
        sr = f"{row.switch_rate:.3f}" if row.switch_rate is not None else "  -  "
        print(f"{row.policy:<20} acc={row.accuracy:.3f} ret={ret:.3f} switch={sr} "
              f"dwell={row.mean_dwell} thr={row.beta_threshold}")


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description="MMLU accuracy vs switch-rate under best-of-pool.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--router-state", type=Path)
    parser.add_argument("--learned-beta-state", type=Path)
    parser.add_argument("--policy-head-state", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run(config, args.output_dir, args.router_state, args.learned_beta_state, args.policy_head_state)


if __name__ == "__main__":
    main()
