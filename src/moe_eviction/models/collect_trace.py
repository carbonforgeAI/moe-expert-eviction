from __future__ import annotations

import argparse
import importlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = (
    "Mixture-of-experts routing should preserve language quality while reducing transfers.",
    "A cache miss occurs when a selected expert is absent before the routing step.",
    "Hardware-priced routing should distinguish byte cost from unit miss count.",
    "The experiment compares co-training with post-hoc cache-conditioned reranking.",
)


def _dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _load_prompts(config: dict[str, Any]) -> list[str]:
    trace_config = config.get("trace", {})
    prompt_file = trace_config.get("prompt_file")
    if prompt_file is not None:
        path = Path(prompt_file)
        prompts: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prompts.append(line)
        if not prompts:
            raise ValueError(f"prompt file is empty: {path}")
        return prompts

    dataset_config = trace_config.get("prompt_dataset", trace_config.get("dataset"))
    if dataset_config is not None:
        return _load_dataset_prompts(dataset_config, trace_config)

    return list(DEFAULT_PROMPTS)


def _load_dataset_prompts(
    dataset_config: str | dict[str, Any],
    trace_config: dict[str, Any],
) -> list[str]:
    if isinstance(dataset_config, str):
        dataset_config = {"path": dataset_config}
    if not isinstance(dataset_config, dict):
        raise TypeError("prompt_dataset/dataset must be a mapping or dataset path string")
    dataset_path = dataset_config.get("path")
    if not dataset_path:
        raise ValueError("prompt_dataset.path is required")

    try:
        datasets = importlib.import_module("datasets")
    except ImportError as exc:
        raise RuntimeError(
            "dataset-backed prompt loading requires the optional 'datasets' dependency"
        ) from exc

    split = str(dataset_config.get("split", "train"))
    load_kwargs: dict[str, Any] = {
        "path": str(dataset_path),
        "split": split,
        "streaming": bool(dataset_config.get("streaming", False)),
    }
    if dataset_config.get("name") is not None:
        load_kwargs["name"] = dataset_config["name"]
    if dataset_config.get("data_files") is not None:
        load_kwargs["data_files"] = dataset_config["data_files"]
    if bool(dataset_config.get("trust_remote_code", False)):
        load_kwargs["trust_remote_code"] = True

    dataset = datasets.load_dataset(**load_kwargs)
    if bool(dataset_config.get("shuffle", False)):
        seed = int(dataset_config.get("seed", trace_config.get("seed", 0)))
        buffer_size = int(dataset_config.get("shuffle_buffer_size", 10_000))
        try:
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
        except TypeError:
            dataset = dataset.shuffle(seed=seed)

    max_records = int(
        dataset_config.get(
            "max_records",
            trace_config.get("max_sequences", 0),
        )
    )
    if max_records < 0:
        raise ValueError("prompt_dataset.max_records must be non-negative")
    skip_records = int(dataset_config.get("skip_records", 0))
    if skip_records < 0:
        raise ValueError("prompt_dataset.skip_records must be non-negative")

    prompts: list[str] = []
    seen_records = 0
    for index, record in enumerate(dataset):
        if index < skip_records:
            continue
        if max_records and seen_records >= max_records:
            break
        seen_records += 1
        prompt = _extract_prompt_from_record(record, dataset_config)
        if prompt:
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"dataset prompt source produced no non-empty prompts: {dataset_path}")
    return prompts


def _extract_prompt_from_record(record: Any, dataset_config: dict[str, Any]) -> str:
    if not isinstance(record, dict):
        return _prompt_value_to_text(record)

    text_field = (
        dataset_config.get("text_field")
        or dataset_config.get("prompt_field")
        or dataset_config.get("field")
    )
    if text_field is not None:
        return _prompt_value_to_text(_nested_get(record, str(text_field)))

    conversation_field = dataset_config.get("conversation_field")
    if conversation_field is not None:
        return _conversation_to_text(
            _nested_get(record, str(conversation_field)),
            dataset_config,
        )

    instruction = _prompt_value_to_text(record.get("instruction"))
    input_text = _prompt_value_to_text(record.get("input"))
    if instruction and input_text:
        return f"{instruction}\n{input_text}"
    if instruction:
        return instruction

    for field in ("text", "prompt", "question", "content"):
        if field in record:
            prompt = _prompt_value_to_text(record[field])
            if prompt:
                return prompt

    for field in ("messages", "conversations", "conversation"):
        if field in record:
            prompt = _conversation_to_text(record[field], dataset_config)
            if prompt:
                return prompt

    return input_text


def _nested_get(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _conversation_to_text(value: Any, dataset_config: dict[str, Any]) -> str:
    if not isinstance(value, list):
        return _prompt_value_to_text(value)
    content_field = str(dataset_config.get("content_field", "content"))
    alternate_content_field = str(dataset_config.get("alternate_content_field", "value"))
    pieces: list[str] = []
    for message in value:
        if isinstance(message, dict):
            content = message.get(content_field, message.get(alternate_content_field))
        else:
            content = message
        text = _prompt_value_to_text(content)
        if text:
            pieces.append(text)
    return "\n".join(pieces).strip()


def _prompt_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        pieces = [_prompt_value_to_text(item) for item in value]
        return "\n".join(piece for piece in pieces if piece).strip()
    return str(value).strip()


def _router_layers(router_logits: Any) -> list[torch.Tensor]:
    if router_logits is None:
        raise RuntimeError(
            "model output did not include router logits; verify Transformers support for "
            "output_router_logits=True on this model"
        )
    if isinstance(router_logits, torch.Tensor):
        return [router_logits]
    return [tensor for tensor in router_logits if tensor is not None]


def _resolved_model_revision(model: torch.nn.Module) -> str | None:
    revision = getattr(model.config, "_commit_hash", None)
    return str(revision) if revision else None


def _layer_tensor(logits: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    if logits.ndim == 3:
        return logits.detach().float().cpu()
    if logits.ndim == 2 and logits.shape[0] == batch_size * seq_len:
        return logits.detach().float().reshape(batch_size, seq_len, logits.shape[-1]).cpu()
    raise RuntimeError(f"unsupported router logit shape: {tuple(logits.shape)}")


def collect_trace(config: dict[str, Any], output_dir: Path) -> None:
    seed = int(config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_config = config["model"]
    trace_config = config.get("trace", {})
    model_id = model_config["id"]
    revision = model_config.get("revision")
    device = torch.device(model_config.get("device", "cuda"))
    dtype = _dtype(model_config.get("dtype", "bfloat16"))

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    prompts = _load_prompts(config)[: int(trace_config.get("max_sequences", 8))]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(trace_config.get("max_length", 256)),
    ).to(device)

    with torch.inference_mode():
        outputs = model(**encoded, output_router_logits=True, use_cache=False)

    input_ids = encoded["input_ids"].detach().cpu().numpy()
    attention_mask = encoded["attention_mask"].detach().cpu().numpy()
    batch_size, seq_len = input_ids.shape
    layers = [
        _layer_tensor(layer, batch_size, seq_len)
        for layer in _router_layers(outputs.router_logits)
    ]

    stacked_logits = torch.stack(layers, dim=1).numpy()
    probabilities = torch.softmax(torch.from_numpy(stacked_logits), dim=-1).numpy()
    top_k = int(getattr(model.config, "num_experts_per_tok", trace_config.get("top_k", 2)))
    selected = torch.topk(torch.from_numpy(stacked_logits), k=top_k, dim=-1).indices.numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "router_trace.npz",
        input_ids=input_ids,
        attention_mask=attention_mask,
        router_logits=stacked_logits,
        router_probabilities=probabilities,
        selected_experts=selected.astype(np.int64),
    )

    metadata = {
        "seed": seed,
        "model_id": model_id,
        "revision": revision,
        "resolved_revision": _resolved_model_revision(model),
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(device),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "num_prompts": len(prompts),
        "sequence_length": seq_len,
        "num_layers_with_router_logits": len(layers),
        "num_experts": int(stacked_logits.shape[-1]),
        "top_k": top_k,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect OLMoE router traces.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path(config.get("output_dir", "outputs/olmoe_trace_smoke"))
    collect_trace(config, output_dir)


if __name__ == "__main__":
    main()
