"""Merge external LoRA adapters (PEFT target_parameters on fused OLMoE) into full models.

Run on a machine with peft>=0.19 available. Saves merged bf16 models + tokenizer.
Usage: python scripts/merge_workingset_adapters.py <out_root>
"""
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "allenai/OLMoE-1B-7B-0125-Instruct"
ADAPTERS = [os.environ["MOE_SFT_ADAPTER"], os.environ["MOE_PC_ADAPTER"]]

out_root = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(BASE)
for adapter in ADAPTERS:
    name = adapter.split("/")[-1]
    print(f"[merge] {adapter} onto {BASE}")
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out = out_root / name
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"[merge] saved {out}")
    del model, merged
print("[merge] done")
