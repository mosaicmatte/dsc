"""Parameter budget accounting. The 4B ceiling is per TASK, summed over every
component that runs at inference time.

What counts
-----------
Everything loaded to produce a submission: word segmenter (if model-based),
bi-encoder, cross-encoder, generator. LoRA adapters do NOT reduce the base
model's count -- a LoRA-tuned Qwen2.5-3B is 3B, not 30M.

What does not count: sparse indexes (BM25 has no parameters), FAISS indexes,
tokenisers.

Check the budget BEFORE fine-tuning something for two days, not after.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

CEILING = 4_000_000_000

# TODO(BLOCKER/phase0-B0): verify EVERY number below against the model card
# before registering that model with BTC, and record the HF revision (commit SHA)
# you will actually use. These are planning figures typed from memory of the
# published cards -- ``count_hf()`` is the source of truth:
#     python -c "from src.params import count_hf; print(count_hf('BAAI/bge-m3'))"
# An under-count here does not fail loudly; it fails at submission time.
KNOWN: Dict[str, int] = {
    "bkai-foundation-models/vietnamese-bi-encoder": 135_000_000,   # PhoBERT-base
    "dangvantuan/vietnamese-embedding":             135_000_000,   # PhoBERT-base
    "AITeamVN/Vietnamese_Embedding":                568_000_000,   # BGE-M3 / XLM-R-large
    "BAAI/bge-m3":                                  568_000_000,
    "AITeamVN/Vietnamese_Reranker":                 568_000_000,
    "BAAI/bge-reranker-v2-m3":                      568_000_000,
    "itdainb/PhoRanker":                            135_000_000,
    "namdp-ptit/ViRanker":                          135_000_000,
    "Qwen/Qwen2.5-1.5B-Instruct":                 1_540_000_000,
    "Qwen/Qwen2.5-3B-Instruct":                   3_090_000_000,
    "Qwen/Qwen2.5-0.5B-Instruct":                   494_000_000,
}


def count_hf(model_name_or_path: str, trust_remote_code: bool = False) -> int:
    """Authoritative count: load the config and sum the actual parameters.

    Uses meta-device instantiation where available so this does not need the
    weights in RAM.
    """
    from transformers import AutoConfig, AutoModel  # type: ignore
    cfg = AutoConfig.from_pretrained(model_name_or_path,
                                     trust_remote_code=trust_remote_code)
    try:
        import torch
        from accelerate import init_empty_weights  # type: ignore
        with init_empty_weights():
            model = AutoModel.from_config(cfg, trust_remote_code=trust_remote_code)
        return sum(p.numel() for p in model.parameters())
    except ImportError:
        model = AutoModel.from_pretrained(model_name_or_path,
                                          trust_remote_code=trust_remote_code)
        return sum(p.numel() for p in model.parameters())


def lookup(name: str) -> int:
    if name in KNOWN:
        return KNOWN[name]
    try:
        return count_hf(name)
    except Exception as e:  # noqa: BLE001
        raise KeyError(
            f"{name!r} is not in KNOWN and could not be counted ({e}). "
            "Add it to src/params.KNOWN after checking the model card."
        ) from e


def budget(components: Iterable[Tuple[str, str]]) -> Dict:
    """``components`` = [(role, model_name), ...] -> a pass/fail budget report."""
    rows: List[Dict] = []
    total = 0
    for role, name in components:
        n = lookup(name)
        total += n
        rows.append({"role": role, "model": name, "params": n,
                     "params_b": round(n / 1e9, 3)})
    return {
        "components": rows,
        "total": total,
        "total_b": round(total / 1e9, 3),
        "ceiling_b": CEILING / 1e9,
        "headroom_b": round((CEILING - total) / 1e9, 3),
        "ok": total < CEILING,
    }


def report(components: Iterable[Tuple[str, str]]) -> str:
    b = budget(components)
    lines = [f"{'role':<12} {'model':<48} {'params':>8}"]
    lines.append("-" * 70)
    for r in b["components"]:
        lines.append(f"{r['role']:<12} {r['model']:<48} {r['params_b']:>7.3f}B")
    lines.append("-" * 70)
    lines.append(f"{'TOTAL':<12} {'':<48} {b['total_b']:>7.3f}B")
    lines.append(f"{'CEILING':<12} {'':<48} {b['ceiling_b']:>7.3f}B")
    lines.append(f"{'HEADROOM':<12} {'':<48} {b['headroom_b']:>7.3f}B")
    lines.append("STATUS: " + ("OK" if b["ok"] else "*** OVER BUDGET ***"))
    return "\n".join(lines)


if __name__ == "__main__":
    print("Task 1 -- retrieve + rerank")
    print(report([("bi-encoder", "AITeamVN/Vietnamese_Embedding"),
                  ("reranker", "AITeamVN/Vietnamese_Reranker")]))
    print("\nTask 2 -- retrieve + rerank + read")
    print(report([("bi-encoder", "AITeamVN/Vietnamese_Embedding"),
                  ("reranker", "AITeamVN/Vietnamese_Reranker"),
                  ("generator", "Qwen/Qwen2.5-1.5B-Instruct")]))
