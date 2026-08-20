"""Run configuration: load, hash, freeze.

Every script takes ``--config configs/something.yaml``. On execution the
*resolved* config (defaults filled in, CLI overrides applied) is written to
``work/configs/<run_id>.yaml``. That frozen copy, not the hand-edited source, is what
reproduces the run.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Any, Dict

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "work", "configs")


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text) or {}
        except ImportError as e:
            raise ImportError("pip install PyYAML, or use a .json config") from e
    return json.loads(text)


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """``--set retriever.k1=1.5 cutoff.alpha=0.9`` -> nested assignment."""
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value: {ov!r}")
        key, raw = ov.split("=", 1)
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


def fingerprint(cfg: Dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def make_run_id(cfg: Dict[str, Any], prefix: str = "run") -> str:
    """``run-0824-a1b2c3d4`` -- date for humans, hash for uniqueness.

    Identical configs produce identical run_ids, which is a feature: it makes an
    accidental duplicate run obvious instead of silently doubling the log.
    """
    return f"{prefix}-{_dt.date.today():%m%d}-{fingerprint(cfg)}"


def freeze(cfg: Dict[str, Any], run_id: str, out_dir: str = CONFIG_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_id}.yaml")
    try:
        import yaml  # type: ignore
        text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=True)
    except ImportError:
        path = path.replace(".yaml", ".json")
        text = json.dumps(cfg, indent=2, ensure_ascii=False, default=str)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def set_seed(seed: int = 42) -> None:
    """Seed everything reachable. Required for the reproduction package."""
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
