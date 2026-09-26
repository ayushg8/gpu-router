"""VRAM / runtime estimates for routing when the spec leaves them out (phase 2).

Deliberately simple and explainable: every estimate carries `reasons`, and `source` says
whether it came from the spec or from these heuristics (spec: "always labels which"). The
phase-5 scoring router reads `manifest.json["estimate"]`; the phase-1 router only filters on
explicit `spec.vram_gb`, so a wrong guess here can never block a job today.

Heuristics, from the entrypoint source plus the other bundled .py files (capped):
  - model size from names like `Llama-3-8B`, `mistral-7b`, `qwen2.5-0.5b` (largest wins)
  - precision from `load_in_4bit` / `bnb_4bit` (0.5 B/param), `load_in_8bit` (1), else fp16 (2)
  - mode: LoRA/PEFT fine-tune, full training (`.backward(`, `Trainer(`, optimizers), or
    inference
  - VRAM: inference = P*b*1.2 + 1; LoRA = P*b*1.3 + 2; full training = P*16 (weights, grads
    and Adam state); unknown model size = DEFAULT_VRAM_GB (fits a free T4)
  - hours: DEFAULT_HOURS, 3 h for LoRA of a >= 1B model, 6 h for full training of one
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gpu_router.models import JobSpec
    from gpu_router.packaging.files import ProjectFile

DEFAULT_VRAM_GB = 8.0
DEFAULT_HOURS = 1.0
MAX_VRAM_GB = 80.0
MAX_SCAN_BYTES = 512 * 1024

_SIZE_RE = re.compile(r"[-_/](\d{1,3}(?:\.\d{1,2})?)[bB](?=[-_\"'\s/)]|$)", re.MULTILINE)
_LORA_RE = re.compile(r"\b(peft|LoraConfig|get_peft_model|lora_r|qlora)\b", re.IGNORECASE)
_4BIT_RE = re.compile(r"load_in_4bit|bnb_4bit|4bit", re.IGNORECASE)
_8BIT_RE = re.compile(r"load_in_8bit|8bit", re.IGNORECASE)
_TRAIN_RE = re.compile(
    r"\.backward\(|\bTrainer\(|SFTTrainer|optim\.(Adam|AdamW|SGD)|\.fit\(|optimizer\.step\("
)


@dataclass(frozen=True, slots=True)
class Estimate:
    vram_gb: float | None
    hours: float | None
    vram_source: Literal["spec", "heuristic"]
    hours_source: Literal["spec", "heuristic"]
    model_params_b: float | None = None
    mode: Literal["inference", "lora", "train", "unknown"] = "unknown"
    reasons: list[str] = field(default_factory=list)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "vram_gb": self.vram_gb,
            "hours": self.hours,
            "vram_source": self.vram_source,
            "hours_source": self.hours_source,
            "model_params_b": self.model_params_b,
            "mode": self.mode,
            "reasons": list(self.reasons),
        }


def _scan_text(files: Sequence[ProjectFile], entry: str | None) -> str:
    """Entrypoint first, then other .py files, up to MAX_SCAN_BYTES in total."""
    ordered = sorted(
        (f for f in files if f.rel.endswith(".py") or f.rel == entry),
        key=lambda f: (f.rel != entry, f.rel),
    )
    chunks: list[str] = []
    budget = MAX_SCAN_BYTES
    for f in ordered:
        if budget <= 0:
            break
        try:
            with f.path.open("rb") as fh:
                data = fh.read(budget)
        except OSError:
            continue
        budget -= len(data)
        chunks.append(data.decode("utf-8", "replace"))
    return "\n".join(chunks)


def _ceil_gb(x: float) -> float:
    return float(min(MAX_VRAM_GB, max(1.0, math.ceil(x))))


def estimate(spec: JobSpec, files: Sequence[ProjectFile]) -> Estimate:
    reasons: list[str] = []
    text = _scan_text(files, spec.script)
    sizes = [float(m.group(1)) for m in _SIZE_RE.finditer(text)]
    sizes = [s for s in sizes if 0.05 <= s <= 500]
    params = max(sizes) if sizes else None
    lora = bool(_LORA_RE.search(text))
    train = bool(_TRAIN_RE.search(text))
    mode: Literal["inference", "lora", "train", "unknown"]
    if lora:
        mode = "lora"
    elif train:
        mode = "train"
    elif params is not None:
        mode = "inference"
    else:
        mode = "unknown"

    Source = Literal["spec", "heuristic"]
    vram: float | None
    vram_src: Source
    hours_src: Source
    if spec.vram_gb is not None:
        vram, vram_src = spec.vram_gb, "spec"
    elif params is None:
        vram, vram_src = DEFAULT_VRAM_GB, "heuristic"
        reasons.append(f"no model size found; default {DEFAULT_VRAM_GB:g} GB (fits a free T4)")
    else:
        vram_src = "heuristic"
        if _4BIT_RE.search(text):
            bpp, prec = 0.5, "4-bit"
        elif _8BIT_RE.search(text):
            bpp, prec = 1.0, "8-bit"
        else:
            bpp, prec = 2.0, "fp16"
        if mode == "lora":
            raw = params * bpp * 1.3 + 2
        elif mode == "train":
            raw = params * 16
            prec = "fp16 + Adam state"
        else:
            raw = params * bpp * 1.2 + 1
        vram = _ceil_gb(raw)
        reasons.append(f"{params:g}B-parameter model, {mode}, {prec}: about {raw:.1f} GB")
        if raw > MAX_VRAM_GB:
            reasons.append(f"capped at {MAX_VRAM_GB:g} GB; likely too big for free GPUs")

    hours: float | None
    if spec.hours is not None:
        hours, hours_src = spec.hours, "spec"
    else:
        hours_src = "heuristic"
        if params is not None and params >= 1 and mode == "train":
            hours = 6.0
        elif params is not None and params >= 1 and mode == "lora":
            hours = 3.0
        else:
            hours = DEFAULT_HOURS
        reasons.append(f"runtime guess {hours:g} h (set `hours:` to be exact)")

    return Estimate(
        vram_gb=vram,
        hours=hours,
        vram_source=vram_src,
        hours_source=hours_src,
        model_params_b=params,
        mode=mode,
        reasons=reasons,
    )
