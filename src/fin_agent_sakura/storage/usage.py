"""Local LLM usage and rough cost estimation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_USAGE_PATH = Path("data/processed/llm_usage_log.jsonl")


@dataclass(frozen=True, slots=True)
class LLMUsageRecord:
    created_at: str
    feature: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for cost-control UI when provider usage is absent."""

    if not text:
        return 0
    return max(1, int(len(text) / 3.5))


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Very rough local estimate; custom gateway pricing may differ."""

    model_key = model.lower()
    if "gpt-5" in model_key:
        input_per_million = 5.0
        output_per_million = 15.0
    else:
        input_per_million = 1.0
        output_per_million = 3.0
    return prompt_tokens / 1_000_000 * input_per_million + completion_tokens / 1_000_000 * output_per_million


def record_llm_usage(
    *,
    feature: str,
    model: str,
    prompt_text: str = "",
    completion_text: str = "",
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
    path: str | Path = DEFAULT_USAGE_PATH,
) -> LLMUsageRecord:
    prompt_count = int(prompt_tokens if prompt_tokens is not None else estimate_tokens(prompt_text))
    completion_count = int(completion_tokens if completion_tokens is not None else estimate_tokens(completion_text))
    record = LLMUsageRecord(
        created_at=pd.Timestamp.now().isoformat(),
        feature=feature,
        model=model,
        prompt_tokens=prompt_count,
        completion_tokens=completion_count,
        total_tokens=prompt_count + completion_count,
        estimated_cost_usd=estimate_cost_usd(model, prompt_count, completion_count),
        metadata=metadata or {},
    )
    usage_path = Path(path)
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    with usage_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n")
    return record


def load_llm_usage(path: str | Path = DEFAULT_USAGE_PATH) -> pd.DataFrame:
    usage_path = Path(path)
    if not usage_path.exists():
        return pd.DataFrame(
            columns=[
                "created_at",
                "feature",
                "model",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "estimated_cost_usd",
                "metadata",
            ]
        )
    rows = []
    with usage_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarize_llm_usage(path: str | Path = DEFAULT_USAGE_PATH) -> dict[str, Any]:
    frame = load_llm_usage(path)
    if frame.empty:
        return {"calls": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
    return {
        "calls": int(len(frame)),
        "total_tokens": int(pd.to_numeric(frame["total_tokens"], errors="coerce").fillna(0).sum()),
        "estimated_cost_usd": float(pd.to_numeric(frame["estimated_cost_usd"], errors="coerce").fillna(0).sum()),
    }
