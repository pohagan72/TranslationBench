"""Per-experiment usage and cost tracking.

Every Gemini API response ships a `usage_metadata` block with token counts.
We accumulate those across all calls in a run, apply Google's published
per-model price table, and write both the raw counts and an estimated USD
cost to the experiment's output folder.

Prices are point-in-time from ai.google.dev and MUST be updated when Google
changes them. Missing model prices produce a "cost: unknown" report rather
than a crash — the token counts are still recorded.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path


# USD per 1M tokens, indexed by model id. Cached input pricing applies to
# tokens served from an explicit context cache — Gemini prices these at
# ~25% of standard input on the Flash tier at time of writing.
# Source: https://ai.google.dev/pricing (2026-08).
PRICE_TABLE: dict[str, dict[str, float]] = {
    "gemini-3.6-flash": {
        "input_per_1m": 1.50,
        "cached_input_per_1m": 0.375,
        "output_per_1m": 7.50,
    },
    "gemini-3.5-flash-lite": {
        "input_per_1m": 0.30,
        "cached_input_per_1m": 0.075,
        "output_per_1m": 2.50,
    },
    # Legacy: kept only so historical runs on the deprecated model still price.
    "gemini-1.5-flash-latest": {
        "input_per_1m": 0.075,
        "cached_input_per_1m": 0.01875,
        "output_per_1m": 0.30,
    },
}


@dataclass
class UsageCounts:
    """Running totals for one run (all API calls, all modes, one translator).

    Fields correspond 1:1 to Gemini's `usage_metadata` attributes:
      - prompt_tokens: input tokens Gemini processed
      - cached_tokens: subset of prompt_tokens served from cache (billed lower)
      - output_tokens: tokens Gemini generated in the response
      - thinking_tokens: tokens consumed by internal reasoning (thinking models)
    """

    calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    def merge(self, other: "UsageCounts") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.cached_tokens += other.cached_tokens
        self.output_tokens += other.output_tokens
        self.thinking_tokens += other.thinking_tokens


class UsageTracker:
    """Thread-safe accumulator for a translator's API usage over one run.

    Instantiate one per experiment (or per mode within an experiment) and
    call `.record(response.usage_metadata)` after every API response. Use
    `.snapshot()` to materialize counts + estimated cost for reporting.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._counts = UsageCounts()
        self._lock = threading.Lock()

    def record(self, usage_metadata) -> None:
        """Accept whatever the SDK returned; treat all fields as optional."""
        if usage_metadata is None:
            return
        prompt = _get(usage_metadata, "prompt_token_count") or 0
        cached = _get(usage_metadata, "cached_content_token_count") or 0
        output = _get(usage_metadata, "candidates_token_count") or 0
        thinking = _get(usage_metadata, "thoughts_token_count") or 0
        with self._lock:
            self._counts.calls += 1
            self._counts.prompt_tokens += prompt
            self._counts.cached_tokens += cached
            self._counts.output_tokens += output
            self._counts.thinking_tokens += thinking

    def snapshot(self) -> dict:
        with self._lock:
            counts = UsageCounts(**asdict(self._counts))
        return {
            "model": self.model_name,
            "counts": asdict(counts),
            "estimated_cost": estimate_cost(self.model_name, counts),
        }


def _get(obj, name: str):
    """usage_metadata is an SDK object with attributes, but also None-safe."""
    if obj is None:
        return None
    return getattr(obj, name, None)


def estimate_cost(model_name: str, counts: UsageCounts) -> dict:
    """USD estimate. Cached tokens are billed at cached_input_per_1m; all other
    prompt tokens at input_per_1m; output+thinking tokens at output_per_1m.
    Thinking tokens follow Gemini's billing: charged as output.
    """
    prices = PRICE_TABLE.get(model_name)
    if prices is None:
        return {
            "model_priced": False,
            "note": f"No price entry for {model_name!r}; update PRICE_TABLE.",
        }

    non_cached_input = max(0, counts.prompt_tokens - counts.cached_tokens)
    output_including_thinking = counts.output_tokens + counts.thinking_tokens

    cost = (
        non_cached_input * prices["input_per_1m"] / 1_000_000
        + counts.cached_tokens * prices["cached_input_per_1m"] / 1_000_000
        + output_including_thinking * prices["output_per_1m"] / 1_000_000
    )
    return {
        "model_priced": True,
        "usd": round(cost, 6),
        "breakdown": {
            "non_cached_input_tokens": non_cached_input,
            "cached_input_tokens": counts.cached_tokens,
            "output_and_thinking_tokens": output_including_thinking,
        },
        "prices_used_per_1m_usd": prices,
    }


def write_usage_report(out_dir: str | Path, snapshots: dict[str, dict]) -> Path:
    """Write per-mode usage + a total to `<out_dir>/usage.json`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_counts = UsageCounts()
    total_usd = 0.0
    all_priced = True
    for snap in snapshots.values():
        total_counts.merge(UsageCounts(**snap["counts"]))
        cost = snap["estimated_cost"]
        if cost.get("model_priced"):
            total_usd += cost["usd"]
        else:
            all_priced = False

    payload = {
        "per_mode": snapshots,
        "total": {
            "counts": asdict(total_counts),
            "estimated_cost_usd": round(total_usd, 6) if all_priced else None,
            "priced": all_priced,
        },
    }
    path = out_dir / "usage.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def format_summary(snapshot: dict) -> str:
    """One-line human-readable summary for stdout."""
    counts = snapshot["counts"]
    cost = snapshot["estimated_cost"]
    if cost.get("model_priced"):
        cost_str = f"${cost['usd']:.4f}"
    else:
        cost_str = "$? (model not priced)"
    return (
        f"{counts['calls']} calls, "
        f"{counts['prompt_tokens']:,} in "
        f"({counts['cached_tokens']:,} cached), "
        f"{counts['output_tokens']:,} out, "
        f"{counts['thinking_tokens']:,} think, "
        f"cost {cost_str}"
    )
