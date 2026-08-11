"""Unit tests for token usage tracking and cost estimation.

The pricing math is small, but every experiment reports a cost figure that
will end up in the RFI Section 5.6 — so it's worth verifying.
"""

import json
from pathlib import Path

import pytest

from translationbench.usage import (
    PRICE_TABLE,
    UsageCounts,
    UsageTracker,
    estimate_cost,
    format_summary,
    write_usage_report,
)


class _FakeUsageMetadata:
    """Stub for the SDK's usage_metadata object; only attribute access matters."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_tracker_accumulates_across_calls():
    tracker = UsageTracker("gemini-3.6-flash")
    tracker.record(_FakeUsageMetadata(
        prompt_token_count=100, candidates_token_count=50,
        cached_content_token_count=0, thoughts_token_count=0,
    ))
    tracker.record(_FakeUsageMetadata(
        prompt_token_count=200, candidates_token_count=80,
        cached_content_token_count=50, thoughts_token_count=25,
    ))
    snap = tracker.snapshot()
    assert snap["counts"]["calls"] == 2
    assert snap["counts"]["prompt_tokens"] == 300
    assert snap["counts"]["cached_tokens"] == 50
    assert snap["counts"]["output_tokens"] == 130
    assert snap["counts"]["thinking_tokens"] == 25


def test_tracker_tolerates_missing_or_none_metadata():
    tracker = UsageTracker("gemini-3.6-flash")
    tracker.record(None)  # e.g. a failed call before metadata was set
    tracker.record(_FakeUsageMetadata(prompt_token_count=10))  # partial fields
    snap = tracker.snapshot()
    assert snap["counts"]["calls"] == 1
    assert snap["counts"]["prompt_tokens"] == 10
    assert snap["counts"]["output_tokens"] == 0


def test_cost_math_flash_no_cache_no_thinking():
    # 1M input, 1M output on 3.6-flash → $1.50 + $7.50 = $9.00 exactly
    counts = UsageCounts(prompt_tokens=1_000_000, output_tokens=1_000_000)
    cost = estimate_cost("gemini-3.6-flash", counts)
    assert cost["model_priced"] is True
    assert cost["usd"] == 9.0
    assert cost["breakdown"]["non_cached_input_tokens"] == 1_000_000
    assert cost["breakdown"]["cached_input_tokens"] == 0
    assert cost["breakdown"]["output_and_thinking_tokens"] == 1_000_000


def test_cost_math_cached_input_discount():
    # 1M cached input on 3.6-flash → $0.375, not $1.50.
    counts = UsageCounts(prompt_tokens=1_000_000, cached_tokens=1_000_000)
    cost = estimate_cost("gemini-3.6-flash", counts)
    assert cost["usd"] == pytest.approx(0.375)
    assert cost["breakdown"]["non_cached_input_tokens"] == 0
    assert cost["breakdown"]["cached_input_tokens"] == 1_000_000


def test_cost_math_thinking_tokens_billed_as_output():
    # 1M thinking tokens on 3.6-flash → $7.50 (output rate), no input cost.
    counts = UsageCounts(thinking_tokens=1_000_000)
    cost = estimate_cost("gemini-3.6-flash", counts)
    assert cost["usd"] == pytest.approx(7.5)
    assert cost["breakdown"]["output_and_thinking_tokens"] == 1_000_000


def test_cost_math_mixed():
    # A realistic-ish mix: 12k input (2k cached), 1500 output on flash-lite.
    counts = UsageCounts(
        calls=2, prompt_tokens=12_000, cached_tokens=2_000, output_tokens=1_500,
    )
    cost = estimate_cost("gemini-3.5-flash-lite", counts)
    expected = (
        10_000 * 0.30 / 1_000_000  # non-cached input
        + 2_000 * 0.075 / 1_000_000  # cached input
        + 1_500 * 2.50 / 1_000_000  # output
    )
    assert cost["usd"] == pytest.approx(round(expected, 6))


def test_cost_math_missing_model_does_not_crash():
    counts = UsageCounts(prompt_tokens=100)
    cost = estimate_cost("gemini-9-not-a-real-model", counts)
    assert cost["model_priced"] is False
    assert "note" in cost


def test_write_usage_report_aggregates_and_persists(tmp_path: Path):
    snapshots = {
        "sentence": {
            "model": "gemini-3.6-flash",
            "counts": {"calls": 50, "prompt_tokens": 4500,
                       "cached_tokens": 0, "output_tokens": 4500,
                       "thinking_tokens": 0},
            "estimated_cost": estimate_cost(
                "gemini-3.6-flash",
                UsageCounts(calls=50, prompt_tokens=4500, output_tokens=4500),
            ),
        },
        "context": {
            "model": "gemini-3.6-flash",
            "counts": {"calls": 2, "prompt_tokens": 24000,
                       "cached_tokens": 0, "output_tokens": 3000,
                       "thinking_tokens": 500},
            "estimated_cost": estimate_cost(
                "gemini-3.6-flash",
                UsageCounts(calls=2, prompt_tokens=24000,
                            output_tokens=3000, thinking_tokens=500),
            ),
        },
    }
    path = write_usage_report(tmp_path, snapshots)
    payload = json.loads(path.read_text())

    assert payload["per_mode"]["sentence"]["counts"]["calls"] == 50
    assert payload["total"]["counts"]["calls"] == 52
    assert payload["total"]["counts"]["prompt_tokens"] == 28_500
    # Total should equal sum of the two per-mode costs (both priced).
    sent_usd = snapshots["sentence"]["estimated_cost"]["usd"]
    ctx_usd = snapshots["context"]["estimated_cost"]["usd"]
    assert payload["total"]["estimated_cost_usd"] == pytest.approx(
        round(sent_usd + ctx_usd, 6)
    )
    assert payload["total"]["priced"] is True


def test_format_summary_readable():
    snap = {
        "counts": {"calls": 2, "prompt_tokens": 24_000, "cached_tokens": 8_000,
                   "output_tokens": 3_000, "thinking_tokens": 500},
        "estimated_cost": {"model_priced": True, "usd": 0.0413},
    }
    s = format_summary(snap)
    assert "2 calls" in s
    assert "24,000" in s and "8,000" in s and "500 think" in s
    assert "$0.0413" in s


def test_price_table_has_expected_models():
    # Guard-rail: if we bump the default model, keep the price entry in sync.
    for model in ("gemini-3.6-flash", "gemini-3.5-flash-lite"):
        assert model in PRICE_TABLE
        for field in ("input_per_1m", "cached_input_per_1m", "output_per_1m"):
            assert field in PRICE_TABLE[model]
            assert PRICE_TABLE[model][field] > 0
