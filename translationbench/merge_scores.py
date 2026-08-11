"""Merge standalone score.*.json files back into an experiment's report.json.

Rationale: `experiment` writes report.json with BLEU/chrF; if `score --comet`
is run separately later (e.g. because COMET's model download required a
manual transfer), those semantic scores land in score.*.json files. This
merges them so report.json is the single source of truth for the RFI.

Usage:
    python -m translationbench.merge_scores <experiment_dir>

Looks for score.candidate.sentence.json → sentence_level.comet(_segments)
       and score.candidate.context.json  → document_context.comet(_segments)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .metrics import Scores
from .report import write_reports

MODE_TO_KEY = {
    "sentence": "sentence_level",
    "context": "document_context",
    "whole": "whole_document",
}


def _to_scores(payload: dict) -> Scores:
    return Scores(
        bleu=payload["bleu"],
        chrf=payload["chrf"],
        bleu_signature=payload["bleu_signature"],
        comet=payload.get("comet"),
        comet_segments=payload.get("comet_segments") or [],
    )


def merge(experiment_dir: str | Path) -> Path:
    d = Path(experiment_dir)
    report_path = d / "report.json"
    if not report_path.is_file():
        raise SystemExit(f"No report.json in {d}")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    merged = 0
    for mode, report_key in MODE_TO_KEY.items():
        score_path = d / f"score.candidate.{mode}.json"
        if not score_path.is_file():
            continue
        if report_key not in report:
            continue
        score = json.loads(score_path.read_text(encoding="utf-8"))
        for field in ("comet", "comet_segments"):
            if field in score:
                report[report_key][field] = score[field]
        merged += 1
        print(f"  merged {mode} COMET from {score_path.name}")

    if not merged:
        print(f"No score.candidate.*.json files found next to {report_path}")
        return report_path

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Updated {report_path}")

    # Regenerate the HTML view so the COMET row is populated.
    if {"sentence_level", "document_context"} <= set(report):
        write_reports(
            d,
            report.get("model", "unknown"),
            report.get("segments", 0),
            _to_scores(report["sentence_level"]),
            _to_scores(report["document_context"]),
        )
        print(f"Regenerated {d/'report.html'}")

    return report_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <experiment_dir>", file=sys.stderr)
        sys.exit(2)
    merge(sys.argv[1])
