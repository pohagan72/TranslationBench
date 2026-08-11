"""Smoke tests for BLEU/chrF scoring (COMET excluded — model download too heavy for CI)."""

import pytest

from translationbench import metrics

SOURCES = ["The house is red.", "The committee will meet on Tuesday."]
REFERENCES = ["La maison est rouge.", "Le comité se réunira mardi."]


def test_perfect_candidate_scores_100():
    result = metrics.score(SOURCES, REFERENCES, REFERENCES)
    assert result.bleu == pytest.approx(100.0)
    assert result.comet is None


def test_worse_candidate_scores_lower():
    good = metrics.score(SOURCES, REFERENCES, REFERENCES)
    bad = metrics.score(
        SOURCES, ["La maison est bleue.", "Le comité se réunira peut-être."], REFERENCES
    )
    assert bad.bleu < good.bleu
    assert bad.chrf < good.chrf


def test_misaligned_corpus_rejected():
    with pytest.raises(ValueError, match="1:1 alignment"):
        metrics.score(SOURCES, REFERENCES[:1], REFERENCES)


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        metrics.score([], [], [])
