"""Smoke test for the Hansard loader. Stubs the `datasets` library so the test
doesn't touch the network or require the ~200 MB dataset download."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def _install_fake_datasets(rows: list[dict]) -> None:
    """Register a minimal fake `datasets` module in sys.modules."""
    fake = types.ModuleType("datasets")

    class FakeDataset:
        def __init__(self, data: list[dict]):
            self._data = list(data)

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

        def filter(self, fn):
            return FakeDataset([r for r in self._data if fn(r)])

        def shuffle(self, seed: int):
            import random as _random
            rng = _random.Random(seed)
            data = list(self._data)
            rng.shuffle(data)
            return FakeDataset(data)

        def select(self, indices):
            return FakeDataset([self._data[i] for i in indices])

    def load_dataset(name, split=None):
        return FakeDataset(rows)

    fake.load_dataset = load_dataset
    sys.modules["datasets"] = fake


def test_hansard_loader_writes_aligned_files(tmp_path: Path):
    _install_fake_datasets(
        [
            {"id": "1", "en": "The committee will meet on Tuesday.",
             "fr": "Le comité se réunira mardi."},
            {"id": "2",
             "en": "The report will be tabled in the House on Tuesday morning.",
             "fr": "Le rapport sera déposé à la Chambre mardi matin."},
            {"id": "3", "en": "", "fr": "Vide côté anglais."},  # dropped: empty en
            {"id": "4", "en": "Short.", "fr": "Court."},  # dropped: below min_len
            {"id": "5", "en": "This is a legitimately valid parliamentary "
                              "utterance for testing purposes.",
             "fr": "Ceci est une déclaration parlementaire légitime "
                   "à des fins de test."},
        ]
    )

    from translationbench import hansard

    src, ref = hansard.fetch_hansard(
        tmp_path, split="test", limit=10, seed=42, min_len=20, max_len=400
    )

    en_lines = src.read_text(encoding="utf-8").splitlines()
    fr_lines = ref.read_text(encoding="utf-8").splitlines()

    assert len(en_lines) == 3, f"expected 3 kept rows, got {len(en_lines)}"
    assert len(en_lines) == len(fr_lines), "misaligned output"
    for en, fr in zip(en_lines, fr_lines):
        assert en.strip() and fr.strip()
        assert "\n" not in en and "\n" not in fr


def test_hansard_loader_respects_limit(tmp_path: Path):
    _install_fake_datasets(
        [
            {"id": str(i),
             "en": f"Segment number {i} is a valid utterance for the record.",
             "fr": f"Le segment numéro {i} est une déclaration valide."}
            for i in range(20)
        ]
    )

    from translationbench import hansard

    src, ref = hansard.fetch_hansard(
        tmp_path, split="test", limit=5, seed=42, min_len=10, max_len=400
    )

    assert len(src.read_text(encoding="utf-8").splitlines()) == 5
    assert len(ref.read_text(encoding="utf-8").splitlines()) == 5
    assert "n5" in src.name and ".seed42" in src.name


def test_hansard_loader_raises_when_datasets_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    from translationbench import hansard

    with pytest.raises(RuntimeError, match='pip install "datasets'):
        hansard.fetch_hansard(tmp_path, limit=1)
