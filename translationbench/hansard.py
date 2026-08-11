"""Fetch sentence-aligned Canadian Hansard EN-FR pairs from Hugging Face.

Source: raeidsaqur/Hansard on Hugging Face (parliamentary proceedings; formal
government register — as close to CMHC's domain as freely-available public data
gets). Emits two line-aligned UTF-8 text files ready for the benchmark harness.
"""

from __future__ import annotations

import random
from pathlib import Path

from .corpora import save_lines

DATASET_ID = "raeidsaqur/Hansard"


def _sample_rows(
    rows: list[dict],
    limit: int | None,
    seed: int | None,
    min_len: int,
    max_len: int,
) -> tuple[list[str], list[str]]:
    """Filter, shuffle, and slice rows into two aligned string lists.

    Kept separate from any I/O so the same behavior applies to both the
    Hugging Face loader path and the local-parquet path.
    """
    kept = [
        r for r in rows
        if (en := (r.get("en") or "").strip()) and (fr := (r.get("fr") or "").strip())
        and min_len <= len(en) <= max_len
        and min_len <= len(fr) <= max_len
    ]
    if seed is not None:
        random.Random(seed).shuffle(kept)
    if limit is not None:
        kept = kept[:limit]
    return (
        [r["en"].strip().replace("\n", " ") for r in kept],
        [r["fr"].strip().replace("\n", " ") for r in kept],
    )


def _load_pairs(
    split: str,
    limit: int | None,
    seed: int | None,
    min_len: int,
    max_len: int,
) -> tuple[list[str], list[str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            'The Hansard loader needs the "datasets" package. '
            'Run: pip install "datasets>=2.14"'
        ) from exc

    ds = load_dataset(DATASET_ID, split=split)
    return _sample_rows(list(ds), limit, seed, min_len, max_len)


def _load_pairs_from_parquet(
    parquet_path: str | Path,
    limit: int | None,
    seed: int | None,
    min_len: int,
    max_len: int,
) -> tuple[list[str], list[str]]:
    """Load rows from a locally-downloaded Hansard parquet file.

    Useful when the machine can't reach Hugging Face's file storage. The
    parquet file was downloaded manually from
    https://huggingface.co/datasets/raeidsaqur/Hansard/tree/main
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            'Loading from a local parquet file needs "pandas" installed. '
            'Run: pip install pandas pyarrow'
        ) from exc

    df = pd.read_parquet(parquet_path)
    missing = {"en", "fr"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Parquet at {parquet_path} is missing expected columns: "
            f"{sorted(missing)}. Found: {list(df.columns)}"
        )
    rows = df[["en", "fr"]].to_dict(orient="records")
    return _sample_rows(rows, limit, seed, min_len, max_len)


def fetch_hansard(
    out_dir: str | Path,
    split: str = "test",
    limit: int | None = 500,
    seed: int | None = 42,
    min_len: int = 20,
    max_len: int = 400,
    from_parquet: str | Path | None = None,
) -> tuple[Path, Path]:
    """Fetch a sample of sentence-aligned Hansard EN-FR pairs.

    Defaults: 500 randomly-sampled test-split segments, deterministic
    (seed=42), 20-400 chars per segment (drops single-word ceremonial
    lines and one-line motions with attached transcripts).

    If from_parquet is set, reads rows from that local parquet file
    instead of downloading from Hugging Face (useful on networks that
    block us.aws.cdn.hf.co).
    """
    if from_parquet is not None:
        en_lines, fr_lines = _load_pairs_from_parquet(
            from_parquet, limit, seed, min_len, max_len
        )
        source_tag = Path(from_parquet).stem
    else:
        en_lines, fr_lines = _load_pairs(split, limit, seed, min_len, max_len)
        source_tag = split
    if len(en_lines) != len(fr_lines):
        raise RuntimeError(
            f"Loader produced misaligned output: {len(en_lines)} EN vs "
            f"{len(fr_lines)} FR."
        )

    out_dir = Path(out_dir)
    tag = f"hansard.{source_tag}.n{len(en_lines)}"
    if seed is not None:
        tag += f".seed{seed}"
    src_path = out_dir / f"{tag}.en.txt"
    ref_path = out_dir / f"{tag}.fr.txt"
    save_lines(src_path, en_lines)
    save_lines(ref_path, fr_lines)
    return src_path, ref_path
