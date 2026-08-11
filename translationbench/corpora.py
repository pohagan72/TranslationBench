"""Load aligned corpora: plain text files (one segment per line) and WMT test sets."""

import subprocess
from pathlib import Path


def load_lines(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    lines = [line.rstrip("\n").rstrip("\r") for line in text.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def save_lines(path: str | Path, lines: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def fetch_wmt(testset: str, langpair: str, out_dir: str | Path) -> tuple[Path, Path]:
    """Download a sacreBLEU test set (e.g. wmt14, en-fr) as aligned src/ref files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_lang, tgt_lang = langpair.split("-")
    src_path = out_dir / f"{testset}.{langpair}.src.{src_lang}.txt"
    ref_path = out_dir / f"{testset}.{langpair}.ref.{tgt_lang}.txt"

    for echo, path in (("src", src_path), ("ref", ref_path)):
        result = subprocess.run(
            ["sacrebleu", "-t", testset, "-l", langpair, "--echo", echo],
            capture_output=True, text=True, encoding="utf-8", check=True,
        )
        path.write_text(result.stdout, encoding="utf-8")

    return src_path, ref_path
