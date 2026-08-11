# TranslationBench

Benchmark harness for machine translation quality. Scores candidate translations
with **BLEU** (sacreBLEU) and **COMET** (Unbabel `wmt22-comet-da`), and runs the
experiment this repo exists for: **does document-level context measurably improve
LLM translation quality over sentence-by-sentence translation?**

Built by [Red Maple Research](https://redmapleresearch.ca). Apache 2.0.

## Why

Traditional CAT (computer-assisted translation) workflows translate segment by
segment — each sentence in isolation. LLMs can read the whole document before
translating any sentence of it. This harness measures that difference on
sentence-aligned parallel corpora, keeping a strict 1:1 segment mapping so
standard corpus metrics apply.

## Quickstart

```bash
# Python 3.11 or 3.12 recommended (COMET depends on torch; check torch support
# before using a newer interpreter). BLEU-only scoring works anywhere.
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install "unbabel-comet>=2.2"               # optional, needed for COMET scores

# Score an existing candidate translation
python -m translationbench score \
  --source data/src.en.txt --reference data/ref.fr.txt \
  --candidate out/candidate.fr.txt --comet

# Fetch Canadian Hansard EN-FR pairs (parliamentary; government register)
python -m translationbench hansard --out-dir data/ --limit 500

# Or a generic WMT test set via sacreBLEU
python -m translationbench fetch --testset wmt14 --langpair en-fr --out-dir data/

# Run the full with-context vs without-context experiment
# (needs GOOGLE_API_KEY; optionally GEMINI_MODEL to pin the production model)
python -m translationbench experiment \
  --source data/src.en.txt --reference data/ref.fr.txt \
  --source-lang English --target-lang "Canadian French" \
  --out-dir out/
```

All text files are one segment per line, UTF-8, aligned by line number.

## The experiment

`experiment` translates the same source segments twice with the same model:

1. **Sentence mode** — each segment translated in isolation (the CAT-tool baseline)
2. **Document mode** — segments translated with the full document provided as context

Both modes emit exactly one output segment per input segment, so BLEU and COMET
compare like with like. The report (JSON + HTML) shows corpus-level scores for
both modes plus per-segment COMET deltas.

## Corpora

Any sentence-aligned parallel text works. The `hansard` subcommand fetches
sentence-aligned **Canadian Hansard** EN-FR pairs (parliamentary proceedings;
formal government register — as close to CMHC's housing/policy domain as
freely-available public data gets) from Hugging Face
([`raeidsaqur/Hansard`](https://huggingface.co/datasets/raeidsaqur/Hansard);
first run downloads ~200 MB). Requires `pip install "datasets>=2.14"`. WMT
test sets are available via `fetch` as a generic fallback. Client-provided
translation memories (TMX) can be converted to aligned text files — segments
with 1:1 alignment only.

## Notes

- **Engines.** The default engine is Gemini (`--engine gemini`), configured
  exactly like the [Synzo](https://synzo.ai) production pipeline: same SDK,
  same `GOOGLE_API_KEY` / `GEMINI_MODEL` environment variables, and — in
  sentence mode — the verbatim production per-segment prompt, so the baseline
  measures the real pipeline, not an approximation of it. `--engine claude`
  (Anthropic API, structured outputs, server-side refusal fallbacks) is
  available to show the document-context effect holds across vendors. Other
  engines: implement `Translator` in `translationbench/translators.py`, or
  score their output files directly with `score` (no API key needed).
- COMET's `wmt22-comet-da` model is a ~1.5 GB one-time download; it runs on CPU
  (slow) or GPU (fast).
- This tool evaluates translators; it is not one. It deliberately lives outside
  any translation product so results are independently reproducible.
