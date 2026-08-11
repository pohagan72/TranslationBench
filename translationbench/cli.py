"""CLI: score, fetch, translate, experiment."""

import argparse
import json
import sys

from . import corpora, metrics
from .report import write_reports
from .translators import make_translator


def cmd_score(args):
    sources = corpora.load_lines(args.source)
    references = corpora.load_lines(args.reference)
    candidates = corpora.load_lines(args.candidate)
    result = metrics.score(
        sources, candidates, references, use_comet=args.comet, gpus=args.gpus
    )
    print(json.dumps(result.to_dict(), indent=2))


def cmd_fetch(args):
    src, ref = corpora.fetch_wmt(args.testset, args.langpair, args.out_dir)
    print(f"Wrote {src}\nWrote {ref}")


def cmd_translate(args):
    segments = corpora.load_lines(args.input)
    translator = make_translator(args.engine, args.model)
    out = translator.translate(
        segments, args.source_lang, args.target_lang,
        document_context=(args.mode == "context"),
    )
    corpora.save_lines(args.out, out)
    print(f"Wrote {len(out)} segments to {args.out}")


def cmd_experiment(args):
    sources = corpora.load_lines(args.source)
    references = corpora.load_lines(args.reference)
    if len(sources) != len(references):
        sys.exit("Source and reference segment counts differ.")
    if args.limit:
        sources, references = sources[: args.limit], references[: args.limit]

    translator = make_translator(args.engine, args.model)
    runs = {}
    for mode, with_doc in (("sentence", False), ("context", True)):
        print(f"Translating {len(sources)} segments ({mode} mode)...", flush=True)
        candidates = translator.translate(
            sources, args.source_lang, args.target_lang, document_context=with_doc
        )
        corpora.save_lines(f"{args.out_dir}/candidate.{mode}.txt", candidates)
        print(f"Scoring ({mode})...", flush=True)
        runs[mode] = metrics.score(
            sources, candidates, references, use_comet=not args.no_comet, gpus=args.gpus
        )

    json_path, html_path = write_reports(
        args.out_dir, translator.label, len(sources), runs["sentence"], runs["context"]
    )
    print(f"Wrote {json_path}\nWrote {html_path}")
    for mode, s in runs.items():
        comet = f", COMET {s.comet:.4f}" if s.comet is not None else ""
        print(f"  {mode:>9}: BLEU {s.bleu:.2f}, chrF {s.chrf:.2f}{comet}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="translationbench",
        description="BLEU/COMET benchmark harness for machine translation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("score", help="Score an existing candidate translation")
    p.add_argument("--source", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--comet", action="store_true", help="Also compute COMET")
    p.add_argument("--gpus", type=int, default=0)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("fetch", help="Download a WMT test set via sacreBLEU")
    p.add_argument("--testset", default="wmt14")
    p.add_argument("--langpair", default="en-fr")
    p.add_argument("--out-dir", default="data")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("translate", help="Translate a source file with Claude")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["sentence", "context"], default="context")
    p.add_argument("--source-lang", required=True)
    p.add_argument("--target-lang", required=True)
    p.add_argument("--engine", choices=["gemini", "claude"], default="gemini")
    p.add_argument("--model", default=None,
                   help="Override the engine default (Gemini: $GEMINI_MODEL)")
    p.set_defaults(func=cmd_translate)

    p = sub.add_parser(
        "experiment",
        help="Run the with-context vs without-context comparison end to end",
    )
    p.add_argument("--source", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--source-lang", required=True)
    p.add_argument("--target-lang", required=True)
    p.add_argument("--out-dir", default="out")
    p.add_argument("--engine", choices=["gemini", "claude"], default="gemini")
    p.add_argument("--model", default=None,
                   help="Override the engine default (Gemini: $GEMINI_MODEL)")
    p.add_argument("--limit", type=int, help="Only use the first N segments")
    p.add_argument("--no-comet", action="store_true", help="Skip COMET (BLEU/chrF only)")
    p.add_argument("--gpus", type=int, default=0)
    p.set_defaults(func=cmd_experiment)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
