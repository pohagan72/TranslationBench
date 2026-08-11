"""Corpus-level BLEU/chrF (sacreBLEU) and COMET scoring."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import sacrebleu

COMET_MODEL = "Unbabel/wmt22-comet-da"
# Point at a locally-downloaded checkpoint (folder containing model.ckpt +
# hparams.yaml) via TRANSLATIONBENCH_COMET_MODEL when the network can't
# reach Hugging Face's file storage.
COMET_MODEL_ENV = "TRANSLATIONBENCH_COMET_MODEL"


@dataclass
class Scores:
    bleu: float
    chrf: float
    bleu_signature: str
    comet: float | None = None
    comet_segments: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bleu": round(self.bleu, 2),
            "chrf": round(self.chrf, 2),
            "bleu_signature": self.bleu_signature,
            "comet": round(self.comet, 4) if self.comet is not None else None,
            "comet_segments": [round(s, 4) for s in self.comet_segments],
        }


def _validate(sources: list[str], candidates: list[str], references: list[str]) -> None:
    if not (len(sources) == len(candidates) == len(references)):
        raise ValueError(
            f"Segment counts differ: {len(sources)} source, "
            f"{len(candidates)} candidate, {len(references)} reference. "
            "BLEU/COMET require strict 1:1 alignment."
        )
    if not sources:
        raise ValueError("Empty corpus.")


def score(
    sources: list[str],
    candidates: list[str],
    references: list[str],
    use_comet: bool = False,
    gpus: int = 0,
) -> Scores:
    _validate(sources, candidates, references)

    bleu_metric = sacrebleu.BLEU()
    bleu = bleu_metric.corpus_score(candidates, [references])
    chrf = sacrebleu.CHRF().corpus_score(candidates, [references])
    result = Scores(
        bleu=bleu.score,
        chrf=chrf.score,
        bleu_signature=str(bleu_metric.get_signature()),
    )

    if use_comet:
        comet_out = _comet(sources, candidates, references, gpus=gpus)
        result.comet = comet_out["system_score"]
        result.comet_segments = comet_out["scores"]

    return result


def _comet(
    sources: list[str], candidates: list[str], references: list[str], gpus: int
) -> dict:
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            'COMET is not installed. Run: pip install "unbabel-comet>=2.2" '
            "(requires torch; Python 3.11/3.12 recommended)."
        ) from exc

    local_path = os.environ.get(COMET_MODEL_ENV)
    if local_path:
        # COMET's loader looks for hparams.yaml in the checkpoint file's
        # PARENT directory (e.g. .../wmt22-comet-da/checkpoints/model.ckpt
        # pairs with .../wmt22-comet-da/hparams.yaml). If the user gave us a
        # flat folder holding both, transparently reshape it to that layout
        # by symlinking (or copying, on Windows without dev mode) into a
        # `checkpoints/` subfolder — no manual mkdir needed.
        root = Path(local_path)
        if root.is_file():
            root = root.parent
        if not root.is_dir():
            raise RuntimeError(f"{COMET_MODEL_ENV} points to {local_path!r} which is not a folder.")

        model_ckpt = root / "checkpoints" / "model.ckpt"
        hparams = root / "hparams.yaml"
        if not model_ckpt.is_file():
            flat_ckpt = root / "model.ckpt"
            if flat_ckpt.is_file() and hparams.is_file():
                (root / "checkpoints").mkdir(exist_ok=True)
                try:
                    model_ckpt.symlink_to(flat_ckpt)
                except (OSError, NotImplementedError):
                    import shutil
                    shutil.move(str(flat_ckpt), str(model_ckpt))
            else:
                raise RuntimeError(
                    f"{COMET_MODEL_ENV}={local_path!r}: expected either "
                    f"checkpoints/model.ckpt + hparams.yaml, or a flat "
                    f"folder containing model.ckpt + hparams.yaml."
                )
        model = load_from_checkpoint(str(model_ckpt))
    else:
        model = load_from_checkpoint(download_model(COMET_MODEL))
    data = [
        {"src": s, "mt": c, "ref": r}
        for s, c, r in zip(sources, candidates, references)
    ]
    out = model.predict(data, gpus=gpus, progress_bar=True)
    return {"system_score": out.system_score, "scores": list(out.scores)}
