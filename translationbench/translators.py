"""Translation adapters. The harness scores any Translator; Claude is the reference one."""

import json
from abc import ABC, abstractmethod

import anthropic

MODEL = "claude-opus-5"
CHUNK_SIZE = 25

OUTPUT_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "translations": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["translations"],
        "additionalProperties": False,
    },
}


class Translator(ABC):
    """Translate aligned segments 1:1. Implement this to benchmark another engine."""

    @abstractmethod
    def translate(
        self,
        segments: list[str],
        source_lang: str,
        target_lang: str,
        document_context: bool,
    ) -> list[str]: ...


class ClaudeTranslator(Translator):
    """Claude via the Anthropic API, with structured outputs to guarantee alignment.

    In document mode the full source document rides along with every chunk
    (with a prompt-cache breakpoint, so it is billed once, then read from cache).
    Server-side refusal fallbacks are enabled by default.
    """

    def __init__(self, model: str = MODEL, chunk_size: int = CHUNK_SIZE):
        self.client = anthropic.Anthropic()
        self.model = model
        self.chunk_size = chunk_size

    def translate(self, segments, source_lang, target_lang, document_context):
        out: list[str] = []
        document = "\n".join(segments) if document_context else None
        for start in range(0, len(segments), self.chunk_size):
            chunk = segments[start : start + self.chunk_size]
            out.extend(
                self._translate_chunk(chunk, source_lang, target_lang, document)
            )
        return out

    def _translate_chunk(
        self,
        chunk: list[str],
        source_lang: str,
        target_lang: str,
        document: str | None,
    ) -> list[str]:
        numbered = "\n".join(f"{i + 1}. {seg}" for i, seg in enumerate(chunk))
        instruction = (
            f"Translate the following {len(chunk)} numbered segments from "
            f"{source_lang} to {target_lang}. Return exactly {len(chunk)} "
            "translations, in order, one per input segment. Translate each "
            "segment faithfully; do not merge, split, or omit segments.\n\n"
        )
        content: list[dict] = []
        if document is not None:
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Full source document, for context. Use it to resolve "
                        "pronouns, terminology, register, and discourse "
                        "consistency across segments:\n\n" + document
                    ),
                    "cache_control": {"type": "ephemeral"},
                }
            )
        else:
            instruction = (
                "Translate each segment independently, using no context beyond "
                "the segment itself.\n\n" + instruction
            )
        content.append({"type": "text", "text": instruction + numbered})

        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"format": OUTPUT_SCHEMA},
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(
                f"Model declined a translation chunk (stop_details="
                f"{response.stop_details!r}); no fallback recovered it."
            )

        text = next(b.text for b in response.content if b.type == "text")
        translations = json.loads(text)["translations"]
        if len(translations) != len(chunk):
            raise ValueError(
                f"Alignment broken: sent {len(chunk)} segments, got "
                f"{len(translations)} translations."
            )
        return translations
