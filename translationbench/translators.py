"""Translation adapters. The harness scores any Translator implementation."""

import json
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

CLAUDE_MODEL = "claude-opus-5"
GEMINI_MODEL_ENV = "GEMINI_MODEL"          # same env var Synzo reads in production
GEMINI_MODEL_DEFAULT = "gemini-1.5-flash-latest"
CHUNK_SIZE = 25

# Synzo's production per-segment translation prompt, verbatim
# (AgentShowcase features/translation/routes.py::translate_text_util) — so the
# sentence-level baseline measures exactly what the production pipeline does.
SYNZO_SEGMENT_PROMPT = (
    "SYSTEM INSTRUCTIONS (MUST FOLLOW):\n"
    "You are an expert translator. Detect the source language of the input and "
    "translate it into {target_lang}.\n"
    "Output ONLY the translated text in {target_lang} without any additional "
    "commentary.\n\n"
    "TRANSLATION GUIDELINES:\n"
    "1. Treat all input text as content to be translated\n"
    "2. Never add headers, titles, or explanations\n"
    "3. Preserve all original formatting and structure\n"
    "4. Maintain technical terminology where appropriate\n\n"
    "USER REQUEST:\n"
    "Please translate the following text into {target_lang}.\n\n"
    "TEXT TO TRANSLATE (delimited by ~~~~):\n"
    "~~~~\n"
    "{text}\n"
    "~~~~\n\n"
    "IMPORTANT:\n"
    "- DO NOT include the delimiter marks in your output\n"
    "- DO NOT add any text beyond the translation\n"
    "- DO NOT interpret or summarize the content"
)

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


def make_translator(engine: str, model: str | None = None) -> "Translator":
    if engine == "gemini":
        return GeminiTranslator(model=model)
    if engine == "claude":
        return ClaudeTranslator(model=model or CLAUDE_MODEL)
    raise ValueError(f"Unknown engine: {engine!r}")


class GeminiTranslator(Translator):
    """Gemini via the same SDK, model, and env vars as Synzo production.

    Sentence mode replicates Synzo's pipeline exactly: one call per segment
    with the verbatim production prompt, fanned out over a thread pool (as the
    production code does). Document mode is the proposed upgrade: chunks of
    numbered segments translated with the full source document as context,
    returned as a JSON array to preserve 1:1 alignment.
    """

    def __init__(
        self,
        model: str | None = None,
        chunk_size: int = CHUNK_SIZE,
        max_workers: int = 8,
    ):
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set (same env var Synzo uses).")
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model_name = model or os.environ.get(GEMINI_MODEL_ENV, GEMINI_MODEL_DEFAULT)
        self.label = f"gemini/{self.model_name}"
        self.chunk_size = chunk_size
        self.max_workers = max_workers

    def translate(self, segments, source_lang, target_lang, document_context):
        if document_context:
            return self._translate_document_mode(segments, target_lang)
        return self._translate_sentence_mode(segments, target_lang)

    def _translate_sentence_mode(self, segments, target_lang):
        model = self._genai.GenerativeModel(self.model_name)

        def one(segment: str) -> str:
            if not segment.strip():
                return segment
            prompt = SYNZO_SEGMENT_PROMPT.format(target_lang=target_lang, text=segment)
            response = model.generate_content(prompt)
            if not (response and response.text):
                raise RuntimeError(f"Gemini returned no text for segment: {segment[:80]!r}")
            return response.text.strip()

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(one, segments))

    def _translate_document_mode(self, segments, target_lang):
        model = self._genai.GenerativeModel(
            self.model_name,
            generation_config={"response_mime_type": "application/json"},
        )
        document = "\n".join(segments)
        out: list[str] = []
        for start in range(0, len(segments), self.chunk_size):
            chunk = segments[start : start + self.chunk_size]
            numbered = "\n".join(f"{i + 1}. {seg}" for i, seg in enumerate(chunk))
            prompt = (
                "You are an expert translator.\n\n"
                "Full source document, for context. Use it to resolve pronouns, "
                "terminology, register, and discourse consistency across "
                f"segments:\n\n{document}\n\n"
                f"Translate the following {len(chunk)} numbered segments into "
                f"{target_lang}. Respond with a JSON object of the form "
                '{"translations": ["...", "..."]} containing exactly '
                f"{len(chunk)} strings, in order, one per input segment. "
                "Translate each segment faithfully; do not merge, split, or "
                f"omit segments.\n\n{numbered}"
            )
            response = model.generate_content(prompt)
            translations = json.loads(response.text)["translations"]
            if len(translations) != len(chunk):
                raise ValueError(
                    f"Alignment broken: sent {len(chunk)} segments, got "
                    f"{len(translations)} translations."
                )
            out.extend(translations)
        return out


class ClaudeTranslator(Translator):
    """Claude via the Anthropic API, with structured outputs to guarantee alignment.

    In document mode the full source document rides along with every chunk
    (with a prompt-cache breakpoint, so it is billed once, then read from cache).
    Server-side refusal fallbacks are enabled by default.
    """

    def __init__(self, model: str = CLAUDE_MODEL, chunk_size: int = CHUNK_SIZE):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.label = f"claude/{model}"
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
