"""Translation adapters. The harness scores any Translator implementation."""

import json
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

CLAUDE_MODEL = "claude-opus-5"
GEMINI_MODEL_ENV = "GEMINI_MODEL"          # same env var Synzo reads in production
# Google retired `gemini-1.5-flash-latest` (Synzo's historical default). Default
# now points at gemini-3.5-flash-lite — the current low-latency, cost-optimized
# Flash-tier model that supports 1M-token context (needed for document mode)
# and structured outputs. Override with GEMINI_MODEL if Synzo pins another
# model in production.
GEMINI_MODEL_DEFAULT = "gemini-3.5-flash-lite"
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


def make_translator(
    engine: str, model: str | None = None, thinking: bool = False
) -> "Translator":
    if engine == "gemini":
        return GeminiTranslator(model=model, thinking=thinking)
    if engine == "claude":
        if thinking:
            # Not a blocker — Claude Opus 5 has its own adaptive-thinking path,
            # separate from the Gemini flag. Silently ignore for now.
            pass
        return ClaudeTranslator(model=model or CLAUDE_MODEL)
    raise ValueError(f"Unknown engine: {engine!r}")


class GeminiTranslator(Translator):
    """Gemini via the modern `google-genai` SDK.

    Sentence mode replicates Synzo's production pipeline shape: one call per
    segment with the verbatim production prompt, fanned out over a thread pool
    (as the production code does).

    Document mode adds two features on top of production:
      1. The full source document is sent as an *explicit* cached prefix, so
         it is billed once per document instead of once per chunk (~10x cost
         reduction on a 500-segment doc chunked 25 at a time).
      2. Optional server-side thinking, controllable via the `thinking`
         constructor arg.

    Whole mode is the one-call variant: whole doc in, whole translated doc out,
    aligned by JSON array. Uses the full 1M-token context window; skips
    chunking entirely.
    """

    _CACHE_MIN_TOKENS = 4096  # min prefix size explicit caching requires

    def __init__(
        self,
        model: str | None = None,
        chunk_size: int = CHUNK_SIZE,
        max_workers: int = 8,
        thinking: bool = False,
    ):
        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set (same env var Synzo uses).")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model or os.environ.get(GEMINI_MODEL_ENV, GEMINI_MODEL_DEFAULT)
        thinking_tag = ".think" if thinking else ""
        self.label = f"gemini/{self.model_name}{thinking_tag}"
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.thinking = thinking

    def translate(self, segments, source_lang, target_lang, document_context):
        if document_context:
            return self._translate_document_mode(segments, target_lang)
        return self._translate_sentence_mode(segments, target_lang)

    def translate_whole(self, segments, target_lang):
        """Single-call whole-document translation, JSON-aligned."""
        from google.genai import types

        numbered = "\n".join(f"{i + 1}. {seg}" for i, seg in enumerate(segments))
        prompt = (
            "You are an expert translator. Translate the following "
            f"{len(segments)} numbered segments into {target_lang}. Use the "
            "whole document as context to keep pronouns, terminology, "
            "register, and discourse consistent across segments. Respond "
            'with a JSON object of the form {"translations": ["...", ...]} '
            f"containing exactly {len(segments)} strings, in order, one per "
            "input segment. Do not merge, split, or omit segments.\n\n"
            + numbered
        )
        cfg_kwargs = dict(response_mime_type="application/json")
        if self.thinking:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        translations = json.loads(response.text)["translations"]
        if len(translations) != len(segments):
            raise ValueError(
                f"Alignment broken: sent {len(segments)} segments, got "
                f"{len(translations)} translations."
            )
        return translations

    def _translate_sentence_mode(self, segments, target_lang):
        def one(segment: str) -> str:
            if not segment.strip():
                return segment
            prompt = SYNZO_SEGMENT_PROMPT.format(target_lang=target_lang, text=segment)
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            if not (response and response.text):
                raise RuntimeError(f"Gemini returned no text for segment: {segment[:80]!r}")
            return response.text.strip()

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(one, segments))

    def _translate_document_mode(self, segments, target_lang):
        from google.genai import types

        document = "\n".join(segments)
        cache = self._maybe_create_cache(document, target_lang)

        def build_config(use_cache: bool):
            # gemini-3.6-flash on client.models.generate_content rejects
            # response_schema (400 INVALID_ARGUMENT); we ask for JSON via mime
            # type and validate alignment on our side. thinking_config is
            # elided when self.thinking is False — some models reject an
            # explicit "disable" and it's not needed either way.
            kwargs = dict(response_mime_type="application/json")
            if self.thinking:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)
            if use_cache:
                kwargs["cached_content"] = cache.name
            return types.GenerateContentConfig(**kwargs)

        def translate_chunk(chunk: list[str], use_cache: bool) -> list[str]:
            numbered = "\n".join(f"{i + 1}. {seg}" for i, seg in enumerate(chunk))
            if use_cache:
                # System instruction (holding the full source doc) rides on
                # the cache; the request itself only carries the per-chunk ask.
                prompt = (
                    f"Translate the following {len(chunk)} numbered segments "
                    f"into {target_lang}. Respond with a JSON object of the "
                    'form {"translations": ["...", ...]} containing exactly '
                    f"{len(chunk)} strings, in order, one per input segment.\n\n"
                    + numbered
                )
            else:
                prompt = (
                    "You are an expert translator.\n\n"
                    "Full source document, for context. Use it to resolve "
                    "pronouns, terminology, register, and discourse "
                    "consistency across segments:\n\n"
                    f"{document}\n\n"
                    f"Translate the following {len(chunk)} numbered segments "
                    f"into {target_lang}. Respond with a JSON object of the "
                    'form {"translations": ["...", ...]} containing exactly '
                    f"{len(chunk)} strings, in order, one per input segment.\n\n"
                    + numbered
                )
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=build_config(use_cache=use_cache),
                )
            except Exception as exc:
                # Surface the full API response body so we can see what the
                # server actually rejected (the SDK's default __repr__ hides it).
                details = getattr(exc, "details", None) or getattr(exc, "response_json", None)
                print(
                    f"[translationbench] generate_content failed. "
                    f"model={self.model_name} use_cache={use_cache} "
                    f"prompt_chars={len(prompt)} thinking={self.thinking}",
                    flush=True,
                )
                if details:
                    print(f"[translationbench] server said: {details}", flush=True)
                raise
            translations = json.loads(response.text)["translations"]
            if len(translations) != len(chunk):
                raise ValueError(
                    f"Alignment broken: sent {len(chunk)} segments, got "
                    f"{len(translations)} translations."
                )
            return translations

        out: list[str] = []
        use_cache = cache is not None
        try:
            for start in range(0, len(segments), self.chunk_size):
                chunk = segments[start : start + self.chunk_size]
                if use_cache:
                    try:
                        out.extend(translate_chunk(chunk, use_cache=True))
                        continue
                    except Exception as exc:
                        # The cache doesn't work with this model+SDK+endpoint
                        # combo. Fall through to inline mode for this and every
                        # subsequent chunk; correct, just costlier.
                        if os.environ.get("TRANSLATIONBENCH_DEBUG_CACHE"):
                            print(
                                f"[translationbench] cached generate_content "
                                f"failed on chunk {start}: {exc}. Falling back "
                                f"to inline document per chunk.",
                                flush=True,
                            )
                        use_cache = False
                out.extend(translate_chunk(chunk, use_cache=False))
        finally:
            if cache is not None:
                try:
                    self.client.caches.delete(name=cache.name)
                except Exception:
                    pass
        return out

    def _maybe_create_cache(self, document: str, target_lang: str):
        """Create an explicit content cache for the source doc + system role.

        Returns None if caching isn't feasible for this request; the caller
        falls back to sending the doc inline per chunk (correct, just costlier).
        Reasons cache creation can decline:
          * doc smaller than the model's minimum cacheable size
          * model doesn't support explicit caching on the current endpoint
          * network / auth blip
        Set TRANSLATIONBENCH_DEBUG_CACHE=1 to see the underlying error.
        """
        from google.genai import types

        # Cheap token estimate — Gemini's tokenizer runs ~3-4 chars/token for
        # English; API-side minimum is 4096 tokens for Flash-tier. We only
        # attempt caching when we're clearly over the threshold.
        if len(document) < self._CACHE_MIN_TOKENS * 4:
            return None
        try:
            return self.client.caches.create(
                model=self.model_name,
                config=types.CreateCachedContentConfig(
                    display_name="translationbench-document-context",
                    system_instruction=(
                        "You are an expert translator. When translating "
                        f"segments into {target_lang}, use the full source "
                        "document below as context to keep pronouns, "
                        "terminology, register, and discourse consistent "
                        f"across segments. Source document:\n\n{document}"
                    ),
                    ttl="600s",
                ),
            )
        except Exception as exc:
            if os.environ.get("TRANSLATIONBENCH_DEBUG_CACHE"):
                print(
                    f"[translationbench] explicit caching declined; falling "
                    f"back to inline document per chunk. Reason: {exc}",
                    flush=True,
                )
            return None


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
