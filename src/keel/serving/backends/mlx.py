"""M1 default inference backend: mlx-lm on Apple Silicon.

A thin wrapper — our value is hardware-aware selection and tuning (M1), not the engine.
``generate``/``stream`` treat ``prompt`` as already-rendered text and complete it raw.
``chat`` renders messages through the model's own tokenizer chat template before
completing them — the template an instruct model was actually trained on, which is what
keeps it answering in character turn by turn instead of free-associating. Strictly local:
mlx-lm loads weights from disk and never calls a remote API.

The mlx-lm imports live behind module-level helpers (``_load``, ``_stream_responses``) so
the loop logic can be unit-tested by monkeypatching them, with no mlx install required.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from keel.serving.backends.base import ChatMessage, GenerationResult, InferenceBackend


def _mlx() -> Any:
    try:
        import mlx_lm
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "MLXBackend needs the 'mlx' extra. Install it with: pip install 'keel[mlx]'"
        ) from exc
    return mlx_lm


def _load(model_id: str) -> tuple[Any, Any]:
    """Load (model, tokenizer) from disk / the local HF cache."""
    model, tokenizer = _mlx().load(model_id)
    return model, tokenizer


def _stream_responses(
    model: Any, tokenizer: Any, prompt: str, max_tokens: int, temperature: float
) -> Iterator[Any]:
    """Yield mlx-lm GenerationResponse objects (.text, .prompt_tokens, .generation_tokens)."""
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature)
    yield from _mlx().stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
    )


def _render_chat(tokenizer: Any, messages: Sequence[ChatMessage]) -> str:
    """Render messages through the tokenizer's own chat template, generation-ready."""
    rendered: str = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        add_generation_prompt=True,
        tokenize=False,
    )
    return rendered


class MLXBackend(InferenceBackend):
    name = "mlx-lm"

    def __init__(self, model_id: str, *, default_max_tokens: int = 1024) -> None:
        self.model_id = model_id
        self.default_max_tokens = default_max_tokens
        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is None:
            self._model, self._tokenizer = _load(self.model_id)
        return self._model, self._tokenizer

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        grammar: str | None = None,
    ) -> GenerationResult:
        if grammar is not None:
            raise NotImplementedError("constrained decoding (grammar) lands in M2")
        model, tokenizer = self._ensure_loaded()
        parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        for resp in _stream_responses(model, tokenizer, prompt, max_tokens, temperature):
            parts.append(resp.text)
            prompt_tokens = resp.prompt_tokens
            completion_tokens = resp.generation_tokens
        return GenerationResult(
            text="".join(parts),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        grammar: str | None = None,
    ) -> Iterator[str]:
        if grammar is not None:
            raise NotImplementedError("constrained decoding (grammar) lands in M2")
        model, tokenizer = self._ensure_loaded()
        for resp in _stream_responses(model, tokenizer, prompt, max_tokens, temperature):
            yield resp.text

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        grammar: str | None = None,
    ) -> GenerationResult:
        if grammar is not None:
            raise NotImplementedError("constrained decoding (grammar) lands in M2")
        _, tokenizer = self._ensure_loaded()
        prompt = _render_chat(tokenizer, messages)
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)
