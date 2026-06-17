"""Pluggable embedding service with provider interface.

Supports FastEmbed (local), Gemini, OpenAI, or any custom provider
implementing the EmbeddingProvider protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol interface for embedding providers."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        ...

    def get_dimensions(self) -> int:
        """Return the vector dimensionality for this provider."""
        ...


class FastEmbedProvider:
    """Local embedding using FastEmbed (no API costs)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self._model_name = model_name
        # Dimension lookup for common models
        self._dims = {
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
        }

    def embed_text(self, text: str) -> list[float]:
        results = list(self._model.embed([text]))
        return results[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [r.tolist() for r in self._model.embed(texts)]

    def get_dimensions(self) -> int:
        return self._dims.get(self._model_name, 384)


class GeminiEmbeddingProvider:
    """Embedding using Google Gemini API."""

    def __init__(self, model_name: str = "models/text-embedding-004", api_key: str | None = None):
        import google.generativeai as genai

        if api_key:
            genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model_name
        self._dims = {"models/text-embedding-004": 768}

    def embed_text(self, text: str) -> list[float]:
        result = self._genai.embed_content(model=self._model_name, content=text)
        return result["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    def get_dimensions(self) -> int:
        return self._dims.get(self._model_name, 768)


class OpenAIEmbeddingProvider:
    """Embedding using OpenAI API."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._model_name = model_name
        self._dims = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def embed_text(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=[text], model=self._model_name)
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self._model_name)
        return [d.embedding for d in resp.data]

    def get_dimensions(self) -> int:
        return self._dims.get(self._model_name, 1536)


class EmbeddingServiceFactory:
    """Factory that creates the correct EmbeddingProvider from manifest config."""

    _registry: dict[str, type] = {
        "fastembed": FastEmbedProvider,
        "gemini": GeminiEmbeddingProvider,
        "openai": OpenAIEmbeddingProvider,
    }

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """Register a custom embedding provider."""
        cls._registry[name] = provider_class

    @classmethod
    def create(cls, config: dict[str, Any]) -> EmbeddingProvider:
        """Create an EmbeddingProvider from a config dict.

        Expected config keys:
            provider: str  — "fastembed", "gemini", "openai", or custom registered name
            model: str     — model name for the provider
            api_key: str   — optional API key (for Gemini/OpenAI)
        """
        provider_name = config.get("provider", "fastembed")
        model = config.get("model", "")
        api_key = config.get("api_key")

        provider_cls = cls._registry.get(provider_name)
        if provider_cls is None:
            raise ValueError(
                f"Unknown embedding provider '{provider_name}'. "
                f"Available: {list(cls._registry.keys())}"
            )

        kwargs: dict[str, Any] = {}
        if model:
            kwargs["model_name"] = model
        if api_key:
            kwargs["api_key"] = api_key

        return provider_cls(**kwargs)
