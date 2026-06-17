"""Unit tests for the pluggable embedding service."""

from unittest.mock import MagicMock, patch

import pytest

from src.knowledge.embedding import (
    EmbeddingProvider,
    EmbeddingServiceFactory,
    FastEmbedProvider,
    GeminiEmbeddingProvider,
    OpenAIEmbeddingProvider,
)


# --- Protocol compliance ---


class MockProvider:
    """A minimal custom provider for testing the protocol."""

    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def get_dimensions(self) -> int:
        return 3


def test_mock_provider_satisfies_protocol():
    provider = MockProvider()
    assert isinstance(provider, EmbeddingProvider)


def test_mock_provider_embed_text():
    provider = MockProvider()
    result = provider.embed_text("hello")
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


def test_mock_provider_embed_batch():
    provider = MockProvider()
    result = provider.embed_batch(["a", "b"])
    assert len(result) == 2


# --- Factory tests ---


def test_factory_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        EmbeddingServiceFactory.create({"provider": "nonexistent"})


def test_factory_register_custom_provider():
    EmbeddingServiceFactory.register_provider("mock", MockProvider)
    provider = EmbeddingServiceFactory.create({"provider": "mock"})
    assert isinstance(provider, EmbeddingProvider)
    assert provider.get_dimensions() == 3
    # Cleanup
    del EmbeddingServiceFactory._registry["mock"]


def test_factory_default_is_fastembed():
    """Factory should default to fastembed when no provider specified."""
    # We can't actually instantiate FastEmbed without the library downloaded,
    # so we mock the import
    with patch("src.knowledge.embedding.FastEmbedProvider") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        EmbeddingServiceFactory._registry["fastembed"] = mock_cls
        provider = EmbeddingServiceFactory.create({})
        mock_cls.assert_called_once()
        # Restore
        EmbeddingServiceFactory._registry["fastembed"] = FastEmbedProvider


# --- FastEmbedProvider unit tests (mocked) ---


def test_fastembed_get_dimensions_known_model():
    provider = FastEmbedProvider.__new__(FastEmbedProvider)
    provider._model_name = "BAAI/bge-small-en-v1.5"
    provider._dims = {"BAAI/bge-small-en-v1.5": 384, "BAAI/bge-base-en-v1.5": 768}
    assert provider.get_dimensions() == 384


def test_fastembed_get_dimensions_unknown_model():
    provider = FastEmbedProvider.__new__(FastEmbedProvider)
    provider._model_name = "custom/model"
    provider._dims = {"BAAI/bge-small-en-v1.5": 384}
    assert provider.get_dimensions() == 384  # fallback


# --- OpenAIEmbeddingProvider unit tests (mocked) ---


def test_openai_get_dimensions():
    provider = OpenAIEmbeddingProvider.__new__(OpenAIEmbeddingProvider)
    provider._model_name = "text-embedding-3-small"
    provider._dims = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
    assert provider.get_dimensions() == 1536


def test_openai_embed_text_mocked():
    provider = OpenAIEmbeddingProvider.__new__(OpenAIEmbeddingProvider)
    provider._model_name = "text-embedding-3-small"
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_client.embeddings.create.return_value = mock_resp
    provider._client = mock_client

    result = provider.embed_text("hello")
    assert result == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once_with(input=["hello"], model="text-embedding-3-small")


# --- GeminiEmbeddingProvider unit tests (mocked) ---


def test_gemini_get_dimensions():
    provider = GeminiEmbeddingProvider.__new__(GeminiEmbeddingProvider)
    provider._model_name = "models/text-embedding-004"
    provider._dims = {"models/text-embedding-004": 768}
    assert provider.get_dimensions() == 768


def test_gemini_embed_text_mocked():
    provider = GeminiEmbeddingProvider.__new__(GeminiEmbeddingProvider)
    provider._model_name = "models/text-embedding-004"
    mock_genai = MagicMock()
    mock_genai.embed_content.return_value = {"embedding": [0.5, 0.6]}
    provider._genai = mock_genai

    result = provider.embed_text("test")
    assert result == [0.5, 0.6]
    mock_genai.embed_content.assert_called_once_with(model="models/text-embedding-004", content="test")
