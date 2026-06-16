"""Embedding 模型封装。

本文件负责把文本转换成向量。当前默认使用 embedding API 生成向量，
模型为 text-embedding-v4，向量维度为 1024。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from src.citrus_agent.core.config import settings


class EmbeddingProvider(ABC):
    """Embedding 提供者基础接口。

    RAG 和 Qdrant 层只依赖这个接口，不直接依赖具体模型库。
    """

    model_name: str
    vector_size: int

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """把单条文本转换成向量。"""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """把多条文本批量转换成向量。"""


class ApiEmbeddingProvider(EmbeddingProvider):
    """OpenAI 兼容 embedding API。

    这个类只负责向量化，不负责聊天模型调用。当前配置用于阿里云 DashScope
    OpenAI 兼容模式的 text-embedding-v4。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.embedding_api_key
        self.base_url = base_url or settings.embedding_base_url
        self.model_name = model_name or settings.embedding_model
        self.vector_size = dimensions or settings.embedding_dimensions
        self._client = None

    def embed_text(self, text: str) -> list[float]:
        """把单条文本转换成向量。"""

        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """批量调用 embedding API。"""

        clean_texts = [text.strip() for text in texts if text and text.strip()]
        if not clean_texts:
            return []

        client = self._load_client()
        response = client.embeddings.create(
            model=self.model_name,
            input=clean_texts,
            dimensions=self.vector_size,
        )
        return [item.embedding for item in response.data]

    def _load_client(self):
        """懒加载 API 客户端。

        本地部署的 embedding 服务（localhost/127.0.0.1）允许 API Key 为空。
        """

        if self._client is not None:
            return self._client

        is_local = (
            "localhost" in self.base_url
            or "127.0.0.1" in self.base_url
            or "0.0.0.0" in self.base_url
        )

        if not is_local and (not self.api_key or self.api_key == "replace-me"):
            raise ValueError("EMBEDDING_API_KEY 不能为空，请在 .env 中配置真实密钥")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("使用 embedding API 需要安装 openai") from exc

        api_key = self.api_key if self.api_key and self.api_key != "replace-me" else "not-needed"
        self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client


class LocalBgeEmbeddingProvider(EmbeddingProvider):
    """本地 BGE embedding 模型。

    默认模型从配置 `EMBEDDING_MODEL_NAME` 读取。第一版建议用较小的
    `BAAI/bge-small-zh-v1.5` 跑通流程，后续服务器资源充足后可改为 `BAAI/bge-m3`。
    模型采用懒加载，只有真正调用 embedding 时才加载，避免应用启动过慢。
    """

    def __init__(
        self,
        model_name: str | None = None,
        vector_size: int | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self.vector_size = vector_size or settings.embedding_vector_size
        self.device = device
        self._model = None

    def embed_text(self, text: str) -> list[float]:
        """把单条文本转换成向量。"""

        vectors = self.embed_texts([text])
        return vectors[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """批量生成向量。

        BGE 系列一般推荐归一化向量，这样 Qdrant 使用 Cosine 距离时更稳定。
        """

        clean_texts = [text.strip() for text in texts if text and text.strip()]
        if not clean_texts:
            return []

        model = self._load_model()
        embeddings = model.encode(
            clean_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.astype(float).tolist() for vector in embeddings]

    def _load_model(self):
        """懒加载 sentence-transformers 模型。"""

        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("使用本地 BGE embedding 需要安装 sentence-transformers") from exc

        if self.device:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        else:
            self._model = SentenceTransformer(self.model_name)

        return self._model


class FixedEmbeddingProvider(EmbeddingProvider):
    """测试或调试用的固定维度 embedding。

    这个类不适合生产检索，只用于没有本地模型时验证入库和检索流程。
    """

    def __init__(self, vector_size: int = 8) -> None:
        self.model_name = "fixed-debug-embedding"
        self.vector_size = vector_size

    def embed_text(self, text: str) -> list[float]:
        """把文本转换成简单、稳定的伪向量。"""

        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """批量生成固定维度伪向量。"""

        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.vector_size
            for index, char in enumerate(text):
                vector[index % self.vector_size] += (ord(char) % 31) / 31.0
            total = sum(value * value for value in vector) ** 0.5
            if total:
                vector = [value / total for value in vector]
            vectors.append(vector)
        return vectors


def create_embedding_provider(provider_name: str | None = None) -> EmbeddingProvider:
    """根据配置创建默认 embedding 提供者。

    Args:
        provider_name: 可选提供者名称。为空时读取 EMBEDDING_PROVIDER。

    Returns:
        EmbeddingProvider: 可直接用于入库和检索的 embedding 对象。
    """

    provider = (provider_name or settings.embedding_provider).lower()
    if provider in {"api", "openai", "dashscope"}:
        return ApiEmbeddingProvider()
    if provider in {"local_bge", "bge", "local"}:
        return LocalBgeEmbeddingProvider()
    if provider in {"fixed", "debug"}:
        return FixedEmbeddingProvider(vector_size=settings.embedding_vector_size)
    raise ValueError(f"不支持的 EMBEDDING_PROVIDER：{provider_name or settings.embedding_provider}")
