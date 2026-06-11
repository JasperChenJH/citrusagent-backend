"""橘子知识库检索器。

当前版本只做向量检索：问题 -> embedding -> Qdrant search -> 返回 top_k 个 chunk。
不做重排，不调用 LLM，不做业务标签加权。
"""

from __future__ import annotations

from typing import Any

from src.citrus_agent.core.config import settings
from src.citrus_agent.pojo.knowledge import SearchQuery, SearchResult
from src.citrus_agent.vectorstores.embeddings import EmbeddingProvider, create_embedding_provider
from src.citrus_agent.vectorstores.qdrant import QdrantStore


class CitrusRetriever:
    """橘子知识库检索器。

    调用方式：
        retriever = CitrusRetriever()
        results = retriever.search("砂糖橘溃疡病怎么防？")
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        qdrant_store: QdrantStore | None = None,
        top_k: int | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider or create_embedding_provider()
        self.qdrant_store = qdrant_store or QdrantStore(
            vector_size=self.embedding_provider.vector_size
        )
        self.top_k = top_k or settings.retrieval_top_k

    def search(
        self,
        query_text: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """检索橘子知识库。

        Args:
            query_text: 用户问题或检索文本。
            top_k: Qdrant 召回数量，不传则使用配置。
            filters: 额外强过滤条件，例如 {"kb_id": 1} 或 {"document_id": 12}。

        Returns:
            list[SearchResult]: Qdrant 向量检索返回的知识片段列表。
        """

        query = SearchQuery(
            query_text=query_text,
            top_k=top_k or self.top_k,
            filters=filters or {},
        )
        return self.search_by_query(query)

    def search_by_query(self, query: SearchQuery) -> list[SearchResult]:
        """根据 SearchQuery 执行检索。"""

        if not query.query_text.strip():
            return []

        query_vector = self.embedding_provider.embed_text(query.query_text)
        raw_results = self.qdrant_store.search(
            query_vector=query_vector,
            top_k=query.top_k,
            filters=query.filters,
        )

        results: list[SearchResult] = []
        for item in raw_results:
            payload = dict(item.get("payload") or {})
            results.append(
                SearchResult(
                    chunk_id=str(payload.get("chunk_id") or item.get("id") or ""),
                    content=str(payload.get("text") or payload.get("content") or ""),
                    score=float(item.get("score") or 0.0),
                    payload=payload,
                )
            )

        return results
