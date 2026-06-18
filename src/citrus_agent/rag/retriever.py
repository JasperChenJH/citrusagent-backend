"""橘子知识库检索器。

当前版本只做向量检索：问题 -> embedding -> Qdrant search -> 返回 top_k 个 chunk。
不做重排，不调用 LLM，不做业务标签加权。
"""

from __future__ import annotations

from typing import Any

from src.citrus_agent.core.config import settings
from src.citrus_agent.pojo.knowledge import SearchQuery, SearchResult
from src.citrus_agent.rag.reranker import QwenRerankerClient
from src.citrus_agent.vectorstores.embeddings import (
    BgeM3ApiEmbeddingProvider,
    EmbeddingProvider,
    create_embedding_provider,
)
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


class HybridCitrusRetriever:
    """BGE-M3 + Qwen reranker 的 hybrid 检索器。

    检索流程：
    1. 用户问题调用 BGE-M3，生成 dense + sparse 查询向量。
    2. Qdrant dense 召回 TopK。
    3. Qdrant sparse 召回 TopK。
    4. 使用 RRF 在本地融合候选片段。
    5. 调用 Qwen3-Reranker-4B 对候选片段重排。
    6. 返回最终 TopN 片段，供后续大模型拼 prompt。
    """

    def __init__(
        self,
        embedding_provider: BgeM3ApiEmbeddingProvider | None = None,
        qdrant_store: QdrantStore | None = None,
        reranker: QwenRerankerClient | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        candidate_top_k: int | None = None,
        final_top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider or BgeM3ApiEmbeddingProvider()
        self.qdrant_store = qdrant_store or QdrantStore(
            collection_name=settings.qdrant_hybrid_collection,
            vector_size=self.embedding_provider.vector_size,
        )
        self.reranker = reranker or QwenRerankerClient()
        self.dense_top_k = dense_top_k or settings.hybrid_dense_top_k
        self.sparse_top_k = sparse_top_k or settings.hybrid_sparse_top_k
        self.candidate_top_k = candidate_top_k or settings.hybrid_candidate_top_k
        self.final_top_k = final_top_k or settings.hybrid_final_top_k
        self.rrf_k = rrf_k or settings.hybrid_rrf_k

    def search(
        self,
        query_text: str,
        top_k: int | None = None,
        kb_id: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """执行 hybrid 检索。

        Args:
            query_text: 用户问题。
            top_k: 最终返回数量，不传时使用配置 HYBRID_FINAL_TOP_K。
            kb_id: 知识库 ID，传入时按 payload.kb_id 强过滤。
            filters: 额外过滤条件，例如 {"document_id": 12}。

        Returns:
            list[SearchResult]: 经过 rerank 后的最终片段列表。
        """

        if not query_text.strip():
            return []

        actual_filters = dict(filters or {})
        if kb_id is not None:
            actual_filters["kb_id"] = kb_id

        query_embedding = self.embedding_provider.embed_hybrid_text(query_text)
        dense_results = self.qdrant_store.search_hybrid_dense(
            query_vector=query_embedding.dense,
            top_k=self.dense_top_k,
            filters=actual_filters,
        )
        sparse_results = self.qdrant_store.search_hybrid_sparse(
            sparse_vector=query_embedding.sparse,
            top_k=self.sparse_top_k,
            filters=actual_filters,
        )

        candidates = self._rrf_merge(
            dense_results=dense_results,
            sparse_results=sparse_results,
            limit=self.candidate_top_k,
        )
        if not candidates:
            return []

        reranked = self._rerank(query_text=query_text, candidates=candidates)
        return reranked[: top_k or self.final_top_k]

    def _rrf_merge(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
        limit: int,
    ) -> list[SearchResult]:
        """使用 RRF 融合 dense 和 sparse 召回结果。

        RRF 公式：score += 1 / (rrf_k + rank)，rank 从 1 开始。
        """

        merged: dict[str, dict[str, Any]] = {}
        self._add_rrf_scores(
            merged=merged,
            results=dense_results,
            source_name="dense",
        )
        self._add_rrf_scores(
            merged=merged,
            results=sparse_results,
            source_name="sparse",
        )

        sorted_items = sorted(
            merged.values(),
            key=lambda item: float(item["rrf_score"]),
            reverse=True,
        )

        candidates: list[SearchResult] = []
        for item in sorted_items[:limit]:
            payload = dict(item["payload"])
            payload["dense_score"] = item.get("dense_score")
            payload["sparse_score"] = item.get("sparse_score")
            payload["rrf_score"] = item["rrf_score"]
            candidates.append(
                SearchResult(
                    chunk_id=str(payload.get("chunk_id") or item["id"]),
                    content=str(payload.get("text") or payload.get("content") or ""),
                    score=float(item["rrf_score"]),
                    payload=payload,
                )
            )
        return candidates

    def _add_rrf_scores(
        self,
        merged: dict[str, dict[str, Any]],
        results: list[dict[str, Any]],
        source_name: str,
    ) -> None:
        """把一路召回结果累加到 RRF 候选表里。"""

        for rank, result in enumerate(results, start=1):
            payload = dict(result.get("payload") or {})
            key = str(payload.get("chunk_id") or result.get("id") or "")
            if not key:
                continue

            if key not in merged:
                merged[key] = {
                    "id": str(result.get("id") or key),
                    "payload": payload,
                    "rrf_score": 0.0,
                }
            merged[key]["rrf_score"] += 1.0 / (self.rrf_k + rank)
            merged[key][f"{source_name}_score"] = float(result.get("score") or 0.0)

    def _rerank(
        self,
        query_text: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        """调用 Qwen reranker 对候选片段重排。"""

        scores = self.reranker.score(
            query_text=query_text,
            documents=[candidate.content for candidate in candidates],
        )
        if len(scores) != len(candidates):
            raise ValueError("Qwen reranker 返回数量和候选片段数量不一致")

        reranked: list[SearchResult] = []
        for candidate, score in zip(candidates, scores):
            payload = dict(candidate.payload)
            payload["rerank_score"] = float(score)
            reranked.append(
                SearchResult(
                    chunk_id=candidate.chunk_id,
                    content=candidate.content,
                    score=float(score),
                    payload=payload,
                )
            )

        return sorted(reranked, key=lambda item: item.score, reverse=True)
