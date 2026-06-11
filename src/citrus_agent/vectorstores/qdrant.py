"""Qdrant 向量库封装。

本文件只处理 Qdrant 连接、collection 管理、批量入库和检索。
API 层或业务层不要直接依赖 qdrant-client 的细节，后续替换向量库时会更容易。
"""

from __future__ import annotations

import uuid
from typing import Any

from src.citrus_agent.core.config import settings
from src.citrus_agent.pojo.knowledge import KnowledgeChunk


class QdrantStore:
    """橘子知识库 Qdrant 适配器。"""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
        distance: str | None = None,
    ) -> None:
        self.url = url or settings.qdrant_url
        self.api_key = api_key if api_key is not None else settings.qdrant_api_key
        self.collection_name = collection_name or settings.qdrant_collection
        self.vector_size = vector_size or settings.embedding_vector_size
        self.distance = distance or settings.qdrant_distance
        self.client = self._create_client()

    def ensure_collection(self) -> None:
        """确保 collection 存在，不存在就创建。

        注意：Qdrant collection 的向量维度创建后不能直接修改。
        如果后续 embedding 模型维度变化，建议新建 collection 并重新入库。
        """

        from qdrant_client import models

        existing = [item.name for item in self.client.get_collections().collections]
        if self.collection_name in existing:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=self._distance_value(),
            ),
        )
        self._create_payload_indexes()

    def upsert_chunks(self, chunks: list[KnowledgeChunk], batch_size: int = 128) -> None:
        """批量写入知识片段。

        Args:
            chunks: 已经带有向量和 payload 的知识片段。
            batch_size: 每批写入数量。批量写入比逐条写入稳定很多。
        """

        if not chunks:
            return

        from qdrant_client import models

        self.ensure_collection()

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            points = [
                models.PointStruct(
                    id=chunk.point_id,
                    vector=chunk.vector,
                    payload=chunk.payload.to_dict(),
                )
                for chunk in batch
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

    def delete_by_document_id(self, document_id: int) -> None:
        """按 MySQL documents.id 删除旧片段。

        重新入库同一个文档前可以先调用该方法，避免旧版本内容残留。
        """

        from qdrant_client import models

        self.ensure_collection()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def delete_by_kb_id(self, kb_id: int) -> None:
        """按 MySQL knowledge_bases.id 删除一个知识库下的所有向量。

        该方法只删除 Qdrant 中 payload.kb_id 等于指定值的 points，
        不会操作 MySQL 的 knowledge_bases 表或 documents 表。
        后端删除知识库时，可以先删 MySQL，再调用这里清理向量；也可以反过来由后端控制事务顺序。
        """

        from qdrant_client import models

        self.ensure_collection()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="kb_id",
                            match=models.MatchValue(value=kb_id),
                        )
                    ]
                )
            ),
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 30,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """执行向量检索，并返回统一字典结果。

        第一版只使用 dense vector。filters 可传入 kb_id 或 document_id 做强过滤。
        """

        qdrant_filter = self._build_filter(filters or {})

        # qdrant-client 1.18 推荐 query_points；保留 search 兜底，兼容旧版本。
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
            points = getattr(response, "points", response)
        else:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )

        results: list[dict[str, Any]] = []
        for point in points:
            payload = dict(point.payload or {})
            results.append(
                {
                    "id": str(point.id),
                    "score": float(point.score),
                    "payload": payload,
                }
            )
        return results

    def build_point_id(self, chunk_id: str) -> str:
        """把业务 chunk_id 转成 Qdrant 可接受的稳定 UUID。"""

        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def _create_client(self):
        """创建 Qdrant 客户端。"""

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError("连接 Qdrant 需要安装 qdrant-client") from exc

        api_key = self.api_key or None
        return QdrantClient(url=self.url, api_key=api_key)

    def _distance_value(self):
        """把配置里的距离名称转换为 qdrant-client 的枚举。"""

        from qdrant_client import models

        normalized = self.distance.lower()
        if normalized == "cosine":
            return models.Distance.COSINE
        if normalized == "dot":
            return models.Distance.DOT
        if normalized in {"euclid", "euclidean"}:
            return models.Distance.EUCLID
        raise ValueError(f"不支持的 Qdrant 距离算法：{self.distance}")

    def _build_filter(self, filters: dict[str, Any]):
        """构造 Qdrant 强过滤条件。

        第一版 payload 不再保存 status，过滤条件由调用方显式传入。
        """

        from qdrant_client import models

        must = []

        for key, value in filters.items():
            if value is None or value == "":
                continue
            must.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )

        if not must:
            return None
        return models.Filter(must=must)

    def _create_payload_indexes(self) -> None:
        """为常用过滤字段创建 payload 索引。

        如果当前 Qdrant 版本或配置不支持创建索引，入库和检索仍然可以继续工作。
        """

        from qdrant_client import models

        fields = {
            "kb_id": models.PayloadSchemaType.INTEGER,
            "document_id": models.PayloadSchemaType.INTEGER,
            "chunk_id": models.PayloadSchemaType.KEYWORD,
        }
        for field_name, field_schema in fields.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception:
                # 索引创建失败不影响主流程，避免本地 Qdrant 版本差异导致启动失败。
                continue
