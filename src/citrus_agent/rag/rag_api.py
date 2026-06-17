"""后端调用 RAG 能力的统一入口。

本文件不是 FastAPI 路由，而是给后端业务代码 import 调用的门面层。
后端只需要传入 MySQL documents 表对应的 DocumentEntity，不需要关心
文档解析、分块、embedding、Qdrant 写入和检索的内部实现。
"""

from __future__ import annotations

from src.citrus_agent.pojo.knowledge import (
    DocumentEntity,
    DocumentIngestResult,
    DocumentSearchResult,
)
from src.citrus_agent.rag.retriever import CitrusRetriever
from src.citrus_agent.services.knowledge_service import HybridKnowledgeIngestor, KnowledgeIngestor
from src.citrus_agent.vectorstores.qdrant import QdrantStore


class RAGApi:
    """后端专用 RAG 调用门面。

    这个类只做很薄的一层封装：
    1. 文档入库：DocumentEntity -> DocumentIngestResult。
    2. BGE-M3 hybrid 文档入库：DocumentEntity -> DocumentIngestResult。
    3. 文档向量删除：按 document_id 删除 Qdrant 中旧 chunk。
    4. 知识库向量删除：按 kb_id 删除 Qdrant 中该知识库的所有 chunk。
    5. 知识检索：传 kb_id 时按知识库过滤，不传 kb_id 时全库检索。
    """
    def __init__(
        self,
        ingestor: KnowledgeIngestor | None = None,
        hybrid_ingestor: HybridKnowledgeIngestor | None = None,
        retriever: CitrusRetriever | None = None,
        qdrant_store: QdrantStore | None = None,
    ) -> None:
        self.ingestor = ingestor or KnowledgeIngestor()
        self.hybrid_ingestor = hybrid_ingestor
        self.retriever = retriever or CitrusRetriever()
        self.qdrant_store = qdrant_store or QdrantStore()

    def ingest_document(
        self,
        document: DocumentEntity,
        replace_existing: bool = True,
    ) -> DocumentIngestResult:
        """入库一个 MySQL documents 文档实体。

        Args:
            document: 后端从 MySQL documents 表查询后组装的文档实体。
            replace_existing: 是否先删除该 如果document_id相同 是否删除旧向量。

        Returns:
            DocumentIngestResult: 后端用于更新 documents 表状态的结果。
        """

        return self.ingestor.ingest_document(
            document=document,
            replace_existing=replace_existing,
        )

    def ingest_document_with_bge_m3(
        self,
        document: DocumentEntity,
        replace_existing: bool = True,
    ) -> DocumentIngestResult:
        """使用服务器 BGE-M3 dense + sparse 入库一个文档。

        原来的 ingest_document() 不变，继续用于本地 API embedding 测试。
        该方法只用于正式服务器 hybrid collection 入库。
        """

        hybrid_ingestor = self.hybrid_ingestor or HybridKnowledgeIngestor()
        return hybrid_ingestor.ingest_document(
            document=document,
            replace_existing=replace_existing,
        )

    def delete_document_vectors(self, document_id: int) -> bool:
        """删除一个文档在 Qdrant 中的所有 chunk 向量。

        Args:
            document_id: MySQL documents.id。

        Returns:
            bool: 删除调用成功返回 True；异常会向上抛出，方便后端统一处理。
        """

        self.qdrant_store.delete_by_document_id(document_id)
        return True

    def delete_knowledge_base_vectors(self, kb_id: int) -> bool:
        """删除一个知识库在 Qdrant 中的所有 chunk 向量。

        Args:
            kb_id: MySQL knowledge_bases.id。

        Returns:
            bool: 删除调用成功返回 True；异常会向上抛出，方便后端统一处理。
        """

        self.qdrant_store.delete_by_kb_id(kb_id)
        return True

    def search(
        self,
        query_text: str,
        top_k: int | None = None,
        kb_id: int | None = None,
    ) -> DocumentSearchResult:
        """检索相关 chunk。

        Args:
            query_text: 用户问题或检索文本。
            top_k: Qdrant 向量检索返回数量。
            kb_id: MySQL knowledge_bases.id。传入时按知识库过滤；不传时全库检索。

        Returns:
            DocumentSearchResult: 检索返回实体，chunks 中是命中的片段。
        """

        filters = {"kb_id": kb_id} if kb_id is not None else None
        chunks = self.retriever.search(
            query_text=query_text,
            top_k=top_k,
            filters=filters,
        )
        actual_top_k = top_k or getattr(self.retriever, "top_k", len(chunks))
        best_score = max((chunk.score for chunk in chunks), default=0.0)
        return DocumentSearchResult(
            kb_id=kb_id,
            query_text=query_text,
            top_k=actual_top_k,
            score=best_score,
            chunks=chunks,
        )
