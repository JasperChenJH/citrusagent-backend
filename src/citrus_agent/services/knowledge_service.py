"""知识库入库服务。

本文件负责编排“文档解析 -> 文本分块 -> Payload 提取 -> embedding -> 写入 Qdrant”。
API 层后续只需要调用 KnowledgeIngestor，不需要知道内部细节。
"""

from __future__ import annotations

import hashlib

from src.citrus_agent.core.config import settings
from src.citrus_agent.pojo.knowledge import (
    ChunkPayload,
    DocumentEntity,
    DocumentIngestResult,
    KnowledgeChunk,
)
from src.citrus_agent.services.document_parser import DocumentParser, ParsedDocument
from src.citrus_agent.vectorstores.embeddings import EmbeddingProvider, create_embedding_provider
from src.citrus_agent.vectorstores.embeddings import BgeM3ApiEmbeddingProvider
from src.citrus_agent.vectorstores.qdrant import QdrantStore


class KnowledgeIngestor:
    """橘子知识库入库编排器。

    它不处理 HTTP 上传，只接收 MySQL documents 表对应的 DocumentEntity。
    后端上传模块先保存文件和数据库记录，再调用本类完成入库。
    """

    def __init__(
        self,
        parser: DocumentParser | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        qdrant_store: QdrantStore | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.parser = parser or DocumentParser()
        self.embedding_provider = embedding_provider or create_embedding_provider()
        self.qdrant_store = qdrant_store or QdrantStore(
            vector_size=self.embedding_provider.vector_size
        )
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")

    def ingest_document(
        self,
        document: DocumentEntity,
        replace_existing: bool = True,
    ) -> DocumentIngestResult:
        """根据 MySQL documents 实体入库一个文档。

        Args:
            document: 后端从 MySQL documents 表查询并组装的文档实体。
            replace_existing: 是否先删除该 document_id 对应的旧向量。

        Returns:
            DocumentIngestResult: 后端用于更新 documents 表状态的结果实体。
        """

        try:
            parsed_document = self.parser.parse_file(document.stored_path)
            chunks = self._build_chunks(
                document=parsed_document,
                kb_id=document.kb_id,
                document_id=document.id,
                file_name=document.original_filename,
            )
            if not chunks:
                raise ValueError("文档没有生成有效知识片段")

            vectors = self.embedding_provider.embed_texts([chunk.text for chunk in chunks])
            if len(vectors) != len(chunks):
                raise ValueError("embedding 返回数量和知识片段数量不一致")

            for chunk, vector in zip(chunks, vectors):
                chunk.vector = vector

            if replace_existing:
                self.qdrant_store.delete_by_document_id(document.id)
            self.qdrant_store.upsert_chunks(chunks)

            return DocumentIngestResult(
                document_id=document.id,
                kb_id=document.kb_id,
                success=True,
                chunk_count=len(chunks),
                error_message="",
            )
        except Exception as exc:
            return DocumentIngestResult(
                document_id=document.id,
                kb_id=document.kb_id,
                success=False,
                chunk_count=0,
                error_message=str(exc),
            )

    def _build_chunks(
        self,
        document: ParsedDocument,
        kb_id: int,
        document_id: int,
        file_name: str,
    ) -> list[KnowledgeChunk]:
        """把文档拆成带 payload 的知识片段。"""

        sections = document.pages or [
            {"page_no": None, "section": document.title, "content": document.content}
        ]

        chunks: list[KnowledgeChunk] = []
        chunk_index = 0
        for section in sections:
            section_text = str(section.get("content") or "")
            page_no = section.get("page_no")

            for text in self._split_text(section_text):
                chunk_id = self._build_chunk_id(
                    kb_id=kb_id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=text,
                )
                payload = ChunkPayload(
                    chunk_id=chunk_id,
                    kb_id=kb_id,
                    document_id=document_id,
                    file_name=file_name,
                    page=page_no if isinstance(page_no, int) else None,
                    chunk_index=chunk_index,
                    text=text,
                )
                point_id = self.qdrant_store.build_point_id(chunk_id)
                chunks.append(
                    KnowledgeChunk(
                        point_id=point_id,
                        text=text,
                        vector=[],
                        payload=payload,
                    )
                )
                chunk_index += 1

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """按固定长度和重叠长度拆分文本。

        这里使用简单字符分块，优先保证逻辑清楚。后续如果资料质量要求更高，
        可以替换成按标题、段落或语义分块。
        """

        clean_text = self._clean_text(text)
        if not clean_text:
            return []
        if len(clean_text) <= self.chunk_size:
            return [clean_text]

        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        while start < len(clean_text):
            end = min(start + self.chunk_size, len(clean_text))
            chunk = clean_text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(clean_text):
                break
            start += step
        return chunks

    def _clean_text(self, text: str) -> str:
        """清理多余空白，保留段落之间的换行。"""

        lines = [" ".join(line.strip().split()) for line in text.splitlines()]
        return "\n".join(line for line in lines if line)

    def _build_chunk_id(
        self,
        kb_id: int,
        document_id: int,
        chunk_index: int,
        text: str,
    ) -> str:
        """生成稳定 chunk_id，避免重复入库时 ID 变化。"""

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{kb_id}:{document_id}:{chunk_index}:{digest}"


class HybridKnowledgeIngestor(KnowledgeIngestor):
    """BGE-M3 服务器版 hybrid 入库编排器。

    该类复用 KnowledgeIngestor 的文档解析、清洗、分块和 payload 构造逻辑，
    只把 embedding 和 Qdrant 写入替换成 dense + sparse 版本。
    """

    def __init__(
        self,
        parser: DocumentParser | None = None,
        embedding_provider: BgeM3ApiEmbeddingProvider | None = None,
        qdrant_store: QdrantStore | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.parser = parser or DocumentParser()
        self.embedding_provider = embedding_provider or BgeM3ApiEmbeddingProvider()
        self.qdrant_store = qdrant_store or QdrantStore(
            collection_name=settings.qdrant_hybrid_collection,
            vector_size=self.embedding_provider.vector_size,
        )
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")

    def ingest_document(
        self,
        document: DocumentEntity,
        replace_existing: bool = True,
    ) -> DocumentIngestResult:
        """使用 BGE-M3 dense + sparse 入库一个文档。"""

        try:
            parsed_document = self.parser.parse_file(document.stored_path)
            chunks = self._build_chunks(
                document=parsed_document,
                kb_id=document.kb_id,
                document_id=document.id,
                file_name=document.original_filename,
            )
            if not chunks:
                raise ValueError("文档没有生成有效知识片段")

            embeddings = self.embedding_provider.embed_hybrid_texts(
                [chunk.text for chunk in chunks]
            )
            if len(embeddings) != len(chunks):
                raise ValueError("BGE-M3 返回数量和知识片段数量不一致")

            for chunk, embedding in zip(chunks, embeddings):
                chunk.dense_vector = embedding.dense
                chunk.sparse_vector = embedding.sparse

            if replace_existing:
                self.qdrant_store.delete_by_document_id_from_hybrid(document.id)
            self.qdrant_store.upsert_hybrid_chunks(chunks)

            return DocumentIngestResult(
                document_id=document.id,
                kb_id=document.kb_id,
                success=True,
                chunk_count=len(chunks),
                error_message="",
            )
        except Exception as exc:
            return DocumentIngestResult(
                document_id=document.id,
                kb_id=document.kb_id,
                success=False,
                chunk_count=0,
                error_message=str(exc),
            )
