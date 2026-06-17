"""橘子知识库数据对象。

本文件只定义数据结构，不连接数据库、不调用模型、不写业务流程。
这些对象会在文档解析、知识入库、Qdrant 检索和 RAG 检索之间传递。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SparseEmbedding:
    """BGE-M3 生成的 sparse 稀疏向量。

    Attributes:
        indices: 稀疏向量中非零 token 的编号列表。
        values: 每个 token 对应的权重列表，长度必须和 indices 一致。
    """

    indices: list[int]
    values: list[float]


@dataclass
class HybridEmbedding:
    """BGE-M3 生成的 hybrid 向量结果。

    Attributes:
        dense: dense 语义向量，BGE-M3 默认 1024 维。
        sparse: sparse 稀疏关键词向量，用于 Qdrant hybrid search。
    """

    dense: list[float]
    sparse: SparseEmbedding


@dataclass
class DocumentEntity:
    """MySQL documents 表对应的文档实体。

    RAG 模块只接收这个实体，不直接连接 MySQL。后端负责从 documents 表查询记录，
    并把这些字段组装后传给入库服务。
    """
    id: int
    kb_id: int
    original_filename: str
    stored_path: str
    file_size: int
    file_hash: str
    mime_type: str | None = None
    status: str = "pending"


@dataclass
class DocumentIngestResult:
    """文档入库返回实体。

    后端拿到该对象后，可以据此更新 MySQL documents 表中的 status、
    chunk_count 和 error_message。
    """

    document_id: int
    kb_id: int
    success: bool
    chunk_count: int = 0
    error_message: str = ""


@dataclass
class ChunkPayload:
    """写入 Qdrant payload 的最小字段。

    第一版只保留删除、知识库过滤、引用展示和 prompt 拼接需要的字段。
    """

    chunk_id: str
    kb_id: int
    document_id: int
    file_name: str
    page: int | None
    chunk_index: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        """转换为 Qdrant 可直接保存的字典。"""

        return {
            "chunk_id": self.chunk_id,
            "kb_id": self.kb_id,
            "document_id": self.document_id,
            "file_name": self.file_name,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }


@dataclass
class KnowledgeChunk:
    """准备写入向量库的知识片段。

    Attributes:
        point_id: Qdrant point ID。Qdrant 推荐使用整数或 UUID 字符串。
        text: 当前片段文本，等于 payload.text。
        vector: 旧版 dense-only 入库使用的向量。
        payload: 当前片段的结构化信息。
        dense_vector: BGE-M3 hybrid 入库使用的 dense 向量。
        sparse_vector: BGE-M3 hybrid 入库使用的 sparse 向量。
    """

    point_id: str
    text: str
    vector: list[float]
    payload: ChunkPayload
    dense_vector: list[float] = field(default_factory=list)
    sparse_vector: SparseEmbedding | None = None


@dataclass
class SearchQuery:
    """橘子知识库检索请求。

    Attributes:
        query_text: 用户问题或检索文本。
        top_k: Qdrant 向量检索返回数量。
        filters: 额外强过滤条件，例如按 kb_id 或 document_id 检索。
    """

    query_text: str
    top_k: int = 30
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """单个向量检索命中的 chunk。

    Attributes:
        chunk_id: 命中的知识片段 ID。
        content: 命中的知识片段正文。
        score: Qdrant 返回的向量相似度分数。
        payload: 命中片段的完整 payload，方便后端展示出处。
    """

    chunk_id: str
    content: str
    score: float
    payload: dict[str, Any]


@dataclass
class DocumentSearchResult:
    """检索返回实体。

    LLM 层给 RAG 一个问题后，RAG 只做向量检索并返回 top_k 个 chunk。
    后续 LLM 可以直接把 chunks 里的 content/text 拼进 prompt。

    Attributes:
        kb_id: 当前检索限定的知识库 ID；不限定知识库时为 None。
        query_text: 本次检索的问题文本。
        top_k: 本次请求召回的片段数量。
        score: 本次检索结果的最高相似度分数，没有命中时为 0.0。
        chunks: 命中的片段列表，每个片段也有自己的 score。
    """

    kb_id: int | None
    query_text: str
    top_k: int
    score: float = 0.0
    chunks: list[SearchResult] = field(default_factory=list)
