"""
RAG 模块对接的数据传输对象 (DTO)
与独立 RAG 模块的 pojo/knowledge.py 对应
后端通过这些数据类与 RAGApi 交互
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DocumentEntity:
    """
    文档实体，传给 RAG 模块进行入库
    对应 RAG 模块的 DocumentEntity
    """
    id: int
    kb_id: int
    original_filename: str
    stored_path: str
    file_size: int = 0
    file_hash: str = ""
    mime_type: str = ""
    status: str = "pending"


@dataclass
class DocumentIngestResult:
    """
    RAG 模块入库返回结果
    对应 RAG 模块的 DocumentIngestResult
    """
    document_id: int
    kb_id: int
    success: bool
    chunk_count: int = 0
    error_message: str = ""


@dataclass
class SearchResult:
    """单个检索结果"""
    chunk_id: str
    content: str
    score: float
    payload: dict = field(default_factory=dict)


@dataclass
class DocumentSearchResult:
    """
    RAG 模块检索返回结果
    对应 RAG 模块的 DocumentSearchResult
    """
    kb_id: Optional[int]
    query_text: str
    top_k: int
    score: float
    chunks: list[SearchResult] = field(default_factory=list)
