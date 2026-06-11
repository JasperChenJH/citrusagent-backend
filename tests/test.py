"""真实上传 Markdown 文档到 Qdrant 的集成测试。

这个测试使用本项目里的 tests/README.md，模拟后端已经把文件记录写入
MySQL documents 表，然后把 DocumentEntity 传给 RAG API。

注意：
    该测试会真实调用 embedding API，并真实写入 Qdrant。
"""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

from src.citrus_agent.pojo.knowledge import DocumentEntity, DocumentSearchResult
from src.citrus_agent.rag.rag_api import RAGApi


def calculate_file_hash(path: Path) -> str:
    """计算文件 hash，模拟 documents.file_hash。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_document_entity_from_md(path: Path) -> DocumentEntity:
    """把 Markdown 文件封装成后端传给 RAG 的 DocumentEntity。"""

    mime_type, _ = mimetypes.guess_type(str(path))
    return DocumentEntity(
        id=2,
        kb_id=1,
        original_filename=path.name,
        stored_path=str(path),
        file_size=path.stat().st_size,
        file_hash=calculate_file_hash(path),
        mime_type=mime_type,
        status="pending",
    )


def test_upload_readme_md_to_qdrant() -> None:
    """真实调用 RAG API，把 tests/README.md 写入 Qdrant。"""


    api = RAGApi()
    d = api.search(query_text="测试", top_k=10, kb_id=2)
    print(d)
    print(api.search(query_text="Qdrant 向量数据库", top_k=10))
    api.delete_knowledge_base_vectors(kb_id=1)
    # result = api.ingest_document(document, replace_existing=True)
    #
    # assert result.document_id == document.id
    # assert result.kb_id == document.kb_id
    # assert result.success is True, result.error_message
    # assert result.chunk_count > 0
    # assert result.error_message == ""
