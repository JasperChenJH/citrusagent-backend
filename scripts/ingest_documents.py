"""本地批量入库脚本。

用法示例：
    conda run -n agent-dev python scripts/ingest_documents.py knowledge_raw --debug-embedding

脚本会：
    1. 确保 MySQL 数据库和表存在。
    2. 创建或复用一个知识库。
    3. 为每个文件写入 documents 记录。
    4. 组装 DocumentEntity。
    5. 调用 KnowledgeIngestor.ingest_document() 写入 Qdrant。
    6. 根据返回结果更新 documents 状态。
"""

from __future__ import annotations

import argparse
import hashlib
import mimetypes
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select

from scripts.init_mysql_tables import create_database_if_needed
from src.citrus_agent.db.models import Base, DocumentModel, KnowledgeBaseModel
from src.citrus_agent.db.session import SessionLocal, engine
from src.citrus_agent.pojo.knowledge import DocumentEntity
from src.citrus_agent.services.document_parser import DocumentParser
from src.citrus_agent.services.knowledge_service import KnowledgeIngestor
from src.citrus_agent.vectorstores.embeddings import FixedEmbeddingProvider, create_embedding_provider


def collect_files(root: Path) -> list[Path]:
    """递归收集支持的知识库文件。"""

    if root.is_file():
        return [root] if root.suffix.lower() in DocumentParser.supported_suffixes else []

    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in DocumentParser.supported_suffixes:
            files.append(path)
    return files


def calculate_file_hash(path: Path) -> str:
    """计算文件 hash，用于 documents.file_hash。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_tables() -> None:
    """确保 MySQL 数据库和表已经存在。"""

    create_database_if_needed()
    Base.metadata.create_all(bind=engine)


def get_or_create_knowledge_base(name: str, description: str | None = None) -> int:
    """创建或复用一个知识库，返回 kb_id。"""

    with SessionLocal() as session:
        existing = session.execute(
            select(KnowledgeBaseModel).where(KnowledgeBaseModel.name == name)
        ).scalar_one_or_none()
        if existing:
            return int(existing.id)

        knowledge_base = KnowledgeBaseModel(name=name, description=description)
        session.add(knowledge_base)
        session.commit()
        session.refresh(knowledge_base)
        return int(knowledge_base.id)


def create_document_record(path: Path, kb_id: int) -> DocumentModel:
    """为本地文件创建 documents 记录。"""

    file_hash = calculate_file_hash(path)
    mime_type, _ = mimetypes.guess_type(str(path))

    with SessionLocal() as session:
        document = DocumentModel(
            kb_id=kb_id,
            original_filename=path.name,
            stored_path=str(path.resolve()),
            file_size=path.stat().st_size,
            file_hash=file_hash,
            mime_type=mime_type,
            status="pending",
        )
        session.add(document)
        session.commit()
        session.refresh(document)
        session.expunge(document)
        return document


def to_document_entity(document: DocumentModel) -> DocumentEntity:
    """把 ORM 文档对象转换成 RAG 入库实体。"""

    return DocumentEntity(
        id=int(document.id),
        kb_id=int(document.kb_id),
        original_filename=document.original_filename,
        stored_path=document.stored_path,
        file_size=int(document.file_size),
        file_hash=document.file_hash,
        mime_type=document.mime_type,
        status=document.status,
    )


def update_document_result(document_id: int, success: bool, chunk_count: int, error_message: str) -> None:
    """根据 RAG 返回结果更新 documents 状态。"""

    with SessionLocal() as session:
        document = session.get(DocumentModel, document_id)
        if not document:
            return
        document.status = "ready" if success else "failed"
        document.chunk_count = chunk_count
        document.error_message = error_message or None
        session.commit()


def main() -> int:
    """脚本入口。"""

    parser = argparse.ArgumentParser(description="按 MySQL documents 实体流程导入橘子知识库文档")
    parser.add_argument("path", help="资料文件或资料目录")
    parser.add_argument("--kb-name", default="橘子知识库", help="本地测试知识库名称")
    parser.add_argument(
        "--debug-embedding",
        action="store_true",
        help="使用固定伪向量，只用于验证流程，不适合真实检索",
    )
    args = parser.parse_args()

    target = Path(args.path)
    files = collect_files(target)
    if not files:
        print(f"没有找到支持的文档：{target}")
        return 1

    ensure_tables()
    kb_id = get_or_create_knowledge_base(args.kb_name, description="本地测试知识库")
    embedding_provider = FixedEmbeddingProvider() if args.debug_embedding else create_embedding_provider()
    ingestor = KnowledgeIngestor(embedding_provider=embedding_provider)

    success_count = 0
    fail_count = 0
    for file_path in files:
        document_record = create_document_record(file_path, kb_id)
        document_entity = to_document_entity(document_record)
        result = ingestor.ingest_document(document_entity)
        update_document_result(
            document_id=result.document_id,
            success=result.success,
            chunk_count=result.chunk_count,
            error_message=result.error_message,
        )

        if result.success:
            success_count += 1
            status = "成功"
        else:
            fail_count += 1
            status = "失败"

        print(f"[{status}] {file_path}")
        print(f"  document_id: {result.document_id}")
        print(f"  kb_id: {result.kb_id}")
        print(f"  chunks: {result.chunk_count}")
        if result.error_message:
            print(f"  error: {result.error_message}")

    print()
    print(f"共发现文件：{len(files)}")
    print(f"入库成功：{success_count}")
    print(f"入库失败：{fail_count}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
