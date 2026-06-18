"""
知识库管理服务
负责文档上传、入库、检索、增删改查
通过 MySQL 持久化元数据，通过 RAGApi 对接 RAG 模块
"""

import hashlib
from pathlib import Path
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.citrus_agent.core.settings import get_settings
from src.citrus_agent.db.db_models import Document, KnowledgeBase
from src.citrus_agent.db.knowledge import (
    KnowledgeDocumentCreate,
    KnowledgeDocumentResponse,
    KnowledgeDocumentUpdate,
    KnowledgeSearchResult,
)
from src.citrus_agent.pojo.rag_dto import DocumentEntity, DocumentIngestResult
from src.citrus_agent.services.logger import logger


def _compute_file_hash(file_path: str) -> str:
    """计算文件 SHA256"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_mime_type(filename: str) -> str:
    """根据文件扩展名推断 MIME 类型"""
    ext = Path(filename).suffix.lower()
    mime_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".csv": "text/csv",
        ".json": "application/json",
    }
    return mime_map.get(ext, "application/octet-stream")


def _doc_to_response(doc: Document) -> KnowledgeDocumentResponse:
    """ORM Document 转为 API 响应"""
    return KnowledgeDocumentResponse(
        id=str(doc.id),
        kb_id=doc.kb_id,
        title=doc.title,
        source_type=doc.source_type,
        description=doc.description,
        tags=doc.tags.split(",") if doc.tags else [],
        chunk_count=doc.chunk_count,
        file_name=doc.original_filename,
        status=doc.status,
        error_message=doc.error_message,
        created_at=doc.created_at.isoformat() if doc.created_at else "",
        updated_at=doc.updated_at.isoformat() if doc.updated_at else None,
    )


class KnowledgeService:
    """知识库管理服务"""

    def __init__(self):
        settings = get_settings()
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 知识库操作 ====================

    async def create_knowledge_base(
        self, db: AsyncSession, name: str, description: str | None = None, icon: str | None = None
    ) -> KnowledgeBase:
        """创建知识库"""
        kb = KnowledgeBase(name=name, description=description, icon=icon)
        db.add(kb)
        await db.flush()
        logger.info(f"知识库已创建: {name} (id={kb.id})")
        return kb

    async def list_knowledge_bases(
        self, db: AsyncSession, page: int = 1, page_size: int = 20
    ) -> tuple[list[KnowledgeBase], int]:
        """获取知识库列表"""
        total_result = await db.execute(select(func.count(KnowledgeBase.id)))
        total = total_result.scalar() or 0

        result = await db.execute(
            select(KnowledgeBase)
            .order_by(KnowledgeBase.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        kbs = list(result.scalars().all())
        return kbs, total

    async def get_knowledge_base(self, db: AsyncSession, kb_id: int) -> KnowledgeBase | None:
        """获取知识库详情"""
        result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
        return result.scalar_one_or_none()

    async def delete_knowledge_base(self, db: AsyncSession, kb_id: int) -> bool:
        """
        删除知识库（同时删除 Qdrant 中的向量）

        注意：MySQL 中的 documents 记录也会被级联删除
        """
        kb = await self.get_knowledge_base(db, kb_id)
        if not kb:
            return False

        # 删除 Qdrant 中该知识库的所有向量
        try:
            from src.citrus_agent.services.rag_bridge import get_rag_api
            rag_api = get_rag_api()
            rag_api.delete_knowledge_base_vectors(kb_id=kb_id)
        except Exception as e:
            logger.warning(f"删除 Qdrant 向量失败: {e}")

        # 删除 MySQL 中的文档记录
        docs_result = await db.execute(select(Document).where(Document.kb_id == kb_id))
        docs = list(docs_result.scalars().all())
        for doc in docs:
            await db.delete(doc)

        # 删除知识库
        await db.delete(kb)
        logger.info(f"知识库已删除: kb_id={kb_id}")
        return True

    # ==================== 文档操作 ====================

    async def upload_document(
        self,
        db: AsyncSession,
        file_path: str,
        file_name: str,
        kb_id: int,
        create_data: KnowledgeDocumentCreate,
    ) -> KnowledgeDocumentResponse:
        """
        上传文档并处理入库

        流程：
        1. 校验文件名是否重复
        2. 在 MySQL 创建 document 记录（status=processing）
        3. 组装 DocumentEntity 调用 RAGApi 入库
        4. 根据入库结果更新 MySQL document 状态（无论成功失败均持久化）
        """
        file_ext = Path(file_name).suffix.lower()
        base_name = Path(file_name).stem
        full_file_name = f"{base_name}{file_ext}"

        existing = await db.execute(
            select(Document).where(
                Document.original_filename == full_file_name,
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise ValueError(f"知识库中已存在同名文件: {full_file_name}")

        file_size = Path(file_path).stat().st_size
        file_hash = _compute_file_hash(file_path)
        mime_type = _get_mime_type(file_name)

        doc = Document(
            kb_id=kb_id,
            title=create_data.title,
            original_filename=full_file_name,
            stored_path=file_path,
            file_size=file_size,
            file_hash=file_hash,
            mime_type=mime_type,
            source_type=create_data.source_type,
            description=create_data.description,
            tags=",".join(create_data.tags) if create_data.tags else None,
            status="processing",
        )
        db.add(doc)
        await db.flush()
        await db.refresh(doc)

        try:
            from src.citrus_agent.services.rag_bridge import get_rag_api
            rag_api = get_rag_api()

            entity = DocumentEntity(
                id=doc.id,
                kb_id=kb_id,
                original_filename=full_file_name,
                stored_path=file_path,
                file_size=file_size,
                file_hash=file_hash,
                mime_type=mime_type,
                status="processing",
            )

            result: DocumentIngestResult = await rag_api.ingest_document(entity)

            if result.success:
                doc.status = "ready"
                doc.chunk_count = result.chunk_count
                logger.info(f"文档入库成功: {create_data.title}，共 {result.chunk_count} 个切片")
            else:
                doc.status = "failed"
                doc.error_message = result.error_message
                logger.error(f"文档入库失败: {result.error_message}")

        except Exception as e:
            doc.status = "failed"
            doc.error_message = str(e)
            logger.error(f"文档入库异常: {e}")

        await db.flush()
        await db.refresh(doc)
        return _doc_to_response(doc)

    async def search(
        self,
        db: AsyncSession,
        query: str,
        top_k: int = 5,
        kb_id: int | None = None,
    ) -> list[KnowledgeSearchResult]:
        """
        知识库检索（调用 RAGApi）

        Args:
            db: 数据库会话
            query: 查询文本
            top_k: 返回数量
            kb_id: 限定知识库ID
        """
        try:
            from src.citrus_agent.services.rag_bridge import get_rag_api
            rag_api = get_rag_api()

            result = await rag_api.search(
                query_text=query,
                top_k=top_k,
                kb_id=kb_id,
            )

            return [
                KnowledgeSearchResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=chunk.score,
                    document_id=str(chunk.payload.get("document_id", "")),
                    document_title=chunk.payload.get("file_name", ""),
                    source_type=chunk.payload.get("source_type", ""),
                    metadata=chunk.payload,
                )
                for chunk in result.chunks
            ]
        except Exception as e:
            logger.error(f"知识库检索失败: {e}")
            return []

    async def list_documents(
        self,
        db: AsyncSession,
        kb_id: int | None = None,
        source_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[KnowledgeDocumentResponse], int]:
        """获取文档列表（分页）"""
        query = select(Document)
        count_query = select(func.count(Document.id))

        if kb_id is not None:
            query = query.where(Document.kb_id == kb_id)
            count_query = count_query.where(Document.kb_id == kb_id)
        if source_type:
            query = query.where(Document.source_type == source_type)
            count_query = count_query.where(Document.source_type == source_type)

        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        result = await db.execute(
            query.order_by(Document.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        docs = list(result.scalars().all())

        return [_doc_to_response(d) for d in docs], total

    async def get_document(self, db: AsyncSession, doc_id: int) -> KnowledgeDocumentResponse | None:
        """获取单个文档详情"""
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return None
        return _doc_to_response(doc)

    async def update_document(
        self, db: AsyncSession, doc_id: int, update_data: KnowledgeDocumentUpdate
    ) -> KnowledgeDocumentResponse | None:
        """更新文档元数据"""
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return None

        if update_data.title is not None:
            doc.title = update_data.title
        if update_data.description is not None:
            doc.description = update_data.description
        if update_data.tags is not None:
            doc.tags = ",".join(update_data.tags) if update_data.tags else None

        await db.flush()
        return _doc_to_response(doc)

    async def delete_document(self, db: AsyncSession, doc_id: int) -> bool:
        """
        删除文档及其 Qdrant 向量

        Args:
            db: 数据库会话
            doc_id: 文档ID

        Returns:
            是否删除成功
        """
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if not doc:
            return False

        # 删除 Qdrant 中的向量
        try:
            from src.citrus_agent.services.rag_bridge import get_rag_api
            rag_api = get_rag_api()
            rag_api.delete_document_vectors(document_id=doc_id)
        except Exception as e:
            logger.warning(f"删除 Qdrant 向量失败: {e}")

        # 删除磁盘文件
        if doc.stored_path:
            file_path = Path(doc.stored_path)
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"磁盘文件已删除: {doc.stored_path}")
            except Exception as e:
                logger.warning(f"删除磁盘文件失败: {doc.stored_path}, 错误: {e}")

        # 删除 MySQL 记录
        await db.delete(doc)
        logger.info(f"文档已删除: doc_id={doc_id}")
        return True