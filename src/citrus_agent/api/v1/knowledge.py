"""
知识库管理 API 路由
提供知识库 CRUD、文档上传、检索、增删改查等接口
所有元数据持久化到 MySQL，向量操作通过 RAGApi
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.citrus_agent.api.deps import get_knowledge_service
from src.citrus_agent.core.settings import get_settings
from src.citrus_agent.core.database import get_db
from src.citrus_agent.common.common import ResponseBase, PaginatedResponse, PaginationMeta
from src.citrus_agent.db.knowledge import (
    KnowledgeDocumentCreate,
    KnowledgeDocumentResponse,
    KnowledgeDocumentUpdate,
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
)
from src.citrus_agent.services.sql_service import KnowledgeService
from src.citrus_agent.services.logger import logger

router = APIRouter(prefix="/knowledge", tags=["知识库管理"])


# ==================== 知识库 CRUD ====================

@router.post("/bases", response_model=ResponseBase, summary="创建知识库")
async def create_knowledge_base(
    name: str = Form(..., description="知识库名称"),
    description: Optional[str] = Form(default=None, description="描述"),
    icon: Optional[str] = Form(default=None, description="图标标识"),
    db: AsyncSession = Depends(get_db),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
) -> ResponseBase:
    """创建知识库"""
    kb = await knowledge_service.create_knowledge_base(
        db, name=name, description=description, icon=icon
    )
    return ResponseBase(data={
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "icon": kb.icon,
        "is_active": kb.is_active,
        "created_at": kb.created_at.isoformat() if kb.created_at else None,
    })


@router.get("/bases", response_model=PaginatedResponse, summary="获取知识库列表")
async def list_knowledge_bases(
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_db),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
) -> PaginatedResponse:
    """获取知识库列表"""
    kbs, total = await knowledge_service.list_knowledge_bases(db, page=page, page_size=page_size)
    total_pages = (total + page_size - 1) // page_size

    data = [
        {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description,
            "icon": kb.icon,
            "is_active": kb.is_active,
            "created_at": kb.created_at.isoformat() if kb.created_at else None,
        }
        for kb in kbs
    ]

    return PaginatedResponse(
        data=data,
        pagination=PaginationMeta(
            total=total, page=page, page_size=page_size, total_pages=total_pages
        ),
    )


@router.delete("/bases/{kb_id}", response_model=ResponseBase, summary="删除知识库")
async def delete_knowledge_base(
    kb_id: int,
    db: AsyncSession = Depends(get_db),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
) -> ResponseBase:
    """删除知识库及其所有文档和向量"""
    success = await knowledge_service.delete_knowledge_base(db, kb_id)
    if not success:
        return ResponseBase(code=404, message="知识库不存在")
    return ResponseBase(message="知识库已删除")


# ==================== 文档操作 ====================

@router.post("/upload", response_model=ResponseBase, summary="上传文档到知识库")
async def upload_document(
    file: UploadFile = File(..., description="上传的文件"),
    kb_id: int = Form(..., description="所属知识库ID"),
    title: str = Form(..., description="文档标题"),
    source_type: str = Form(default="general", description="来源类型"),
    description: Optional[str] = Form(default=None, description="文档描述"),
    tags: str = Form(default="", description="标签，逗号分隔"),
    db: AsyncSession = Depends(get_db),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
) -> ResponseBase:
    """
    上传文档到知识库

    支持的文件类型: PDF, DOCX, TXT, MD, XLSX, CSV, JSON
    文档将自动切片、向量化并写入 Qdrant
    """
    settings = get_settings()

    # 校验文件类型
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        return ResponseBase(
            code=400,
            message=f"不支持的文件类型: {file_ext}，支持: {settings.ALLOWED_EXTENSIONS}",
        )

    # 保存上传文件
    upload_dir = Path(settings.UPLOAD_DIR)
    # 以年月份进行创建每月的文件夹
    month_dir = datetime.now().strftime("%Y-%m")
    upload_dir = upload_dir/month_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = upload_dir / f"{uuid.uuid4()}{file_ext}"

    try:
        with open(stored_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 解析标签
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        create_data = KnowledgeDocumentCreate(
            title=title,
            source_type=source_type,
            description=description,
            tags=tag_list,
        )

        result = await knowledge_service.upload_document(
            db=db,
            file_path=str(stored_path),
            file_name=file.filename,
            kb_id=kb_id,
            create_data=create_data,
        )

        return ResponseBase(data=result.model_dump())

    except ValueError as e:
        logger.warning(f"文档上传校验失败: {e}")
        if stored_path.exists():
            stored_path.unlink()
        return ResponseBase(code=400, message=str(e))
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        # 清理文件
        if stored_path.exists():
            stored_path.unlink()
        return ResponseBase(code=500, message=f"文档上传失败: {str(e)}")


@router.post("/search", response_model=ResponseBase, summary="知识库检索")
async def search_knowledge(
    request: KnowledgeSearchRequest,
    db: AsyncSession = Depends(get_db),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
) -> ResponseBase:
    """
    知识库语义检索

    基于向量相似度搜索相关文档切片，可通过 kb_id 限定知识库
    """
    results = await knowledge_service.search(
        db=db,
        query=request.query,
        top_k=request.top_k,
        kb_id=request.kb_id,
    )
    return ResponseBase(data=[r.model_dump() for r in results])


@router.get("/documents", response_model=PaginatedResponse, summary="获取文档列表")
async def list_documents(
    kb_id: Optional[int] = Query(default=None, description="按知识库ID过滤"),
    source_type: Optional[str] = Query(default=None, description="按来源类型过滤"),
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页数量"),
    db: AsyncSession = Depends(get_db),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
) -> PaginatedResponse:
    """获取知识库文档列表（分页）"""
    docs, total = await knowledge_service.list_documents(
        db=db,
        kb_id=kb_id,
        source_type=source_type,
        page=page,
        page_size=page_size,
    )
    total_pages = (total + page_size - 1) // page_size

    return PaginatedResponse(
        data=[d.model_dump() for d in docs],
        pagination=PaginationMeta(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        ),
    )


@router.get(
    "/documents/{doc_id}",
    response_model=ResponseBase,
    summary="获取文档详情",
)
async def get_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
) -> ResponseBase:
    """获取指定文档的详细信息"""
    doc = await knowledge_service.get_document(db, doc_id)
    if not doc:
        return ResponseBase(code=404, message="文档不存在")
    return ResponseBase(data=doc.model_dump())


@router.put(
    "/documents/{doc_id}",
    response_model=ResponseBase,
    summary="更新文档信息",
)
async def update_document(
    doc_id: int,
    update_data: KnowledgeDocumentUpdate,
    db: AsyncSession = Depends(get_db),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
) -> ResponseBase:
    """更新文档元数据（标题、描述、标签）"""
    result = await knowledge_service.update_document(db, doc_id, update_data)
    if not result:
        return ResponseBase(code=404, message="文档不存在")
    return ResponseBase(data=result.model_dump())


@router.delete(
    "/documents/{doc_id}",
    response_model=ResponseBase,
    summary="删除文档",
)
async def delete_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    knowledge_service: KnowledgeService = Depends(get_knowledge_service),
) -> ResponseBase:
    """删除文档及其所有切片"""
    success = await knowledge_service.delete_document(db, doc_id)
    if not success:
        return ResponseBase(code=404, message="文档不存在")
    return ResponseBase(message="文档已删除")