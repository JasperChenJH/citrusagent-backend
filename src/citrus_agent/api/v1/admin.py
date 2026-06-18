"""
管理后台 API 路由
提供问答日志查看、坏案例纠错、知识库统计等管理接口
所有数据持久化到 MySQL
"""

import json
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.citrus_agent.core.database import get_db
from src.citrus_agent.pojo.db_models import ChatLog, Document, KnowledgeBase
from src.citrus_agent.common.common import ResponseBase, PaginatedResponse, PaginationMeta
from src.citrus_agent.services.logger import logger

router = APIRouter(prefix="/admin", tags=["管理后台"])


@router.get("/logs", response_model=PaginatedResponse, summary="获取问答日志")
async def get_chat_logs(
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=20, ge=1, le=100, description="每页数量"),
    keyword: Optional[str] = Query(default=None, description="关键词搜索"),
    is_corrected: Optional[bool] = Query(default=None, description="是否已纠错"),
    db: AsyncSession = Depends(get_db),
) -> PaginatedResponse:
    """
    获取问答日志列表

    支持按关键词搜索和纠错状态过滤
    """
    query = select(ChatLog)
    count_query = select(func.count(ChatLog.id))

    # 关键词过滤
    if keyword:
        query = query.where(ChatLog.question.like(f"%{keyword}%"))
        count_query = count_query.where(ChatLog.question.like(f"%{keyword}%"))

    # 纠错状态过滤
    if is_corrected is not None:
        query = query.where(ChatLog.is_corrected == is_corrected)
        count_query = count_query.where(ChatLog.is_corrected == is_corrected)

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    result = await db.execute(
        query.order_by(ChatLog.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    logs = list(result.scalars().all())

    total_pages = (total + page_size - 1) // page_size

    # 转换为字典列表
    log_dicts = []
    for log in logs:
        log_dicts.append({
            "id": log.id,
            "conversation_id": log.conversation_id,
            "question": log.question,
            "answer": log.answer,
            "sources": json.loads(log.sources) if log.sources else [],
            "tool_used": log.tool_used,
            "is_corrected": log.is_corrected,
            "correction": log.correction,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        })

    return PaginatedResponse(
        data=log_dicts,
        pagination=PaginationMeta(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        ),
    )


@router.post(
    "/logs/{log_id}/correct",
    response_model=ResponseBase,
    summary="纠错问答记录",
)
async def correct_chat_log(
    log_id: int,
    correction: str,
    db: AsyncSession = Depends(get_db),
) -> ResponseBase:
    """
    对坏案例进行人工纠错

    Args:
        log_id: 日志ID
        correction: 纠错内容
    """
    result = await db.execute(select(ChatLog).where(ChatLog.id == log_id))
    log = result.scalar_one_or_none()

    if not log:
        return ResponseBase(code=404, message="日志记录不存在")

    log.is_corrected = True
    log.correction = correction
    await db.flush()

    return ResponseBase(message="纠错成功")


@router.get("/stats", response_model=ResponseBase, summary="知识库统计信息")
async def get_stats(
    db: AsyncSession = Depends(get_db),
) -> ResponseBase:
    """获取知识库和系统的统计信息"""
    try:
        # 知识库数量
        kb_count_result = await db.execute(select(func.count(KnowledgeBase.id)))
        kb_count = kb_count_result.scalar() or 0

        # 文档数量
        doc_count_result = await db.execute(select(func.count(Document.id)))
        doc_count = doc_count_result.scalar() or 0

        # 各状态文档数量
        status_result = await db.execute(
            select(Document.status, func.count(Document.id))
            .group_by(Document.status)
        )
        status_counts = {row[0]: row[1] for row in status_result.all()}

        # 日志数量
        log_count_result = await db.execute(select(func.count(ChatLog.id)))
        log_count = log_count_result.scalar() or 0

        # 已纠错数量
        corrected_result = await db.execute(
            select(func.count(ChatLog.id)).where(ChatLog.is_corrected == True)
        )
        corrected_count = corrected_result.scalar() or 0

        # Qdrant 信息
        qdrant_info = {}
        try:
            from src.citrus_agent.vectorstores.qdrant import QdrantStore
            qdrant = QdrantStore()
            qdrant_info = qdrant.client.get_collection(qdrant.collection_name).dict() if qdrant.collection_name else {}
        except Exception:
            pass

        return ResponseBase(data={
            "knowledge_bases": {
                "total": kb_count,
            },
            "documents": {
                "total": doc_count,
                "by_status": status_counts,
            },
            "chat_logs": {
                "total": log_count,
                "corrected": corrected_count,
            },
            "qdrant": qdrant_info,
        })

    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return ResponseBase(
            code=500,
            message=f"获取统计信息失败: {str(e)}",
        )


async def add_chat_log(
    db: AsyncSession,
    conversation_id: str,
    question: str,
    answer: str,
    sources: list[dict] | None = None,
    tool_used: str | None = None,
) -> None:
    """
    添加问答日志到 MySQL

    Args:
        db: 数据库会话
        conversation_id: 会话ID
        question: 用户提问
        answer: 模型回答
        sources: 参考资料列表
        tool_used: 使用的工具名称
    """
    log = ChatLog(
        conversation_id=conversation_id,
        question=question,
        answer=answer,
        sources=json.dumps(sources, ensure_ascii=False) if sources else None,
        tool_used=tool_used,
    )
    db.add(log)
    await db.flush()