"""MySQL ORM 模型。

本文件只定义知识库和文档元数据表，不保存向量。向量和 chunk payload 保存在 Qdrant。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy ORM 基类。"""


class KnowledgeBaseModel(Base):
    """knowledge_bases 表：一个知识库包含多个文档。"""

    __tablename__ = "knowledge_bases"
    __table_args__ = {"comment": "知识库表：一个知识库包含多个文档"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="知识库主键 ID")
    name: Mapped[str] = mapped_column(String(255), nullable=False, comment="知识库名称，前端列表展示使用")
    description: Mapped[str | None] = mapped_column(Text, nullable=True, comment="知识库描述，可为空")
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        comment="更新时间",
    )

class DocumentModel(Base):
    """documents 表：保存原始文件路径和入库状态，不保存向量。"""

    __tablename__ = "documents"
    __table_args__ = {"comment": "文档表：保存原始文件路径和入库状态，不保存向量"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="文档主键 ID")
    kb_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        index=True,
        comment="所属知识库 ID，对应 knowledge_bases.id",
    )
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False, comment="用户上传时的原始文件名")
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False, comment="服务器本地保存路径")
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="文件大小，单位 byte")
    file_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True, comment="文件内容 hash")
    mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True, comment="文件 MIME 类型")
    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="pending",
        server_default="pending",
        index=True,
        comment="文档状态：pending/processing/ready/failed/deleted",
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
        comment="成功写入 Qdrant 的 chunk 数量",
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="入库失败原因")
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        comment="更新时间",
    )
