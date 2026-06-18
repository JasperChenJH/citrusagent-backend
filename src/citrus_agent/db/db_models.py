"""
MySQL ORM 模型定义
对应数据库表结构
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.citrus_agent.core.database import Base


class KnowledgeBase(Base):
    """知识库表"""
    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="知识库名称")
    description: Mapped[str | None] = mapped_column(Text, nullable=True, comment="描述")
    icon: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="图标标识")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, comment="是否启用")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), comment="创建时间"
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime, onupdate=func.now(), comment="更新时间"
    )


class Document(Base):
    """文档表"""
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True, comment="所属知识库ID")
    title: Mapped[str] = mapped_column(String(200), nullable=False, comment="文档标题")
    original_filename: Mapped[str | None] = mapped_column(String(500), nullable=True, comment="原始文件名")
    stored_path: Mapped[str | None] = mapped_column(String(1000), nullable=True, comment="服务器存储路径")
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="文件大小(字节)")
    file_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="文件SHA256")
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="MIME类型")
    source_type: Mapped[str] = mapped_column(
        String(50), default="general",
        comment="来源类型: general/literature/policy/structured/external"
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True, comment="文档描述")
    tags: Mapped[str | None] = mapped_column(String(500), nullable=True, comment="标签，逗号分隔")
    status: Mapped[str] = mapped_column(
        String(20), default="pending",
        comment="状态: pending/processing/ready/failed"
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, comment="切片数量")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True, comment="错误信息")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), comment="创建时间"
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime, onupdate=func.now(), comment="更新时间"
    )


class User(Base):
    """用户表"""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, comment="用户名")
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False, comment="密码哈希")
    nickname: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="昵称")
    role: Mapped[str] = mapped_column(
        String(50), default="user",
        comment="角色: admin/user/guest，预留权限控制"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, comment="是否启用")
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="最后登录时间")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), comment="创建时间"
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime, onupdate=func.now(), comment="更新时间"
    )


class ChatLog(Base):
    """问答日志表"""
    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True, comment="会话ID")
    question: Mapped[str] = mapped_column(Text, nullable=False, comment="用户提问")
    answer: Mapped[str] = mapped_column(Text, nullable=False, comment="模型回答")
    sources: Mapped[str | None] = mapped_column(Text, nullable=True, comment="参考资料(JSON)")
    tool_used: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="使用的工具")
    is_corrected: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否已纠错")
    correction: Mapped[str | None] = mapped_column(Text, nullable=True, comment="纠错内容")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), comment="创建时间"
    )
