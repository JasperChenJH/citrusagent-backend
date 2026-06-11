"""MySQL 数据库连接。

本模块只提供 SQLAlchemy engine 和 session 工厂。RAG 入库逻辑默认不直接更新 MySQL，
但本地建表脚本和后端服务可以复用这里的连接配置。
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.citrus_agent.core.config import settings


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session():
    """创建一个数据库 session。

    调用方负责在使用完成后关闭 session。
    """

    return SessionLocal()
