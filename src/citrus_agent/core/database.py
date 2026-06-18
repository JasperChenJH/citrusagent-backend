"""
数据库连接与会话管理模块
使用 SQLAlchemy 2.0 异步引擎
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.citrus_agent.core.settings import get_settings
from src.citrus_agent.services.logger import logger


class Base(DeclarativeBase):
    """ORM 基类"""
    pass


# 全局引擎和会话工厂（延迟初始化）
_engine = None
_async_session_factory = None


def _init_engine():
    """初始化数据库引擎（首次调用时执行）"""
    global _engine, _async_session_factory
    if _engine is not None:
        return

    settings = get_settings()
    _engine = create_async_engine(
        settings.MYSQL_DATABASE_URL,
        pool_size=settings.MYSQL_POOL_SIZE,
        max_overflow=settings.MYSQL_MAX_OVERFLOW,
        pool_recycle=settings.MYSQL_POOL_RECYCLE,
        echo=settings.DEBUG,
    )
    _async_session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    logger.info("MySQL 异步引擎初始化完成")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    获取数据库会话（FastAPI 依赖注入用）

    用法:
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...
    """
    if _async_session_factory is None:
        _init_engine()

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    初始化数据库（建表）
    在应用启动时调用
    """
    if _engine is None:
        _init_engine()

    # 导入所有模型，确保 Base.metadata 包含所有表定义
    import src.citrus_agent.db.db_models  # noqa: F401

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("MySQL 数据库表初始化完成")


async def close_db() -> None:
    """关闭数据库连接池"""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        logger.info("MySQL 连接池已关闭")