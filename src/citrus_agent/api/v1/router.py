"""
V1 版本路由聚合
将所有子路由注册到统一的 v1 路由下
"""

from fastapi import APIRouter

from src.citrus_agent.api.v1.auth import router as auth_router
from src.citrus_agent.api.v1.chat import router as chat_router
from src.citrus_agent.api.v1.knowledge import router as knowledge_router
from src.citrus_agent.api.v1.tool import router as tool_router
from src.citrus_agent.api.v1.admin import router as admin_router

# V1 版本总路由
router = APIRouter()

# 注册子路由（auth 不需要登录保护）
router.include_router(auth_router)
router.include_router(chat_router)
router.include_router(knowledge_router)
router.include_router(tool_router)
router.include_router(admin_router)
