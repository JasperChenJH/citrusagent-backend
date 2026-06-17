"""CitrusAgent FastAPI 应用入口。

启动方式：
    conda run -n agent-dev uvicorn src.citrus_agent.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI

from src.citrus_agent.api.v1.chat import router as chat_router
from src.citrus_agent.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="橘子知识库问答后端 — 文档解析、知识入库、向量检索、RAG 问答",
)

app.include_router(chat_router)


@app.get("/health")
def health_check():
    """健康检查接口。"""

    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}
