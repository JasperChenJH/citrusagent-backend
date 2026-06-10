"""项目配置读取模块。

本文件只负责读取环境变量和 `.env` 文件，不写业务逻辑。

使用方式：
    from src.citrus_agent.core.config import settings

    print(settings.app_name)

团队约定：
    1. API Key、数据库地址、Qdrant 地址等配置都从 `.env` 读取。
    2. 不要把真实密钥写死到代码里。
    3. 新增配置项时，要在 `.env.example` 和 README 中同步说明。
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置对象。

    字段名使用小写下划线，环境变量使用大写下划线。
    例如：`deepseek_api_key` 会自动读取 `.env` 中的 `DEEPSEEK_API_KEY`。
    """

    app_name: str = Field(default="CitrusAgent", description="项目名称")
    app_env: str = Field(default="local", description="运行环境，例如 local、dev、prod")
    project_root: Path = Field(
        default=Path("G:/py_workplace/CitrusAgent"),
        description="当前项目根目录",
    )

    api_v1_prefix: str = Field(default="/api/v1", description="v1 接口统一前缀")

    deepseek_api_key: str = Field(default="", description="DeepSeek API Key")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API 地址",
    )
    deepseek_model: str = Field(default="deepseek-chat", description="DeepSeek 模型名称")

    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant 服务地址")
    qdrant_api_key: str = Field(default="", description="Qdrant API Key，本地可为空")
    qdrant_collection: str = Field(
        default="agriculture_knowledge",
        description="农业知识库向量集合名称",
    )

    database_url: str = Field(
        default="sqlite:///./citrus_agent.db",
        description="关系型数据库连接地址",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """读取并缓存项目配置。

    Returns:
        Settings: 当前运行环境下的配置对象。
    """

    return Settings()


settings = get_settings()
