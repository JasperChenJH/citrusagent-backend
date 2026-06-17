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

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT_PATH = Path(__file__).resolve().parents[3]
ENV_FILE_PATH = PROJECT_ROOT_PATH / ".env"


class Settings(BaseSettings):
    """全局配置对象。

    字段名使用小写下划线，环境变量使用大写下划线。
    例如：`deepseek_api_key` 会自动读取 `.env` 中的 `DEEPSEEK_API_KEY`。
    """

    app_name: str = Field(default="CitrusAgent", description="项目名称")
    app_env: str = Field(default="local", description="运行环境，例如 local、dev、prod")
    project_root: Path = Field(
        default=PROJECT_ROOT_PATH,
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
        default="orange_knowledge",
        description="橘子知识库向量集合名称",
    )
    qdrant_hybrid_collection: str = Field(
        default="orange_knowledge_hybrid",
        description="BGE-M3 hybrid 入库使用的 Qdrant 集合名称",
    )
    qdrant_distance: str = Field(
        default="Cosine",
        description="Qdrant 向量距离算法，第一版默认使用 Cosine",
    )
    qdrant_dense_vector_name: str = Field(default="dense", description="Qdrant dense 向量名称")
    qdrant_sparse_vector_name: str = Field(default="sparse", description="Qdrant sparse 向量名称")

    embedding_provider: str = Field(
        default="api",
        description="embedding 提供方，支持 api、local_bge、fixed",
    )
    embedding_api_key: str = Field(
        default="",
        description="embedding API Key；只用于向量化，不用于聊天模型",
    )
    embedding_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="embedding API 地址，默认使用阿里云 DashScope OpenAI 兼容模式",
    )
    embedding_model: str = Field(
        default="text-embedding-v4",
        description="embedding API 模型名称",
    )
    embedding_dimensions: int = Field(
        default=1024,
        description="embedding API 返回向量维度",
    )
    embedding_model_name: str = Field(
        default="text-embedding-v4",
        description="embedding 模型名称，兼容本地模型和 API 模型",
    )
    embedding_vector_size: int = Field(
        default=1024,
        description="embedding 向量维度，必须和 Qdrant collection 的向量维度一致",
    )
    bge_m3_url: str = Field(
        default="http://172.21.72.18:8001",
        description="BGE-M3 embedding 服务地址，返回 dense + sparse",
    )
    bge_m3_timeout: int = Field(default=60, description="BGE-M3 服务请求超时时间，单位秒")
    bge_m3_batch_size: int = Field(default=8, description="调用 BGE-M3 服务时的默认批大小")

    retrieval_top_k: int = Field(default=30, description="Qdrant 第一阶段召回数量")
    chunk_size: int = Field(default=500, description="知识片段目标字符数")
    chunk_overlap: int = Field(default=80, description="相邻知识片段重叠字符数")

    database_url: str = Field(
        default="mysql+pymysql://root:replace-me@localhost:3306/CitrusAgent?charset=utf8mb4",
        description="关系型数据库连接地址",
    )
    mysql_host: str = Field(default="localhost", description="MySQL 主机地址")
    mysql_port: int = Field(default=3306, description="MySQL 端口")
    mysql_user: str = Field(default="root", description="MySQL 用户名")
    mysql_password: str = Field(default="", description="MySQL 密码")
    mysql_database: str = Field(default="CitrusAgent", description="MySQL 数据库名")

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int) -> int:
        """校验分块重叠长度，避免出现负数。"""

        if value < 0:
            raise ValueError("CHUNK_OVERLAP 不能小于 0")
        return value

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
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
