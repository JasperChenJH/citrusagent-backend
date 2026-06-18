"""
全局配置管理模块
使用 pydantic-settings 从环境变量 / .env 文件加载配置
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "CitrusAgent"  # 设置默认值

    # ==================== 应用基础配置 ====================
    APP_NAME: str = "Orange-RAG"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"

    # ==================== Qdrant 向量数据库配置 ====================
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "orange_knowledge"

    # ==================== LLM 大模型配置 ====================
    LLM_PROVIDER: str = "openai"  # openai / zhipu / deepseek 等
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL_NAME: str = ""
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # ==================== Embedding 模型配置 ====================
    EMBEDDING_PROVIDER: str = "openai"  # openai / local / zhipu
    EMBEDDING_API_KEY: Optional[str] = None
    EMBEDDING_BASE_URL: Optional[str] = None
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    embedding_vector_size: int = 1024

    # ==================== 文档切片配置 ====================
    CHUNK_SIZE: int = 500  # 切片最大字符数
    CHUNK_OVERLAP: int = 50  # 切片重叠字符数
    CHUNK_SEPARATOR: str = "\n\n"  # 切片分隔符

    # ==================== RAG 检索配置 ====================
    RAG_TOP_K: int = 5  # 检索返回的文档数量
    RAG_SCORE_THRESHOLD: float = 0.7  # 相似度阈值，低于此值不返回

    # ==================== 对话历史配置 ====================
    CHAT_HISTORY_MAX_TURNS: int = 10  # 多轮对话保留的最大轮数

    # ==================== 文件上传配置 ====================
    UPLOAD_DIR: str = "uploads"  # 这里尽量使用绝对路径进行存储
    MAX_UPLOAD_SIZE_MB: int = 50  # 单文件最大 MB，可实际需求调整
    ALLOWED_EXTENSIONS: list[str] = [
        ".pdf", ".docx", ".xlsx", ".csv", ".md", ".txt" #与rag模块支持的文件类型保持一致
    ]

    # ==================== MySQL 数据库配置 ====================
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "orange_rag"  #后续根据实际情况修改
    MYSQL_POOL_SIZE: int = 10
    MYSQL_MAX_OVERFLOW: int = 20
    MYSQL_POOL_RECYCLE: int = 3600

    @property
    def MYSQL_DATABASE_URL(self) -> str:
        """异步数据库连接 URL"""
        return (
            f"mysql+aiomysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            f"?charset=utf8mb4"
        )

    @property
    def MYSQL_DATABASE_URL_SYNC(self) -> str:
        """同步数据库连接 URL（用于建表等操作）"""
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            f"?charset=utf8mb4"
        )

    # ==================== 日志配置 ====================
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"

    # ==================== 认证与安全配置 ====================
    JWT_SECRET_KEY: str = "change-me-to-a-random-secret-key"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # Token 有效期，默认 24 小时
    CAPTCHA_EXPIRE_SECONDS: int = 300  # 验证码有效期，默认 5 分钟
    CAPTCHA_LENGTH: int = 4  # 验证码字符数


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置单例（带缓存）"""
    return Settings()
