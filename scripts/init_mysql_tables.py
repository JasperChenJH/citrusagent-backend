"""初始化本地 MySQL 数据库和知识库相关表。

用法：
    conda run -n agent-dev python scripts/init_mysql_tables.py

脚本会连接本地 MySQL，创建 `CitrusAgent` 数据库，并按项目约定 DDL 创建
`knowledge_bases` 和 `documents` 两张表。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, text

from src.citrus_agent.core.config import settings

CREATE_KNOWLEDGE_BASES_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_bases
(
    id          BIGINT AUTO_INCREMENT COMMENT '知识库主键 ID' PRIMARY KEY,
    name        VARCHAR(255)                       NOT NULL COMMENT '知识库名称，前端列表展示使用',
    description TEXT                               NULL COMMENT '知识库描述，可为空',
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT '创建时间',
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
)
COMMENT '知识库表：一个知识库包含多个文档'
"""

CREATE_DOCUMENTS_SQL = """
CREATE TABLE IF NOT EXISTS documents
(
    id                BIGINT AUTO_INCREMENT COMMENT '文档主键 ID' PRIMARY KEY,
    kb_id             BIGINT                                NOT NULL COMMENT '所属知识库 ID，对应 knowledge_bases.id',
    original_filename VARCHAR(512)                          NOT NULL COMMENT '用户上传时的原始文件名',
    stored_path       VARCHAR(1024)                         NOT NULL COMMENT '服务器本地保存路径，worker 根据该路径读取文件',
    file_size         BIGINT                                NOT NULL COMMENT '文件大小，单位 byte',
    file_hash         VARCHAR(128)                          NOT NULL COMMENT '文件内容 hash，用于去重和重新入库判断',
    mime_type         VARCHAR(128)                          NULL COMMENT '文件 MIME 类型，可能为空',
    status            VARCHAR(30) DEFAULT 'pending'         NOT NULL COMMENT '文档状态：pending/processing/ready/failed/deleted',
    chunk_count       INT         DEFAULT 0                 NOT NULL COMMENT '成功写入 Qdrant 的 chunk 数量',
    error_message     TEXT                                  NULL COMMENT '入库失败原因，便于前端展示',
    created_at        DATETIME    DEFAULT CURRENT_TIMESTAMP NOT NULL COMMENT '创建时间',
    updated_at        DATETIME    DEFAULT CURRENT_TIMESTAMP NOT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
)
COMMENT '文档表：保存原始文件路径和入库状态，不保存向量'
"""

CREATE_INDEX_SQL_LIST = [
    "CREATE INDEX idx_documents_hash ON documents (file_hash)",
    "CREATE INDEX idx_documents_kb_id ON documents (kb_id)",
    "CREATE INDEX idx_documents_status ON documents (status)",
]


def create_database_if_needed() -> None:
    """先连接 MySQL 服务，再创建 CitrusAgent 数据库。"""

    server_url = (
        f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/?charset=utf8mb4"
    )
    server_engine = create_engine(server_url, pool_pre_ping=True)
    with server_engine.connect() as connection:
        connection.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        )
        connection.commit()


def main() -> int:
    """脚本入口。"""

    create_database_if_needed()
    database_engine = create_engine(settings.database_url, pool_pre_ping=True)
    with database_engine.connect() as connection:
        connection.execute(text(CREATE_KNOWLEDGE_BASES_SQL))
        connection.execute(text(CREATE_DOCUMENTS_SQL))
        for sql in CREATE_INDEX_SQL_LIST:
            try:
                connection.execute(text(sql))
            except Exception:
                # 索引已存在时 MySQL 会报错，忽略即可。
                continue
        connection.commit()
    print(f"MySQL 数据库和表已准备好：{settings.mysql_database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
