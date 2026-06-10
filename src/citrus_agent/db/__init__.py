"""关系型数据库目录。

职责：
    1. 管理 MySQL、PostgreSQL 或 SQLite 等数据库连接。
    2. 放用户表、对话日志表、知识库元数据表等 ORM 模型。
    3. 提供事务、会话、迁移相关代码。

建议文件：
    - session.py：数据库连接和 session。
    - models.py：ORM 模型。
    - migrations/：数据库迁移脚本。
"""
