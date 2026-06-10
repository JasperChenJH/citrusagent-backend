"""向量数据库适配目录。

职责：
    1. 连接 Qdrant。
    2. 管理 collection、向量字段、metadata 字段。
    3. 提供相似度检索、批量入库、删除索引等方法。

建议文件：
    - qdrant.py：Qdrant 连接与检索封装。
    - embeddings.py：Embedding 模型封装。

注意：
    向量库细节不要暴露给 API 层，由 RAG 或 service 层间接调用。
"""
