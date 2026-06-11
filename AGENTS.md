# CitrusAgent 项目指令

## 项目背景

本项目是橘子知识库问答后端，主要包含文档解析、知识入库、向量检索和 RAG 问答相关模块。

当前 RAG 入库流程以 MySQL 的 `documents` 表记录为入口：后端从数据库查出文档实体后，传给 RAG 模块；RAG 模块根据 `stored_path` 读取文件、解析、分块、生成 embedding，并写入 Qdrant。

RAG 服务模块不直接连接 MySQL，不负责创建表，也不更新 `documents.status`、`chunk_count`、`error_message`。这些数据库状态由后端服务根据 RAG 返回结果自行更新。本地脚本可以连接 MySQL，用来模拟后端正式流程。

## 开发环境

默认使用 Windows + PowerShell。

运行 Python 和测试时优先使用 conda 环境：

```powershell
conda run -n agent-dev python ...
```

常用验证命令：

```powershell
conda run -n agent-dev python -m py_compile src\citrus_agent\pojo\knowledge.py src\citrus_agent\services\knowledge_service.py src\citrus_agent\vectorstores\qdrant.py src\citrus_agent\rag\retriever.py
conda run -n agent-dev python -m pytest -q tests\test_knowledge_service.py tests\test_retriever.py tests\test_document_parser.py --basetemp=.pytest_tmp_agent_dev
```

测试后如产生 `.pytest_tmp_agent_dev`，可以删除。

## 代码边界

`api/` 只放 HTTP 接口，不直接操作 Qdrant、数据库或模型。

`services/knowledge_service.py` 负责知识入库编排，正式入口为
`KnowledgeIngestor.ingest_document(document: DocumentEntity)`，包括：

- 接收 `DocumentEntity`
- 根据 `stored_path` 解析文件
- 文本分块
- 构造 Qdrant payload
- 调用 embedding
- 写入 Qdrant
- 返回 `DocumentIngestResult`

`services/document_parser.py` 只负责文档解析，不做分块、不写 Qdrant。

`vectorstores/qdrant.py` 只封装 Qdrant collection、upsert、delete、search 等向量库操作。

`vectorstores/embeddings.py` 只封装 embedding 生成逻辑。

`rag/retriever.py` 只负责检索，不负责生成最终回答。
当前检索只做向量召回，不做 rerank、不做规则加权、不调用 LLM。

## MySQL 文档实体约定

RAG 入库入口使用 `DocumentEntity`，对应 MySQL `documents` 表。

关键字段：

- `id` 对应 Qdrant payload 的 `document_id`
- `kb_id` 对应 Qdrant payload 的 `kb_id`
- `original_filename` 对应 Qdrant payload 的 `file_name`
- `stored_path` 是本地文件路径，解析器根据它读取文件

入库返回 `DocumentIngestResult`：

- `document_id`
- `kb_id`
- `success`
- `chunk_count`
- `error_message`

后端根据该返回对象更新 MySQL 文档状态。

## Qdrant Payload 约定

第一版 payload 保持最小可用，不要随意加字段。

当前字段：

- `chunk_id`
- `kb_id`
- `document_id`
- `file_name`
- `page`
- `chunk_index`
- `text`

删除文档时，按 `document_id` 删除 Qdrant points。

检索时，优先按 `kb_id` 过滤，避免不同知识库互相串数据。

拼接 prompt 时，使用 payload 里的 `text` 字段。

## 配置约定

配置从 `.env` 读取，不要把真实密钥写进代码。

当前 embedding 使用 API：

```env
EMBEDDING_PROVIDER=api
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024
EMBEDDING_VECTOR_SIZE=1024
```

Qdrant 地址从 `QDRANT_URL` 读取。

如果 Qdrant collection 已经用其他维度创建过，切换 embedding 维度后必须重建 collection。

## 代码风格

Python 代码尽量使用简单、直观的写法。

新增模块、类和关键方法要写中文 docstring 或注释，说明输入、输出和职责。

不要把业务逻辑堆到 `common/`。

不要在 RAG 模块里直接写 MySQL 用户名、密码或连接逻辑，除非用户明确要求改变边界。

修改已有文件前先看当前实现，避免覆盖用户或其他模块的改动。
