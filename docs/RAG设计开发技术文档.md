# 橘子知识库 RAG 模块设计开发技术文档

## 1. 文档目的

本文档用于说明橘子知识库项目中 RAG 模块的设计思路、核心流程、数据结构、接口使用方式和后续优化方向。

当前 RAG 模块主要负责知识库文档的向量化入库与向量检索，不负责 HTTP 接口、不负责 MySQL 表维护、不负责调用大模型生成最终回答。后端服务只需要把 MySQL 中的文档记录封装成实体传给 RAG 模块，即可完成文档解析、分块、向量化和 Qdrant 写入。

## 2. 模块定位

### 2.1 RAG 模块负责的内容

- 根据后端传入的 `DocumentEntity` 读取本地文件。
- 支持多格式文档解析，包括 `pdf`、`docx`、`xlsx`、`csv`、`md`、`txt`。
- 将解析后的文本切分成适合向量检索的 chunk。
- 为每个 chunk 构造最小可用 payload。
- 调用 Embedding API 生成向量。
- 将 chunk 向量和 payload 写入 Qdrant。
- 支持按 `document_id` 删除某个文档的全部向量。
- 支持按 `kb_id` 删除某个知识库的全部向量。
- 支持按 `kb_id` 过滤检索，也支持不加过滤的全库检索。
- 将检索结果封装成统一的返回实体，方便后续 LLM 层拼接 prompt。

### 2.2 RAG 模块不负责的内容

- 不直接连接 MySQL。
- 不创建或修改 MySQL 表结构。
- 不更新 `documents.status`、`chunk_count`、`error_message` 等字段。
- 不提供 FastAPI 路由。
- 不处理用户登录、权限校验、文件上传接口。
- 不调用聊天模型生成最终回答。
- 当前版本不做 rerank 重排。
- 当前版本不做业务规则加权排序。

## 3. 总体架构

当前 RAG 模块采用分层设计，后端通过 `RAGApi` 统一调用内部能力。

```text
后端业务层
    |
    | 传入 DocumentEntity / query_text / kb_id
    v
RAGApi
    |
    |-- KnowledgeIngestor
    |       |-- DocumentParser
    |       |-- EmbeddingProvider
    |       |-- QdrantStore
    |
    |-- CitrusRetriever
            |-- EmbeddingProvider
            |-- QdrantStore
```

模块说明：

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 后端调用门面 | `src/citrus_agent/rag/rag_api.py` | 给后端提供统一调用入口 |
| 检索器 | `src/citrus_agent/rag/retriever.py` | 根据问题生成 embedding 并检索 Qdrant |
| 入库编排 | `src/citrus_agent/services/knowledge_service.py` | 串联解析、分块、payload、embedding、写入 |
| 文档解析 | `src/citrus_agent/services/document_parser.py` | 解析多格式文档 |
| Embedding 封装 | `src/citrus_agent/vectorstores/embeddings.py` | 封装本地或 API embedding 模型 |
| Qdrant 封装 | `src/citrus_agent/vectorstores/qdrant.py` | 负责 collection、upsert、search、delete |
| 数据对象 | `src/citrus_agent/pojo/knowledge.py` | 定义入库和检索的数据实体 |

## 4. 数据存储设计

### 4.1 MySQL 存储边界

MySQL 用于保存知识库和文档元数据，不保存向量。

当前设计中有两张核心表：

- `knowledge_bases`：知识库表。
- `documents`：文档表。

其中：

- `knowledge_bases.id` 对应 Qdrant payload 中的 `kb_id`。
- `documents.id` 对应 Qdrant payload 中的 `document_id`。
- `documents.stored_path` 用于让 RAG 模块读取本地文件。
- `documents.status`、`chunk_count`、`error_message` 由后端根据 RAG 返回结果更新。

RAG 模块不直接操作 MySQL，这样可以让数据库事务、状态更新和权限控制都留在后端业务层。

### 4.2 Qdrant 存储设计

Qdrant 用于保存 chunk 向量和最小 payload。

当前使用单集合：

```text
orange_knowledge
```

当前版本只使用 dense vector，不使用 sparse vector，不接 rerank 模型。

Qdrant payload 字段如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `chunk_id` | `str` | chunk 的业务 ID |
| `kb_id` | `int` | 所属知识库 ID，对应 `knowledge_bases.id` |
| `document_id` | `int` | 所属文档 ID，对应 `documents.id` |
| `file_name` | `str` | 原始文件名 |
| `page` | `int | None` | 页码，部分文本类文件没有页码 |
| `chunk_index` | `int` | 文档内的 chunk 序号 |
| `text` | `str` | chunk 正文，用于后续拼接 prompt |

第一版 payload 保持最小可用，主要满足三个需求：

- 根据 `document_id` 删除单个文档的全部向量。
- 根据 `kb_id` 隔离或删除某个知识库的全部向量。
- 检索后直接取 `text` 拼接到 prompt。

## 5. 核心数据对象

### 5.1 DocumentEntity

`DocumentEntity` 表示后端从 MySQL `documents` 表查出的文档记录。

```python
DocumentEntity(
    id=12,
    kb_id=3,
    original_filename="砂糖橘溃疡病.md",
    stored_path="文件保存路径",
    file_size=1024,
    file_hash="文件 hash",
    mime_type="text/markdown",
    status="pending",
)
```

该实体是 RAG 入库的唯一正式入口参数。

### 5.2 DocumentIngestResult

`DocumentIngestResult` 表示单个文档入库结果。

```python
DocumentIngestResult(
    document_id=12,
    kb_id=3,
    success=True,
    chunk_count=8,
    error_message="",
)
```

后端根据该结果更新 MySQL：

- `success=True`：更新文档状态为 `ready`。
- `success=False`：更新文档状态为 `failed`，并保存 `error_message`。

### 5.3 SearchResult

`SearchResult` 表示单个命中的 chunk。

```python
SearchResult(
    chunk_id="3_12_0",
    content="砂糖橘溃疡病在雨季高发...",
    score=0.91,
    payload={...},
)
```

其中：

- `content` 是 chunk 文本。
- `score` 是 Qdrant 返回的相似度分数。
- `payload` 可用于展示来源文件、页码、chunk 序号等信息。

### 5.4 DocumentSearchResult

`DocumentSearchResult` 表示一次检索的整体结果。

```python
DocumentSearchResult(
    kb_id=3,
    query_text="砂糖橘溃疡病怎么防？",
    top_k=5,
    score=0.91,
    chunks=[...],
)
```

其中：

- `kb_id` 是本次过滤的知识库 ID；如果不加过滤，则为 `None`。
- `score` 是本次检索结果中最高的 chunk 分数。
- `chunks` 是实际命中的片段列表。

## 6. 文档入库流程设计

### 6.1 入库时序

```text
后端查询 MySQL documents 表
    -> 组装 DocumentEntity
    -> 调用 RAGApi.ingest_document(document)
    -> 检查文件是否存在
    -> 根据后缀选择解析器
    -> 提取文本和页码
    -> 文本清洗
    -> 文本分块
    -> 构造 chunk_id 和 payload
    -> 调用 EmbeddingProvider 生成向量
    -> 按 document_id 删除旧向量
    -> 批量写入 Qdrant
    -> 返回 DocumentIngestResult
    -> 后端更新 MySQL documents 表
```

### 6.2 分块策略

当前使用基于字符长度的简单分块策略。

配置项：

```text
CHUNK_SIZE=500
CHUNK_OVERLAP=80
```

设计原因：

- 实现简单，便于第一版稳定上线。
- 适合中文农业资料的初步切分。
- overlap 可以降低重要上下文被切断的概率。

后续可以升级为：

- 按 Markdown 标题切分。
- 按段落切分。
- 按语义边界切分。
- 按 token 数切分。

### 6.3 重新入库策略

同一个文档重新入库时，默认先删除旧向量，再写入新向量。

删除条件：

```text
payload.document_id == documents.id
```

这样可以避免旧版本内容残留在 Qdrant 中。

## 7. 检索流程设计

### 7.1 按知识库过滤检索

如果用户在某个知识库内提问，后端应传入 `kb_id`。

```python
from src.citrus_agent.rag.rag_api import RAGApi

api = RAGApi()

result = api.search(
    query_text="砂糖橘溃疡病怎么防？",
    top_k=5,
    kb_id=3,
)
```

内部构造 Qdrant filter：

```python
{"kb_id": 3}
```

检索流程：

```text
用户问题
    -> 生成 query embedding
    -> Qdrant 向量检索
    -> 按 kb_id 强过滤
    -> 返回 top_k 个 chunk
    -> 封装成 DocumentSearchResult
```

### 7.2 不加过滤检索

如果是后台测试、全局搜索或临时调试，可以不传 `kb_id`。

```python
result = api.search(
    query_text="砂糖橘溃疡病怎么防？",
    top_k=5,
)
```

不传 `kb_id` 时，会在整个 Qdrant collection 中检索。

正式用户问答场景中，如果用户已经选择了知识库，建议始终传入 `kb_id`。

### 7.3 当前检索策略

当前版本只使用向量相似度检索：

```text
query_text -> embedding -> Qdrant dense vector search -> top_k chunks
```

当前没有：

- rerank 模型。
- 关键词加权。
- 品种、病虫害、地区等业务字段过滤。
- 多路召回。
- LLM 改写问题。

这样设计是为了先保证入库、删除、过滤、检索和 prompt 拼接链路稳定。

## 8. 删除流程设计

### 8.1 删除单个文档向量

后端删除某个文档时，调用：

```python
api.delete_document_vectors(document_id=12)
```

Qdrant 删除条件：

```text
payload.document_id == 12
```

该操作只删除向量库中的 chunk，不删除 MySQL 文档记录。

### 8.2 删除整个知识库向量

后端删除某个知识库时，调用：

```python
api.delete_knowledge_base_vectors(kb_id=3)
```

Qdrant 删除条件：

```text
payload.kb_id == 3
```

该操作只删除向量库中的 points，不删除 MySQL 中的 `knowledge_bases` 或 `documents`。

## 9. 后端对接方式

### 9.1 初始化 RAGApi

建议后端在服务启动时初始化一次：

```python
from src.citrus_agent.rag.rag_api import RAGApi

rag_api = RAGApi()
```

不要在每次请求里频繁创建对象，避免重复初始化 embedding client 和 Qdrant client。

### 9.2 文档入库

```python
result = rag_api.ingest_document(document)
```

`document` 必须是 `DocumentEntity`。

### 9.3 删除文档向量

```python
rag_api.delete_document_vectors(document_id=12)
```

### 9.4 删除知识库向量

```python
rag_api.delete_knowledge_base_vectors(kb_id=3)
```

### 9.5 检索

```python
result = rag_api.search(
    query_text="砂糖橘溃疡病怎么防？",
    top_k=5,
    kb_id=3,
)
```

获取 prompt 上下文：

```python
context = "\n\n".join(chunk.content for chunk in result.chunks)
```

获取引用来源：

```python
for chunk in result.chunks:
    file_name = chunk.payload.get("file_name")
    page = chunk.payload.get("page")
    chunk_index = chunk.payload.get("chunk_index")
```

## 10. 配置项说明

相关配置在 `.env` 和 `src/citrus_agent/core/config.py` 中读取。

常用配置：

| 配置项 | 说明 |
| --- | --- |
| `QDRANT_URL` | Qdrant 服务地址 |
| `QDRANT_COLLECTION` | Qdrant collection 名称 |
| `QDRANT_DISTANCE` | 向量距离算法 |
| `EMBEDDING_PROVIDER` | embedding 提供方 |
| `EMBEDDING_BASE_URL` | embedding API 地址 |
| `EMBEDDING_MODEL` | embedding 模型名称 |
| `EMBEDDING_DIMENSIONS` | embedding 向量维度 |
| `EMBEDDING_VECTOR_SIZE` | Qdrant collection 向量维度 |
| `RETRIEVAL_TOP_K` | 默认召回数量 |
| `CHUNK_SIZE` | 分块大小 |
| `CHUNK_OVERLAP` | 分块重叠长度 |

注意：

- `EMBEDDING_DIMENSIONS`、`EMBEDDING_VECTOR_SIZE` 和 Qdrant collection 创建时的向量维度必须一致。
- 如果更换 embedding 模型导致维度变化，建议新建 collection 并重新入库。
- 文档中不要写真实 API Key 或数据库密码。

## 11. 测试设计

当前已覆盖的测试包括：

| 测试文件 | 覆盖内容 |
| --- | --- |
| `tests/test_document_parser.py` | Markdown、TXT、CSV 等文档解析 |
| `tests/test_knowledge_service.py` | 文档入库、分块、payload 构造、失败返回 |
| `tests/test_qdrant_store.py` | Qdrant 删除过滤条件 |
| `tests/test_rag_api.py` | 后端门面方法、删除、检索返回结构 |
| `tests/test_retriever.py` | 检索器读取 payload 文本和过滤参数 |

本地运行不连接外部服务的测试：

```powershell
G:\worktool\Anaconda\envs\agent-dev\python.exe -m pytest -q tests\test_document_parser.py tests\test_knowledge_service.py tests\test_qdrant_store.py tests\test_rag_api.py tests\test_retriever.py
```

真实入库测试会调用 embedding API 和 Qdrant，需要确认 `.env` 配置正确。

## 12. 当前设计优点

- 模块边界清晰：MySQL 由后端负责，Qdrant 由 RAG 模块负责。
- 接口简单：后端只需要调用 `RAGApi`。
- payload 简洁：只保留删除、过滤、引用和 prompt 拼接所需字段。
- 可替换性较好：EmbeddingProvider 封装后，后续可替换模型或服务。
- 删除逻辑明确：文档删除按 `document_id`，知识库删除按 `kb_id`。
- 检索链路稳定：第一版只做向量检索，减少复杂规则带来的不确定性。

## 13. 当前限制

- 分块策略还比较简单，没有按语义边界切分。
- 没有 rerank，长文档或相似内容较多时排序可能不够精确。
- 没有关键词召回，纯向量检索可能漏掉精确术语。
- payload 第一版没有保存病虫害、品种、地区、生长期等业务字段。
- 没有接 LLM 问题改写。
- 没有做多知识库权限控制，权限仍需后端保证。

## 14. 后续优化方向

后续服务器资源充足后，可以逐步优化：

1. 接入 rerank 模型，例如 `BAAI/bge-reranker-v2-m3`。
2. 增加业务 metadata，例如品种、病虫害、地区、生长期。
3. 引入 hybrid search，将向量检索和关键词检索结合。
4. 优化分块策略，按标题、段落或 token 数切分。
5. 增加召回结果去重，避免同一文档连续多个 chunk 内容过于重复。
6. 增加引用来源格式化，方便前端展示资料出处。
7. 增加检索日志，用于分析用户问题和命中效果。
8. 支持多 collection 或按租户隔离数据。

## 15. 总结

本 RAG 模块当前完成了橘子知识库的第一版核心能力：文档解析、文本分块、向量化入库、按文档删除、按知识库删除、按知识库过滤检索和全库检索。

整体设计以稳定和清晰为优先目标，将 MySQL 元数据管理留给后端，将 Qdrant 向量管理封装在 RAG 模块内部。后端只需要传入 `DocumentEntity` 或检索问题，即可完成知识入库和知识召回。

后续可以在当前稳定链路上继续叠加 rerank、业务 metadata、多路召回和更精细的 prompt 拼接策略。
