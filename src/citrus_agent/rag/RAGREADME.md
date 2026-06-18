# RAG 模块说明

本目录提供给后端调用的 RAG 入库、删除和检索能力。

当前版本只负责：

- 接收后端传入的 `DocumentEntity`
- 根据文档路径解析文件
- 文本分块
- 调用 embedding 模型生成向量
- 写入 Qdrant
- 调用服务器 BGE-M3 服务生成 dense + sparse 并写入 hybrid collection
- 从 Qdrant 检索相关 chunk
- 按 `document_id` 或 `kb_id` 删除 Qdrant 中的向量

当前版本不负责：

- 不连接 MySQL
- 不创建 MySQL 表
- 不更新 `documents.status`
- 不调用大模型生成最终回答
- 不做 rerank 重排
- 不做规则加权排序
- 不处理 HTTP 请求

## 一、后端调用入口

后端统一使用 `RAGApi`。

```python
from src.citrus_agent.rag.rag_api import RAGApi

api = RAGApi()
```

不要直接调用 `QdrantStore`、`KnowledgeIngestor`、`CitrusRetriever`，除非是在写内部测试。

## 二、文档入库流程

后端从 MySQL `documents` 表查询记录后，组装成 `DocumentEntity`，再传给 RAG 模块。

整体流程：

```text
后端查询 MySQL documents 表
    -> 组装 DocumentEntity
    -> api.ingest_document(document)
    -> 根据 document.stored_path 读取文件
    -> 解析文档
    -> 文本分块
    -> 生成最小 payload
    -> 调用 embedding API 生成向量
    -> 按 document_id 删除旧向量
    -> 写入新的 chunk 向量
    -> 返回 DocumentIngestResult
    -> 后端根据结果更新 MySQL documents 表
```

调用示例：

```python
from src.citrus_agent.pojo.knowledge import DocumentEntity
from src.citrus_agent.rag.rag_api import RAGApi

api = RAGApi()

document = DocumentEntity(
    id=12,
    kb_id=3,
    original_filename="砂糖橘溃疡病.md",
    stored_path=r"G:\py_workplace\CitrusAgent\tests\README.md",
    file_size=1024,
    file_hash="文件 hash",
    mime_type="text/markdown",
    status="pending",
)

result = api.ingest_document(document)
```

返回结果：

```python
DocumentIngestResult(
    document_id=12,
    kb_id=3,
    success=True,
    chunk_count=8,
    error_message="",
)
```

后端应该根据 `result.success` 更新 MySQL：

- 成功：`status = ready`，写入 `chunk_count`
- 失败：`status = failed`，写入 `error_message`

## 三、服务器 BGE-M3 Hybrid 入库流程

如果需要使用甲方服务器上的 BGE-M3 服务入库，调用：

```python
from src.citrus_agent.rag.rag_api import RAGApi

api = RAGApi()
result = api.ingest_document_with_bge_m3(document)
```

这条链路会调用：

```text
BGE_M3_URL=http://172.21.72.18:8001
```

并写入 hybrid collection：

```text
orange_knowledge_hybrid
```

写入 Qdrant 的 point 包含：

```text
dense vector
sparse vector
payload
```

原来的本地入库入口仍然保留：

```python
api.ingest_document(document)
```

因此本地测试可以继续使用 `text-embedding-v4` dense-only 流程，服务器正式入库再使用 `ingest_document_with_bge_m3()`。

## 四、Qdrant payload 字段

第一版 payload 只保留最小字段：

```text
chunk_id
kb_id
document_id
file_name
page
chunk_index
text
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `chunk_id` | chunk 的业务 ID |
| `kb_id` | MySQL `knowledge_bases.id` |
| `document_id` | MySQL `documents.id` |
| `file_name` | 原始文件名 |
| `page` | 页码，Markdown/TXT/CSV 这类文件可能为 `None` |
| `chunk_index` | 当前文档内的 chunk 序号 |
| `text` | chunk 正文，后续会拼进 LLM prompt |

## 五、删除单个文档向量

后端删除某个文档时，可以调用：

```python
from src.citrus_agent.rag.rag_api import RAGApi

api = RAGApi()
api.delete_document_vectors(document_id=12)
```

这个方法只会删除 Qdrant 中 `payload.document_id == 12` 的 points。

它不会删除 MySQL 中的 `documents` 记录。

如果删除 BGE-M3 hybrid collection 中的文档向量，调用：

```python
from src.citrus_agent.rag.rag_api import RAGApi

api = RAGApi()
api.delete_document_vectors_from_bge_m3(document_id=12)
```

这个方法只会删除 `orange_knowledge_hybrid` 中 `payload.document_id == 12` 的 points。

## 六、删除整个知识库向量

后端删除知识库时，可以调用：

```python
from src.citrus_agent.rag.rag_api import RAGApi

api = RAGApi()
api.delete_knowledge_base_vectors(kb_id=3)
```

这个方法只会删除 Qdrant 中 `payload.kb_id == 3` 的所有 points。

它不会删除 MySQL 中的 `knowledge_bases` 或 `documents` 记录。

如果删除 BGE-M3 hybrid collection 中某个知识库的全部向量，调用：

```python
from src.citrus_agent.rag.rag_api import RAGApi

api = RAGApi()
api.delete_knowledge_base_vectors_from_bge_m3(kb_id=3)
```

这个方法只会删除 `orange_knowledge_hybrid` 中 `payload.kb_id == 3` 的 points。

## 七、检索指定知识库

如果用户当前选择了某个知识库，推荐传入 `kb_id`，避免不同知识库之间串数据。

```python
from src.citrus_agent.rag.rag_api import RAGApi

api = RAGApi()

result = api.search(
    query_text="砂糖橘溃疡病怎么防？",
    top_k=5,
    kb_id=3,
)
```

内部会构造 Qdrant filter：

```python
{"kb_id": 3}
```

## 八、不加过滤检索

如果只是本地测试，或者后台管理搜索，可以不传 `kb_id`。

```python
from src.citrus_agent.rag.rag_api import RAGApi

api = RAGApi()

result = api.search(
    query_text="砂糖橘溃疡病怎么防？",
    top_k=5,
)
```

不传 `kb_id` 时，会检索整个 Qdrant collection。

正式用户问答如果有明确知识库，建议优先传 `kb_id`。

## 九、检索返回结构

`api.search()` 返回 `DocumentSearchResult`。

```python
DocumentSearchResult(
    kb_id=3,
    query_text="砂糖橘溃疡病怎么防？",
    top_k=5,
    score=0.91,
    chunks=[
        SearchResult(
            chunk_id="3_12_0",
            content="砂糖橘溃疡病在雨季高发，应加强清园和排水...",
            score=0.91,
            payload={
                "chunk_id": "3_12_0",
                "kb_id": 3,
                "document_id": 12,
                "file_name": "砂糖橘溃疡病.md",
                "page": None,
                "chunk_index": 0,
                "text": "砂糖橘溃疡病在雨季高发，应加强清园和排水...",
            },
        )
    ],
)
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `kb_id` | 本次检索限定的知识库 ID；不传 `kb_id` 时为 `None` |
| `query_text` | 本次检索的问题 |
| `top_k` | 本次请求召回的数量 |
| `score` | 本次检索结果中的最高相似度分数 |
| `chunks` | 命中的 chunk 列表 |

每个 `SearchResult` 字段说明：

| 字段 | 说明 |
| --- | --- |
| `chunk_id` | 命中的 chunk ID |
| `content` | 命中的文本内容，可直接拼进 prompt |
| `score` | 当前 chunk 的向量相似度分数 |
| `payload` | Qdrant 中保存的完整 payload |

## 十、拼接 Prompt 的建议

后端或 LLM 层拿到检索结果后，可以直接使用：

```python
texts = [chunk.content for chunk in result.chunks]
context = "\n\n".join(texts)
```

如果需要引用来源，可以使用：

```python
for chunk in result.chunks:
    file_name = chunk.payload.get("file_name")
    page = chunk.payload.get("page")
    chunk_index = chunk.payload.get("chunk_index")
```

## 十一、当前文件职责

| 文件 | 职责 |
| --- | --- |
| `rag_api.py` | 后端调用门面，只暴露入库、删除、检索 |
| `retriever.py` | 向量检索器，负责问题 embedding 和 Qdrant search |
| `RAGREADME.md` | 本说明文档 |

相关数据类在：

```text
src/citrus_agent/pojo/knowledge.py
```

相关 Qdrant 封装在：

```text
src/citrus_agent/vectorstores/qdrant.py
```

相关文档解析和入库编排在：

```text
src/citrus_agent/services/document_parser.py
src/citrus_agent/services/knowledge_service.py
```

## 十二、注意事项

- RAG 模块不保存 MySQL 账号密码。
- RAG 模块不直接连接 MySQL。
- 后端负责把 MySQL 查询结果转换成 `DocumentEntity`。
- 后端负责根据 `DocumentIngestResult` 更新 MySQL 文档状态。
- 重新入库同一文档时，默认会先按 `document_id` 删除旧向量，再写入新向量。
- 删除知识库向量时，只会按 `kb_id` 删除 Qdrant points，不会删除 MySQL 数据。
- 当前没有 rerank，返回顺序就是 Qdrant 向量检索返回顺序。
- `orange_knowledge` 是旧版 dense-only collection。
- `orange_knowledge_hybrid` 是 BGE-M3 dense + sparse collection。
- 如果 hybrid 入库报 `Not existing vector name error: sparse`，通常是 hybrid collection 被误创建成 dense-only，需要删除 `orange_knowledge_hybrid` 后重新入库。
