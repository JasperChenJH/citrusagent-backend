# 橘子知识库 RAG 模块设计开发技术文档

## 1. 文档目的

本文档用于说明 CitrusAgent 项目中 RAG 模块的当前设计、代码边界、入库流程、检索流程、服务器部署情况、后端调用方式和运维注意事项。

当前 RAG 模块主要负责：

- 接收后端传入的 MySQL `documents` 表文档实体。
- 根据 `stored_path` 读取并解析文件。
- 对文本进行清洗和分块。
- 生成 embedding 向量。
- 将 chunk 向量和 payload 写入 Qdrant。
- 从 Qdrant 检索相关知识片段。
- 将检索结果封装成结构化对象返回给后端或 LLM 层。

当前 RAG 模块不负责：

- 不直接连接 MySQL。
- 不创建 MySQL 表。
- 不更新 `documents.status`、`chunk_count`、`error_message`。
- 不提供 HTTP API 路由。
- 不处理文件上传接口。
- 不调用聊天大模型生成最终回答。

## 2. 当前整体架构

当前系统保留两条链路：

| 链路 | 用途 | 入口方法 | Qdrant collection | 向量类型 |
| --- | --- | --- | --- | --- |
| 本地测试链路 | 本地开发、单机测试 | `RAGApi.ingest_document()` / `RAGApi.search()` | `orange_knowledge` | dense-only |
| 服务器正式链路 | 甲方服务器正式 RAG | `RAGApi.ingest_document_with_bge_m3()` / `RAGApi.search_with_bge_m3_rerank()` | `orange_knowledge_hybrid` | dense + sparse + rerank |

当前正式服务器规划和现状：

```text
Qdrant:               http://172.21.72.18:6333
BGE-M3 embedding:     http://172.21.72.18:8001
Qwen3 reranker:       http://172.21.72.18:8002
```

整体模块调用关系：

```text
后端业务层
    |
    | 传入 DocumentEntity / query_text / kb_id
    v
RAGApi
    |
    |-- KnowledgeIngestor
    |       |-- DocumentParser
    |       |-- ApiEmbeddingProvider
    |       |-- QdrantStore
    |
    |-- HybridKnowledgeIngestor
    |       |-- DocumentParser
    |       |-- BgeM3ApiEmbeddingProvider
    |       |-- QdrantStore
    |
    |-- CitrusRetriever
    |       |-- ApiEmbeddingProvider
    |       |-- QdrantStore
    |
    |-- HybridCitrusRetriever
            |-- BgeM3ApiEmbeddingProvider
            |-- QdrantStore
            |-- QwenRerankerClient
```

## 3. 代码模块职责

| 文件 | 职责 |
| --- | --- |
| `src/citrus_agent/rag/rag_api.py` | 后端调用门面，暴露入库、删除、检索方法 |
| `src/citrus_agent/rag/retriever.py` | dense 检索器和 hybrid 检索器 |
| `src/citrus_agent/rag/reranker.py` | Qwen3-Reranker-4B HTTP 客户端 |
| `src/citrus_agent/services/knowledge_service.py` | 文档入库编排，串联解析、分块、embedding、写入 |
| `src/citrus_agent/services/document_parser.py` | 多格式文档解析 |
| `src/citrus_agent/vectorstores/embeddings.py` | embedding provider 封装，包括 API embedding 和 BGE-M3 服务调用 |
| `src/citrus_agent/vectorstores/qdrant.py` | Qdrant collection、upsert、delete、search 封装 |
| `src/citrus_agent/pojo/knowledge.py` | RAG 相关数据实体 |
| `scripts/server/bge_m3_server.py` | 部署到服务器的 BGE-M3 FastAPI 服务脚本 |

## 4. MySQL 元数据边界

MySQL 用于保存知识库和文档元数据，不保存向量。

当前涉及两张表：

- `knowledge_bases`：知识库表。
- `documents`：文档表。

字段对应关系：

| MySQL 字段 | RAG / Qdrant 对应字段 | 说明 |
| --- | --- | --- |
| `knowledge_bases.id` | `payload.kb_id` | 知识库 ID |
| `documents.id` | `payload.document_id` | 文档 ID |
| `documents.original_filename` | `payload.file_name` | 原始文件名 |
| `documents.stored_path` | 文档解析路径 | RAG 根据该路径读取文件 |
| `documents.status` | 不由 RAG 更新 | 后端根据入库结果更新 |
| `documents.chunk_count` | 不由 RAG 更新 | 后端根据入库结果更新 |
| `documents.error_message` | 不由 RAG 更新 | 后端根据入库结果更新 |

RAG 模块只接收后端组装好的 `DocumentEntity`，不直接访问数据库。

## 5. 核心数据对象

### 5.1 DocumentEntity

`DocumentEntity` 是 RAG 入库的正式入口参数，对应 MySQL `documents` 表记录。

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

### 5.2 DocumentIngestResult

`DocumentIngestResult` 是单个文档入库结果，后端根据它更新 MySQL 文档状态。

```python
DocumentIngestResult(
    document_id=12,
    kb_id=3,
    success=True,
    chunk_count=8,
    error_message="",
)
```

失败示例：

```python
DocumentIngestResult(
    document_id=12,
    kb_id=3,
    success=False,
    chunk_count=0,
    error_message="No space left on device: WAL buffer size exceeds available disk space",
)
```

### 5.3 SearchResult

`SearchResult` 表示单个命中的 chunk。

```python
SearchResult(
    chunk_id="3:12:0:xxxx",
    content="砂糖橘溃疡病在雨季高发...",
    score=0.91,
    payload={...},
)
```

在旧版 dense-only 检索中，`score` 是 Qdrant dense 相似度分数。

在新版 hybrid + rerank 检索中，`score` 是 Qwen reranker 最终分数，`payload` 中会额外带有：

```text
dense_score
sparse_score
rrf_score
rerank_score
```

这些字段主要用于调试检索效果。

### 5.4 DocumentSearchResult

`DocumentSearchResult` 表示一次检索整体返回。

```python
DocumentSearchResult(
    kb_id=3,
    query_text="砂糖橘溃疡病怎么防？",
    top_k=8,
    score=0.98,
    chunks=[...],
)
```

其中：

- `kb_id`：本次限定的知识库 ID；不限定时为 `None`。
- `query_text`：用户问题。
- `top_k`：请求返回数量。
- `score`：当前返回结果中的最高分。
- `chunks`：命中的知识片段列表。

## 6. Qdrant 存储设计

### 6.1 Collection 设计

当前有两个 collection：

```text
orange_knowledge
orange_knowledge_hybrid
```

说明：

| collection | 用途 | 结构 |
| --- | --- | --- |
| `orange_knowledge` | 本地 dense-only 测试链路 | 单 dense vector |
| `orange_knowledge_hybrid` | 服务器 BGE-M3 正式链路 | named dense vector + sparse vector |

`orange_knowledge_hybrid` 的向量名：

```text
dense
sparse
```

注意：

- dense-only collection 和 hybrid collection 结构不同，不能混用。
- 如果 `orange_knowledge_hybrid` 被误创建成 dense-only collection，写入 sparse 时会报错。
- 典型错误是：`Not existing vector name error: sparse`。
- 解决方式是删除错误 collection 后重新入库。

### 6.2 Payload 设计

第一版 payload 保持最小可用。

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

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `chunk_id` | `str` | chunk 业务 ID |
| `kb_id` | `int` | 所属知识库 ID |
| `document_id` | `int` | 所属文档 ID |
| `file_name` | `str` | 原始文件名 |
| `page` | `int | None` | 页码，Markdown/TXT 可能为空 |
| `chunk_index` | `int` | 文档内 chunk 序号 |
| `text` | `str` | chunk 正文，用于拼接 prompt |

暂不把品种、病虫害、地区、生长期等业务标签写入 payload。当前优先保证：

- 能按 `document_id` 删除文档向量。
- 能按 `kb_id` 过滤知识库。
- 能将检索到的 `text` 直接拼接到 prompt。

## 7. 文档解析与分块

### 7.1 支持格式

当前 `DocumentParser` 支持：

```text
pdf
docx
xlsx
csv
md
txt
```

### 7.2 分块策略

当前使用简单字符分块：

```text
CHUNK_SIZE=500
CHUNK_OVERLAP=80
```

含义：

- 每个 chunk 目标长度约 500 个字符。
- 相邻 chunk 重叠 80 个字符。

设计原因：

- 逻辑简单，便于稳定调试。
- 适合第一版知识库入库。
- overlap 可以减少关键信息被切断的概率。

后续可以升级为：

- 按 Markdown 标题切分。
- 按段落切分。
- 按 token 数切分。
- 按语义边界切分。

## 8. 入库流程

### 8.1 本地 dense-only 入库

入口：

```python
result = api.ingest_document(document)
```

流程：

```text
后端查询 MySQL documents
    -> 组装 DocumentEntity
    -> RAGApi.ingest_document(document)
    -> DocumentParser 解析文件
    -> KnowledgeIngestor 清洗并分块
    -> 构造最小 payload
    -> 调用 text-embedding-v4 API
    -> 按 document_id 删除旧 point
    -> 写入 orange_knowledge
    -> 返回 DocumentIngestResult
```

该链路主要用于本地测试，方便在没有服务器 BGE-M3 的情况下验证基本流程。

### 8.2 服务器 BGE-M3 hybrid 入库

入口：

```python
result = api.ingest_document_with_bge_m3(document)
```

流程：

```text
后端查询 MySQL documents
    -> 组装 DocumentEntity
    -> RAGApi.ingest_document_with_bge_m3(document)
    -> DocumentParser 解析文件
    -> HybridKnowledgeIngestor 清洗并分块
    -> 构造最小 payload
    -> 调用 BGE-M3 服务 /embed
    -> 获得 dense vector + sparse vector
    -> 按 document_id 删除 orange_knowledge_hybrid 中旧 point
    -> 写入 dense + sparse + payload
    -> 返回 DocumentIngestResult
```

BGE-M3 `/embed` 返回结构：

```json
{
  "model": "BAAI/bge-m3",
  "dense_dim": 1024,
  "results": [
    {
      "index": 0,
      "dense": [0.01, 0.02],
      "sparse": {
        "indices": [123, 456],
        "values": [0.8, 0.3]
      }
    }
  ]
}
```

写入 Qdrant point 的结构：

```python
{
    "id": "point_id",
    "vector": {
        "dense": [...],
        "sparse": SparseVector(indices=[...], values=[...]),
    },
    "payload": {
        "chunk_id": "...",
        "kb_id": 1,
        "document_id": 12,
        "file_name": "README.md",
        "page": None,
        "chunk_index": 0,
        "text": "...",
    },
}
```

### 8.3 重新入库策略

默认 `replace_existing=True`。

同一文档重新入库时，会先按 `document_id` 删除旧 point，再写入新 point。

删除条件：

```text
payload.document_id == documents.id
```

这样可以避免旧版本 chunk 残留。

## 9. 检索流程

### 9.1 本地 dense-only 检索

入口：

```python
result = api.search(
    query_text="砂糖橘溃疡病怎么防？",
    top_k=5,
    kb_id=1,
)
```

流程：

```text
用户问题
    -> text-embedding-v4 生成 query dense vector
    -> Qdrant orange_knowledge dense 检索
    -> 可选 kb_id 强过滤
    -> 返回 DocumentSearchResult
```

不传 `kb_id` 时，会做全库检索。

### 9.2 服务器 hybrid + rerank 检索

入口：

```python
result = api.search_with_bge_m3_rerank(
    query_text="砂糖橘溃疡病怎么防？",
    top_k=8,
    kb_id=1,
)
```

正式服务器检索流程：

```text
用户问题
    -> 调用 BGE-M3 生成查询 dense + sparse
    -> Qdrant dense Top40
    -> Qdrant sparse Top40
    -> Python 层 RRF 融合 Top25
    -> 调用 Qwen3-Reranker-4B 打分重排
    -> 保留 Top5～8
    -> 返回 DocumentSearchResult
```

当前默认配置：

```text
HYBRID_DENSE_TOP_K=40
HYBRID_SPARSE_TOP_K=40
HYBRID_CANDIDATE_TOP_K=25
HYBRID_FINAL_TOP_K=8
HYBRID_RRF_K=60
```

RRF 公式：

```text
score += 1 / (HYBRID_RRF_K + rank)
```

其中 `rank` 从 1 开始。

### 9.3 检索结果拼接 prompt

LLM 层可以直接使用：

```python
context = "\n\n".join(chunk.content for chunk in result.chunks)
```

引用来源：

```python
for chunk in result.chunks:
    file_name = chunk.payload.get("file_name")
    page = chunk.payload.get("page")
    chunk_index = chunk.payload.get("chunk_index")
```

## 10. 删除流程

### 10.1 删除 dense-only 文档向量

```python
api.delete_document_vectors(document_id=12)
```

删除条件：

```text
payload.document_id == 12
```

### 10.2 删除 hybrid 文档向量

```python
api.delete_document_vectors_from_bge_m3(document_id=12)
```

删除 `orange_knowledge_hybrid` 中指定文档的全部 point。

### 10.3 删除 dense-only 知识库向量

```python
api.delete_knowledge_base_vectors(kb_id=1)
```

删除条件：

```text
payload.kb_id == 1
```

### 10.4 删除 hybrid 知识库向量

```python
api.delete_knowledge_base_vectors_from_bge_m3(kb_id=1)
```

删除 `orange_knowledge_hybrid` 中指定知识库的全部 point。

所有删除方法只操作 Qdrant，不操作 MySQL。

## 11. 后端对接方式

### 11.1 初始化

建议后端服务启动时初始化一次：

```python
from src.citrus_agent.rag.rag_api import RAGApi

rag_api = RAGApi()
```

### 11.2 本地测试入库

```python
result = rag_api.ingest_document(document)
```

### 11.3 服务器正式入库

```python
result = rag_api.ingest_document_with_bge_m3(document)
```

### 11.4 本地测试检索

```python
result = rag_api.search(
    query_text="Qdrant 向量数据库怎么部署？",
    top_k=5,
    kb_id=1,
)
```

### 11.5 服务器正式检索

```python
result = rag_api.search_with_bge_m3_rerank(
    query_text="Qdrant 向量数据库怎么部署？",
    top_k=8,
    kb_id=1,
)
```

## 12. 配置项

配置来自 `.env` 和 `src/citrus_agent/core/config.py`。

### 12.1 Qdrant 配置

| 配置项 | 当前含义 |
| --- | --- |
| `QDRANT_URL` | Qdrant 服务地址，例如 `http://172.21.72.18:6333` |
| `QDRANT_COLLECTION` | dense-only collection，默认 `orange_knowledge` |
| `QDRANT_HYBRID_COLLECTION` | hybrid collection，默认 `orange_knowledge_hybrid` |
| `QDRANT_DISTANCE` | 向量距离算法，默认 `Cosine` |
| `QDRANT_DENSE_VECTOR_NAME` | hybrid dense 向量名，默认 `dense` |
| `QDRANT_SPARSE_VECTOR_NAME` | hybrid sparse 向量名，默认 `sparse` |

### 12.2 本地 API embedding 配置

| 配置项 | 当前含义 |
| --- | --- |
| `EMBEDDING_PROVIDER` | 默认 `api` |
| `EMBEDDING_BASE_URL` | OpenAI 兼容 embedding API 地址 |
| `EMBEDDING_MODEL` | 默认 `text-embedding-v4` |
| `EMBEDDING_DIMENSIONS` | 默认 `1024` |
| `EMBEDDING_VECTOR_SIZE` | 默认 `1024` |

### 12.3 BGE-M3 配置

| 配置项 | 当前含义 |
| --- | --- |
| `BGE_M3_URL` | BGE-M3 服务地址，默认 `http://172.21.72.18:8001` |
| `BGE_M3_TIMEOUT` | 请求超时时间 |
| `BGE_M3_BATCH_SIZE` | 入库时批量 embedding 大小 |

### 12.4 Qwen reranker 配置

| 配置项 | 当前含义 |
| --- | --- |
| `QWEN_RERANKER_URL` | Qwen reranker 服务地址，默认 `http://172.21.72.18:8002` |
| `QWEN_RERANKER_MODEL` | vLLM 中的模型名，默认 `qwen3-reranker-4b` |
| `QWEN_RERANKER_TIMEOUT` | 请求超时时间 |
| `QWEN_RERANKER_BATCH_SIZE` | rerank 打分批大小 |

### 12.5 检索配置

| 配置项 | 当前含义 |
| --- | --- |
| `RETRIEVAL_TOP_K` | dense-only 检索默认召回数 |
| `HYBRID_DENSE_TOP_K` | hybrid dense 召回数 |
| `HYBRID_SPARSE_TOP_K` | hybrid sparse 召回数 |
| `HYBRID_CANDIDATE_TOP_K` | RRF 后进入 rerank 的候选数 |
| `HYBRID_FINAL_TOP_K` | rerank 后最终返回数量 |
| `HYBRID_RRF_K` | RRF 平滑常数 |
| `CHUNK_SIZE` | 分块大小 |
| `CHUNK_OVERLAP` | 分块重叠 |

## 13. 服务器部署资源

当前甲方服务器相关资源如下。

### 13.1 Docker 容器

```text
qdrant
bge-m3-embedder
qwen3-reranker-4b
```

端口：

```text
qdrant:              6333/6334
bge-m3-embedder:    8001
qwen3-reranker-4b:  8002
```

### 13.2 Docker 镜像

```text
qdrant/qdrant
pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
vllm/vllm-openai:v0.8.5
```

当前 `vllm/vllm-openai:latest` 不建议使用，因为该镜像版本过新，曾出现需要 CUDA 13 的问题。服务器当前驱动适配 `vllm/vllm-openai:v0.8.5`。

### 13.3 宿主机目录

| 路径 | 作用 | 是否可删除 |
| --- | --- | --- |
| `/home/yunxuan/qdrant_storage` | Qdrant 向量数据持久化目录 | 不能直接删除 |
| `/data/models` | BGE-M3 和 Qwen3-Reranker-4B 模型目录 | 不能删除 |
| `/data/rag_services/bge_m3_server.py` | BGE-M3 HTTP 服务脚本 | 不能删除 |
| `/var/lib/docker` | Docker 核心数据目录 | 不能直接删除 |

说明：

- `/home/yunxuan/qdrant_storage` 保存 Qdrant collection、向量、payload、WAL。
- `/data/models` 保存模型文件，删除后 embedding/reranker 服务会无法启动或需要重新下载。
- `/data/rag_services/bge_m3_server.py` 是 BGE-M3 服务启动脚本。
- `/var/lib/docker` 保存 Docker 容器、镜像层、容器日志、volume 元数据，不能手动 `rm -rf`。

## 14. 磁盘空间问题与处理建议

当前服务器曾出现 Qdrant 入库失败：

```text
No space left on device: WAL buffer size exceeds available disk space
```

含义：

- Qdrant 写入 point 前需要写 WAL。
- 服务器磁盘剩余空间不足时，WAL 无法写入。
- 入库失败后 hybrid collection 中没有新 chunk，所以检索结果可能为 0。

排查命令：

```bash
df -h
sudo du -h --max-depth=1 / | sort -hr
sudo docker system df -v
sudo du -h --max-depth=1 /var/lib/docker | sort -hr
du -sh /home/yunxuan/qdrant_storage
du -h --max-depth=2 /data/models | sort -hr | head -30
```

清理注意：

- 不要直接删除 `/home/yunxuan/qdrant_storage`。
- 不要直接删除 `/data/models`。
- 不要直接删除 `/var/lib/docker`。
- 如果必须清理 Qdrant 数据，优先通过 Qdrant API 删除确认不需要的 collection。

查看 collection：

```bash
curl http://172.21.72.18:6333/collections
```

删除指定 collection：

```bash
curl -X DELETE http://172.21.72.18:6333/collections/collection_name
```

清理 Docker 时建议由甲方运维执行：

```bash
sudo docker system df
sudo docker builder prune
sudo docker system prune
```

不要执行：

```bash
sudo rm -rf /var/lib/docker
```

## 15. 测试设计

当前测试覆盖：

| 测试文件 | 覆盖内容 |
| --- | --- |
| `tests/test_document_parser.py` | Markdown、TXT、CSV 等解析 |
| `tests/test_knowledge_service.py` | 入库、分块、payload、失败返回 |
| `tests/test_qdrant_store.py` | Qdrant 删除、hybrid upsert、hybrid search 封装 |
| `tests/test_bge_m3_embedding_provider.py` | BGE-M3 `/embed` 响应解析 |
| `tests/test_qwen_reranker.py` | Qwen reranker score 响应解析 |
| `tests/test_hybrid_retriever.py` | BGE-M3 + dense/sparse + RRF + rerank 流程 |
| `tests/test_rag_api.py` | 后端门面方法 |
| `tests/test_bge_m3_hybrid_ingest_real.py` | 真实 BGE-M3 + Qdrant hybrid 入库，默认跳过 |

本地测试命令：

```powershell
G:\worktool\Anaconda\envs\agent-dev\python.exe -m pytest -q tests\test_rag_api.py tests\test_hybrid_retriever.py tests\test_qwen_reranker.py tests\test_qdrant_store.py --basetemp=.pytest_tmp_agent_dev
```

真实服务器入库测试需要保证：

- `QDRANT_URL=http://172.21.72.18:6333`
- `BGE_M3_URL=http://172.21.72.18:8001`
- Qdrant 磁盘空间充足。
- `orange_knowledge_hybrid` collection 结构正确。

真实 hybrid 查询测试还需要：

- `QWEN_RERANKER_URL=http://172.21.72.18:8002`
- `qwen3-reranker-4b` 容器正常运行。

## 16. 当前已完成能力

当前已完成：

- MySQL `DocumentEntity` 驱动的文档入库。
- 多格式文档解析。
- 字符分块和 overlap。
- 最小 payload 构造。
- 本地 API dense-only 入库。
- 服务器 BGE-M3 dense + sparse hybrid 入库。
- 按 `document_id` 删除 dense-only 和 hybrid 向量。
- 按 `kb_id` 删除 dense-only 和 hybrid 向量。
- 本地 dense-only 检索。
- 服务器 BGE-M3 dense + sparse 双路召回。
- Python 层 RRF 融合。
- Qwen3-Reranker-4B 重排。
- 统一 `DocumentSearchResult` 返回结构。

## 17. 当前限制

当前仍存在的限制：

- 分块策略仍然是简单字符分块，尚未按语义边界切分。
- payload 暂未保存品种、病虫害、地区、生长期等业务字段。
- 没有接 LLM 问题改写。
- 没有接最终回答生成链路。
- 没有做多用户权限控制，权限仍由后端保证。
- reranker 服务依赖服务器 GPU 和磁盘空间，服务不可用时 hybrid 查询会失败。
- Qdrant 磁盘空间不足时，入库会因为 WAL 无法写入而失败。

## 18. 后续优化方向

建议后续优化：

1. 优化分块策略，按标题、段落或 token 数切分。
2. 增加业务 metadata，例如品种、病虫害、地区、生长期。
3. 增加检索日志，记录 query、召回分数、rerank 分数和最终命中。
4. 增加 reranker 降级策略，例如 reranker 不可用时返回 RRF TopN。
5. 增加结果去重，避免同一文档连续 chunk 内容重复过多。
6. 增加引用来源格式化，方便前端展示文件名、页码、chunk 序号。
7. 接入最终 LLM 回答生成链路。
8. 根据甲方服务器资源，调整 TopK、batch size 和模型并发参数。

## 19. 总结

当前 RAG 模块已经从第一版 dense-only 检索升级为两套并存的结构：

- 本地保留 `text-embedding-v4` dense-only 链路，方便开发和测试。
- 服务器新增 BGE-M3 dense + sparse hybrid 入库和检索链路，并接入 Qwen3-Reranker-4B 做最终重排。

正式链路的核心流程是：

```text
文档入库：
DocumentEntity
    -> 文档解析
    -> 文本分块
    -> BGE-M3 dense + sparse
    -> Qdrant orange_knowledge_hybrid

用户检索：
query_text
    -> BGE-M3 dense + sparse
    -> Qdrant dense Top40 + sparse Top40
    -> RRF Top25
    -> Qwen3-Reranker-4B
    -> Top5～8
    -> DocumentSearchResult
```

后端只需要调用 `RAGApi`，不需要关心文档解析、向量生成、Qdrant 查询和 rerank 的内部细节。
