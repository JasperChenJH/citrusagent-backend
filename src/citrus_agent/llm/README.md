# YAGO intellect LLM 模块说明文档

## 模块概览

```
llm/
├── base.py           # ChatLLM 客户端：封装 DeepSeek API，支持流式/非流式对话
├── intent_router.py  # IntentRouter 意图路由器：LLM 意图分类 → 决定是否检索知识库
└── README.md         # 本文件
```

上游调用方：`src/citrus_agent/api/v1/chat.py` — 三个前端聊天入口的流式 SSE 接口。

---

## 一、ChatLLM 客户端 (`base.py`)

### 构造函数

```python
llm = ChatLLM(
    api_key=None,       # 默认从 .env DEEPSEEK_API_KEY 读取
    base_url=None,      # 默认 https://api.deepseek.com
    model=None,         # 默认 deepseek-chat
    temperature=0.7,    # 默认温度
    max_tokens=2048,    # 默认最大 token 数
)
```

### 核心方法

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `chat(messages)` | `str` | 非流式对话，返回完整回复 |
| `chat_stream(messages)` | `Generator[str]` | 流式对话，逐块 yield delta 文本 |
| `chat_with_context(question, chunks, prompt)` | `str` | 带知识库上下文的非流式问答 |
| `chat_with_context_stream(question, chunks, prompt)` | `Generator[str]` | 带知识库上下文的流式问答 |

### 上下文注入机制 (`_build_rag_messages`)

RAG 检索片段通过 `{context}` 占位符注入 system prompt，三种注入模式：

| 条件 | 行为 |
|------|------|
| `system_prompt=None` | 使用默认 prompt，自动拼接知识片段 |
| `system_prompt` 含 `{context}` | 将 `{context}` 替换为实际检索片段 |
| `system_prompt` 不含占位符 | 在 prompt 末尾自动追加知识片段 |

> 这是防止流式输出格式异常的关键修复：之前传入自定义 prompt 时 context 会被静默忽略，导致模型无知识片段可引用，从而编造内容。

---

## 二、IntentRouter 意图路由器 (`intent_router.py`)

### 调用流程

```
用户问题 → IntentRouter.classify(question, route) → IntentResult
                                                          ├── need_rag: bool
                                                          ├── kb_id: int | None
                                                          └── reasoning: str
```

### 三个路由的分类规则

| 路由 | 触发 RAG 条件 | kb_id |
|------|-------------|-------|
| `general` | 需要专业知识才能回答 | 1 |
| `agri` | 柑橘种植、施肥、土壤、品种等专业问题 | 2 |
| `diagnose` | 病虫害症状识别、防治方案、农药使用 | 3 |

### 输出格式约束

分类 LLM 必须输出严格 JSON，通过正则提取和解析兜底确保鲁棒性：

```json
{"need_rag": true, "kb_id": 2, "reasoning": "用户询问柑橘溃疡病防治方法"}
```

解析失败时默认 `need_rag=False`，走直接回答模式，保证系统不退化为错误。

---

## 三、Prompt 模板结构

所有 Prompt 模板定义在 `src/citrus_agent/api/v1/chat.py`，参考西智蔗 prompt 架构设计。

### 三层结构

```
┌─ 身份信息层 ──────────────────────────────────┐
│ 名称、定位、核心能力                            │
├─ 领域规则层 ──────────────────────────────────┤
│ 基于知识库的回答规则 / 诊断专用规则              │
├─ 输出格式约束 ────────────────────────────────┤
│ HTML 标签、禁止 Markdown、禁止 <think>          │
├─ 防幻觉规则 ──────────────────────────────────┤
│ 不确定时如实说明、禁止编造数据                   │
└─ 检索片段注入 ────────────────────────────────┘
│ {context} 占位符 → ChatLLM 自动填入检索结果     │
```

### 六套 Prompt 模板

| 模板变量 | 路由 | 模式 | 用途 |
|----------|------|------|------|
| `_GENERAL_DIRECT_PROMPT` | general | 直接回答 | 通用问答，不走 RAG |
| `_AGRI_DIRECT_PROMPT` | agri | 直接回答 | 农业知识直接回答 |
| `_DIAGNOSE_DIRECT_PROMPT` | diagnose | 直接回答 | 病虫害诊断直接回答 |
| `_GENERAL_RAG_PROMPT` | general | RAG | 通用 + 检索片段 |
| `_AGRI_RAG_PROMPT` | agri | RAG | 农业知识 + 检索片段 |
| `_DIAGNOSE_RAG_PROMPT` | diagnose | RAG | 病虫害诊断 + 检索片段 |

### 身份体系

三个助手统一归属 **YAGO intellect** 品牌：

| 路由 | 助手名称 | 定位 |
|------|----------|------|
| general | YAGO intellect 通用问答助手 | 农业科研与日常咨询 |
| agri | YAGO intellect 农业知识 AI 助手 | 柑橘及农业知识库精准问答 |
| diagnose | YAGO intellect 病虫害诊断助手 | 病虫害识别、诊断与防治建议 |

---

## 四、流式输出格式

### SSE 协议

```
Content-Type: text/event-stream

data: {"delta": "文本片段1"}

data: {"delta": "文本片段2"}

data: {"delta": "", "done": true, "from_knowledge_base": true, "kb_id": 2, "reasoning": "..."}
```

### done 事件字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `delta` | str | 空字符串，标记流结束 |
| `done` | bool | 固定 `true` |
| `from_knowledge_base` | bool | 是否来自知识库检索 |
| `kb_id` | int | 实际使用的知识库 ID（仅 RAG 模式） |
| `reasoning` | str | 意图判断理由，用于调试日志 |

---

## 五、防流式输出格式异常的措施

参考西智蔗 prompt 经验，重点关注 LLM 在流式输出时可能出现的问题：

### 1. 强制 HTML 输出 (`_OUTPUT_FORMAT_RULES`)

```
- 使用 HTML 标签：<p> 包裹段落，<b> 加粗
- 列表用 <ul>/<li> 或 <ol>/<li>
- 严禁 Markdown（**、#、- 列表等）
- 严禁 <think>、</think> 标签
- 严禁输出 JSON、代码块
```

**原理**：不指定格式时，模型可能在流式输出中混用 Markdown 和 HTML，
导致前端无法正确渲染。统一为 HTML 标签后，前端用单一 HTML 渲染器即可处理所有 delta 片段。

### 2. 防幻觉约束 (`_ANTI_HALLUCINATION_RULES`)

```
- 不确定时说"暂未收录"或"无法确定"
- 不编造数据、文献、结论
- 引用时保持原文信息
```

**原理**：当检索片段不足时，模型倾向于"自信地编造"。显式约束让模型在流式输出中自行中断不确定的内容，避免输出前后矛盾的异常文本。

### 3. 占位符注入策略

`{context}` 占位符在 prompt 中的位置固定为末尾，模型在 consume 所有约束规则后看到实际检索片段，再回答用户问题。这种"先规则后数据"的结构确保：
- 模型先理解身份和输出要求
- 再读取检索结果
- 最后组织回答

避免"先看到数据后忘记规则"导致格式漂移。

### 4. IntentRouter 输出隔离

意图分类使用独立的 `ChatLLM(temperature=0.0, max_tokens=256)` 实例，与主对话 LLM 完全隔离：
- 低 temperature 保证分类结果稳定
- 小 max_tokens 限制输出长度，只返回 JSON
- 正则兜底解析，JSON 解析失败也不影响主流程

### 5. 历史消息不保留

当前 `_build_rag_messages` 只构建 `[system, user]` 两条消息，不携带历史对话。
这是有意为之：多轮对话上下文中，旧回答的格式残留可能误导模型在新一轮中输出不一致的格式。
后续如需多轮对话，建议清洗历史消息中的格式标记后再传入。

---

## 六、配置项

所有配置从 `.env` 读取（`src/citrus_agent/core/config.py`）：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DEEPSEEK_API_KEY` | (必填) | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型名称 |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant 向量库地址 |
| `EMBEDDING_PROVIDER` | `api` | embedding 提供方 |
| `EMBEDDING_MODEL` | `text-embedding-v4` | embedding 模型 |
| `RETRIEVAL_TOP_K` | `30` | 向量召回数量 |

---

## 七、调试建议

1. **验证 RAG 是否生效**：检查 SSE done 事件的 `from_knowledge_base` 和 `kb_id` 字段。
2. **验证意图分类**：检查 done 事件的 `reasoning` 字段，确认分类逻辑正确。
3. **验证 prompt 注入**：在 `_build_rag_messages` 中临时打印 `system_prompt[:200]` 确认 `{context}` 已被替换。
4. **格式异常排查**：如果前端收到不渲染的内容，检查 delta 片段是否包含 Markdown 标记或 `<think>` 标签。
