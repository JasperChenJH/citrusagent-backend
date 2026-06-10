"""RAG 检索增强生成链路。

职责：
    1. 负责问题改写、知识检索、上下文拼接、引用来源整理。
    2. 连接 `llm` 和 `vectorstores`。
    3. 管理农业问答 Prompt 模板。

建议文件：
    - chain.py：RAG 主流程。
    - retriever.py：检索器封装。
    - prompts.py：Prompt 模板。
    - rerank.py：重排逻辑。

注意：
    RAG 层不处理 HTTP 请求，也不直接返回前端响应格式。
"""
