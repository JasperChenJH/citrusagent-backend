"""
RAG 检索增强生成服务
核心流程：查询向量化 → 向量检索 → 构建Prompt → LLM生成回答
"""

from typing import Optional

from src.citrus_agent.core.config import settings
from src.citrus_agent.vectorstores.qdrant import QdrantStore
from src.citrus_agent.llm.base import ChatLLM
from src.citrus_agent.vectorstores.embeddings import create_embedding_provider
from src.citrus_agent.pojo.chat import ChatStreamChunk
from src.citrus_agent.services.logger import logger

CHAT_HISTORY_MAX_TURNS = 10
SCORE_THRESHOLD = 0.7

SYSTEM_PROMPT = """你是一个专业的广西橙子/柑橘产业智能助手，专门为农户、客商和相关从业者提供信息服务。

## 核心原则
1. **严格基于知识库回答**：你必须且只能根据检索到的参考资料来回答问题。
2. **拒绝超范围问题**：如果知识库中没有相关内容，你必须明确告知用户"当前知识库中未找到相关信息，无法回答该问题"，不得自行编造或推测。
3. **标注来源**：每个回答必须标注参考资料的出处，格式为 [来源: 文档标题]。
4. **多轮对话**：结合上下文理解用户意图，必要时进行追问或引导。

## 回答格式
- 回答问题时，先给出直接答案，再补充详细说明
- 涉及数据时，必须引用具体来源
- 如果用户问题模糊，主动引导用户补充信息

## 工具调用
- 当用户需要计算类操作时（如施肥量计算），使用对应工具完成
- 工具调用结果也需要结合知识库内容进行解释
"""


class RAGService:
    """RAG 检索增强生成服务"""

    def __init__(self):
        self.settings = settings
        self.embedding_provider = create_embedding_provider()
        self.qdrant = QdrantStore(vector_size=self.embedding_provider.vector_size)
        self.llm = ChatLLM()

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        向量检索相关文档

        Args:
            query: 查询文本
            top_k: 返回数量
            score_threshold: 相似度阈值
            filters: 元数据过滤

        Returns:
            检索结果列表
        """
        top_k = top_k or self.settings.retrieval_top_k
        score_threshold = score_threshold or SCORE_THRESHOLD

        query_vector = self.embedding_provider.embed_text(query)

        results = self.qdrant.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

        results = [r for r in results if r.get("score", 0) >= score_threshold]

        logger.info(f"RAG 检索完成，查询: '{query[:30]}...'，命中 {len(results)} 条")
        return results

    def _build_context(self, search_results: list[dict]) -> str:
        """
        将检索结果构建为上下文文本

        Args:
            search_results: 检索结果列表

        Returns:
            格式化的上下文文本
        """
        if not search_results:
            return "（知识库中未找到相关内容）"

        context_parts = []
        for i, result in enumerate(search_results, 1):
            payload = result.get("payload", {})
            source_title = payload.get("file_name", payload.get("document_title", "未知来源"))
            content = payload.get("text", payload.get("content", ""))
            score = result.get("score", 0)

            context_parts.append(
                f"[参考资料 {i}] 来源: {source_title} (相似度: {score:.2f})\n{content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _build_messages(
        self,
        question: str,
        context: str,
        history: Optional[list[dict]] = None,
    ) -> list[dict[str, str]]:
        """
        构建完整的消息列表

        Args:
            question: 用户问题
            context: 检索到的上下文
            history: 对话历史

        Returns:
            OpenAI 格式的消息列表
        """
        full_messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        if history:
            for msg in history[-CHAT_HISTORY_MAX_TURNS * 2:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "assistant"):
                    full_messages.append({"role": role, "content": content})

        full_messages.append(
            {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"}
        )

        return full_messages

    async def generate(
        self,
        question: str,
        history: Optional[list[dict]] = None,
        filters: Optional[dict] = None,
    ) -> dict:
        """
        完整的 RAG 生成流程

        Args:
            question: 用户问题
            history: 对话历史
            filters: 元数据过滤

        Returns:
            包含 answer 和 sources 的字典
        """
        search_results = await self.retrieve(question, filters=filters)

        context = self._build_context(search_results)

        messages = self._build_messages(question, context, history)

        answer = self.llm.chat(messages)

        sources = [
            {
                "document_title": r.get("payload", {}).get("file_name", r.get("payload", {}).get("document_title", "")),
                "content": r.get("payload", {}).get("text", r.get("payload", {}).get("content", ""))[:200],
                "score": r.get("score", 0),
                "chunk_id": r.get("payload", {}).get("chunk_id", r.get("id", "")),
            }
            for r in search_results
        ]

        return {
            "answer": answer,
            "sources": sources,
        }

    async def generate_stream(
        self,
        question: str,
        history: Optional[list[dict]] = None,
        filters: Optional[dict] = None,
    ):
        """
        流式 RAG 生成

        Yields:
            ChatStreamChunk
        """
        search_results = await self.retrieve(question, filters=filters)
        context = self._build_context(search_results)

        messages = self._build_messages(question, context, history)

        sources = [
            {
                "document_title": r.get("payload", {}).get("file_name", r.get("payload", {}).get("document_title", "")),
                "content": r.get("payload", {}).get("text", r.get("payload", {}).get("content", ""))[:200],
                "score": r.get("score", 0),
                "chunk_id": r.get("payload", {}).get("chunk_id", r.get("id", "")),
            }
            for r in search_results
        ]

        yield ChatStreamChunk(type="source", sources=sources)

        for chunk in self.llm.chat_stream(messages):
            yield ChatStreamChunk(type="content", content=chunk)

        yield ChatStreamChunk(type="done")