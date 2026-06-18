"""
对话服务模块
管理会话、对话历史、工具调用决策
"""

import uuid
from typing import AsyncIterator, Optional

from src.citrus_agent.services.rag_service import RAGService
from src.citrus_agent.tools.base import ToolRegistry
from src.citrus_agent.pojo.chat import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    QuickCard,
)
from src.citrus_agent.services.logger import logger


# ==================== 预设快捷推荐卡片 ====================
QUICK_CARDS = [
    QuickCard(
        id="pest_query",
        title="病虫害查询",
        description="快速识别和查询柑橘常见病虫害",
        icon="bug",
        preset_question="柑橘常见的病虫害有哪些？如何防治？",
    ),
    QuickCard(
        id="price_trend",
        title="价格行情",
        description="查看近期柑橘市场价格走势",
        icon="chart",
        preset_question="近期柑橘市场价格走势如何？",
    ),
    QuickCard(
        id="fertilizer_calc",
        title="施肥计算",
        description="计算种植所需的肥料用量",
        icon="calculator",
        preset_question="帮我计算3亩柑橘地需要多少复合肥",
    ),
    QuickCard(
        id="policy_info",
        title="政策资讯",
        description="查询最新农业政策与补贴信息",
        icon="policy",
        preset_question="最新的柑橘种植补贴政策有哪些？",
    ),
    QuickCard(
        id="weather",
        title="天气影响",
        description="了解天气对柑橘生长的影响",
        icon="weather",
        preset_question="持续降雨对柑橘生长有什么影响？",
    ),
    QuickCard(
        id="variety_guide",
        title="品种指南",
        description="了解不同柑橘品种的特点",
        icon="book",
        preset_question="广西适合种植哪些柑橘品种？各有什么特点？",
    ),
]


# 这里进行简单处理，后续调用API，执行知识库问答
class ChatService:
    """对话服务"""

    def __init__(self):
        self.rag_service = RAGService()
        # 内存中的会话存储（后续可替换为 Redis/DB）
        self._conversations: dict[str, list[dict]] = {}

    def _get_or_create_conversation(
        self, conversation_id: Optional[str] = None
    ) -> tuple[str, list[dict]]:
        """
        获取或创建会话

        Args:
            conversation_id: 会话ID，为空则新建

        Returns:
            (conversation_id, messages)
        """
        if conversation_id and conversation_id in self._conversations:
            return conversation_id, self._conversations[conversation_id]

        new_id = conversation_id or str(uuid.uuid4())
        self._conversations[new_id] = []
        return new_id, self._conversations[new_id]

    def _detect_tool_call(self, question: str) -> Optional[dict]:
        """
        检测用户问题是否需要调用工具

        Args:
            question: 用户问题

        Returns:
            工具调用信息，或 None
        """
        # 简单的关键词匹配（后续可替换为 LLM function calling）
        question_lower = question.lower()

        # 肥料计算相关
        calc_keywords = ["计算", "多少肥", "施肥量", "用量", "需要买", "复合肥", "尿素"]
        if any(kw in question_lower for kw in calc_keywords):
            # 尝试提取面积
            area = self._extract_area(question)
            crop_type = self._extract_crop_type(question)
            if area and crop_type:
                return {
                    "tool_name": "fertilizer_calculator",
                    "parameters": {
                        "area": area,
                        "crop_type": crop_type,
                        "fertilizer_type": "复合肥",
                    },
                }

        return None

    @staticmethod
    def _extract_area(text: str) -> Optional[float]:
        """从文本中提取面积数值"""
        import re
        patterns = [r"(\d+(?:\.\d+)?)\s*亩", r"(\d+(?:\.\d+)?)\s*地"]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        return None

    @staticmethod
    def _extract_crop_type(text: str) -> Optional[str]:
        """从文本中提取作物类型"""
        crops = ["柑橘", "番茄", "辣椒", "白菜", "黄瓜", "水稻"]
        for crop in crops:
            if crop in text:
                return crop
        return None

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        处理对话请求（非流式）

        Args:
            request: 对话请求

        Returns:
            对话响应
        """
        conversation_id, history = self._get_or_create_conversation(
            request.conversation_id
        )

        # 检测是否需要工具调用
        tool_call = self._detect_tool_call(request.question)
        tool_used = None

        if tool_call:
            tool = ToolRegistry.get_tool(tool_call["tool_name"])
            if tool:
                tool_result = await tool.execute(**tool_call["parameters"])
                tool_used = tool.name

                # 将工具结果作为补充上下文
                if tool_result.get("success"):
                    tool_context = f"\n\n[工具计算结果] {tool_result['message']}"
                    # 仍然走 RAG 流程，但附加工具结果
                    rag_result = await self.rag_service.generate(
                        question=request.question + tool_context,
                        history=history,
                    )
                    answer = rag_result["answer"] + f"\n\n📊 {tool_result['message']}"
                    sources = rag_result["sources"]
                else:
                    answer = tool_result["message"]
                    sources = []
            else:
                rag_result = await self.rag_service.generate(
                    question=request.question, history=history
                )
                answer = rag_result["answer"]
                sources = rag_result["sources"]
        else:
            # 常规 RAG 问答
            rag_result = await self.rag_service.generate(
                question=request.question, history=history
            )
            answer = rag_result["answer"]
            sources = rag_result["sources"]

        # 更新对话历史
        history.append({"role": "user", "content": request.question})
        history.append({"role": "assistant", "content": answer})

        return ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            sources=sources,
            tool_used=tool_used,
        )

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncIterator[ChatStreamChunk]:
        """
        处理对话请求（流式）

        Args:
            request: 对话请求

        Yields:
            流式响应片段
        """
        conversation_id, history = self._get_or_create_conversation(
            request.conversation_id
        )

        # 检测工具调用
        tool_call = self._detect_tool_call(request.question)

        if tool_call:
            tool = ToolRegistry.get_tool(tool_call["tool_name"])
            if tool:
                tool_result = await tool.execute(**tool_call["parameters"])
                yield ChatStreamChunk(type="tool", tool_name=tool.name)

                if tool_result.get("success"):
                    tool_context = f"\n\n[工具计算结果] {tool_result['message']}"
                    async for chunk in self.rag_service.generate_stream(
                        question=request.question + tool_context,
                        history=history,
                    ):
                        yield chunk
                else:
                    yield ChatStreamChunk(type="content", content=tool_result["message"])
                    yield ChatStreamChunk(type="done")
                return

        # 常规 RAG 流式问答
        full_answer = ""
        async for chunk in self.rag_service.generate_stream(
            question=request.question, history=history
        ):
            if chunk.type == "content":
                full_answer += chunk.content or ""
            yield chunk

        # 更新对话历史
        history.append({"role": "user", "content": request.question})
        history.append({"role": "assistant", "content": full_answer})

    def get_quick_cards(self) -> list[QuickCard]:
        """获取预设快捷推荐卡片"""
        return QUICK_CARDS

    def get_chat_history(self, conversation_id: str) -> list[dict]:
        """
        获取对话历史

        Args:
            conversation_id: 会话ID

        Returns:
            消息列表
        """
        return self._conversations.get(conversation_id, [])
