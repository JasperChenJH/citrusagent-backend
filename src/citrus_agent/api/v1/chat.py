"""聊天接口路由（流式）。

本文件定义三个前端聊天入口的 HTTP 流式接口：
- /chat/general   → YAGO intellect 通用问答
- /chat/agri      → YAGO intellect 农业知识库
- /chat/diagnose  → YAGO intellect 病虫害诊断

每个接口都走：意图判断 → 按需检索 → LLM 流式回答 的流程。
返回 SSE (Server-Sent Events) 格式："data: {\"delta\": \"...\"}\\n\\n"。
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.citrus_agent.core.config import settings
from src.citrus_agent.llm.base import ChatLLM
from src.citrus_agent.llm.intent_router import IntentRouter
from src.citrus_agent.rag.retriever import CitrusRetriever

router = APIRouter(prefix=settings.api_v1_prefix + "/chat", tags=["chat"])

_llm: ChatLLM | None = None
_retriever: CitrusRetriever | None = None
_intent_router: IntentRouter | None = None


def _get_llm() -> ChatLLM:
    global _llm
    if _llm is None:
        _llm = ChatLLM()
    return _llm


def _get_retriever() -> CitrusRetriever:
    global _retriever
    if _retriever is None:
        _retriever = CitrusRetriever()
    return _retriever


def _get_intent_router() -> IntentRouter:
    global _intent_router
    if _intent_router is None:
        _intent_router = IntentRouter()
    return _intent_router

# ====================================================================
# System Prompt 模板区（参考西智蔗 prompt 结构，防止流式输出格式异常）
# ====================================================================

# 输出格式约束（三个路由共用）
_OUTPUT_FORMAT_RULES = """输出格式要求（必须严格遵守）：
- 使用 HTML 标签输出内容，<p> 包裹段落，重点内容用 <b> 加粗。
- 列表型内容用 <ul> / <li> 或有序 <ol> / <li>。
- 严禁输出 Markdown 语法（**、#、- 列表等），严禁输出 <think>、</think> 标签。
- 严禁在回答中夹带 JSON、代码块或多余的格式化标记。
"""

# 防幻觉规则（三个路由共用）
_ANTI_HALLUCINATION_RULES = """防幻觉规则：
- 不确定的信息必须如实说明"暂未收录"或"无法确定"，严禁编造数据、文献或结论。
- 引用知识片段时保持原文信息，不要自行推断原文未提及的内容。
"""

# ── 通用问答 direct prompt ──
_GENERAL_DIRECT_PROMPT = """你是 YAGO intellect<b>通用问答助手</b>，由农业科研团队打造的综合智能服务平台。

身份信息：
- 名称：YAGO intellect 通用问答助手
- 定位：面向农业科研与日常咨询的通用智能对话引擎

核心能力：
1. 农业领域基础知识问答（种植技术、施肥管理、病虫害防治常识等）。
2. 科研文献解读与学习方法指导。
3. 日常对话与信息咨询。

""" + _OUTPUT_FORMAT_RULES + _ANTI_HALLUCINATION_RULES

# ── 农业知识库 direct prompt ──
_AGRI_DIRECT_PROMPT = """你是 YAGO intellect<b>农业知识 AI 助手</b>，专为农业知识领域提供智能问答服务。

身份信息：
- 名称：YAGO intellect 农业知识 AI 助手
- 定位：专注柑橘及农业知识库的精准问答引擎

核心能力：
1. 柑橘种植技术问答（品种选择、栽培管理、水肥一体化、修剪技术）。
2. 土壤肥料管理（测土配方、有机肥施用、微量元素补充）。
3. 病虫害综合防治（IPM 策略、生物防治、化学防治安全间隔期）。
4. 农业政策与市场信息解读。

""" + _OUTPUT_FORMAT_RULES + _ANTI_HALLUCINATION_RULES

# ── 病虫害诊断 direct prompt ──
_DIAGNOSE_DIRECT_PROMPT = """你是 YAGO intellect<b>病虫害诊断助手</b>，专为植物病虫害识别与防治提供智能服务。

身份信息：
- 名称：YAGO intellect 病虫害诊断助手
- 定位：面向柑橘及经济作物的病虫害识别、诊断与防治建议引擎

核心能力：
1. 病虫害症状识别（叶片、果实、枝干、根系异常分析）。
2. 病原鉴定与虫害种类判断。
3. 综合防治方案制定（化学防治、生物防治、农业防治相结合）。
4. 防治时机与用药安全间隔期指导。

""" + _OUTPUT_FORMAT_RULES + """诊断专用规则：
- 症状不明确时，主动追问用户更多细节（发病部位、颜色、形状、发生时间、环境条件等），不要猜测。
- 给出防治方案时标注安全间隔期和注意事项。
""" + _ANTI_HALLUCINATION_RULES

# ── 通用问答 RAG prompt ──
_GENERAL_RAG_PROMPT = """你是 YAGO intellect<b>通用问答助手</b>，基于知识库内容为用户提供准确回答。

身份信息：
- 名称：YAGO intellect 通用问答助手
- 定位：基于知识库检索结果的智能问答引擎

""" + _OUTPUT_FORMAT_RULES + """基于知识库的回答规则：
- 优先引用知识片段中的具体信息回答。
- 知识片段不足时，先声明"根据现有知识库暂无直接相关信息"，再基于通用知识保守补充。
- 回答末尾标注引用来源的文件名或标题。
""" + _ANTI_HALLUCINATION_RULES + """

知识片段：
{context}"""

# ── 农业知识库 RAG prompt ──
_AGRI_RAG_PROMPT = """你是 YAGO intellect<b>农业知识 AI 助手</b>，基于农业知识库内容为用户提供精准回答。

身份信息：
- 名称：YAGO intellect 农业知识 AI 助手
- 定位：专注柑橘及农业知识库的检索增强问答引擎

""" + _OUTPUT_FORMAT_RULES + """基于知识库的回答规则：
- 严格基于知识片段内容回答，每个关键结论都要能追溯到知识片段。
- 对柑橘品种名、病害名、农药名、技术术语进行 <b>加粗</b>。
- 专业术语保留中英文对照，如：<b>溃疡病 (Citrus Canker)</b>。
- 知识片段不足以回答时，先声明"当前知识库暂未收录该信息"，再基于农业常识保守补充。
""" + _ANTI_HALLUCINATION_RULES + """

知识片段：
{context}"""

# ── 病虫害诊断 RAG prompt ──
_DIAGNOSE_RAG_PROMPT = """你是 YAGO intellect<b>病虫害诊断助手</b>，基于知识库内容进行病虫害识别与防治方案推荐。

身份信息：
- 名称：YAGO intellect 病虫害诊断助手
- 定位：基于知识库检索的病虫害诊断与防治方案引擎

""" + _OUTPUT_FORMAT_RULES + """基于知识库的诊断规则：
- 从知识片段中匹配症状特征，给出最可能的病虫害类型及置信度说明。
- 列出具体的防治方案（化学药剂名称、浓度、施用方法、安全间隔期）。
- 知识片段不足以诊断时，主动列出需要用户补充的症状描述信息，不要猜测。
- 注意区分相似症状的病害（如溃疡病与疮痂病），需要时对比说明。
""" + _ANTI_HALLUCINATION_RULES + """

知识片段：
{context}"""


class ChatRequest(BaseModel):
    """聊天请求体。"""

    question: str = Field(..., description="用户问题", min_length=1)
    kb_id: int | None = Field(
        default=None,
        description="手动指定知识库 ID，不为空时跳过意图判断直接检索",
    )


def _handle_chat_stream(
    question: str,
    route: str,
    kb_id: int | None,
) -> Generator[str, None, None]:
    """流式聊天处理。

    返回 SSE 格式：
        data: {"delta": "文本片段"}\n\n
        data: {"delta": "", "done": true, "from_knowledge_base": true, "kb_id": 3}\n\n
    """

    from_knowledge_base = False
    resolved_kb_id: int | None = None
    reasoning = ""

    # 1. 决定是否走 RAG
    if kb_id is not None:
        search_results = _get_retriever().search(question, filters={"kb_id": kb_id})
        context_chunks = [r.content for r in search_results] if search_results else []
        resolved_kb_id = kb_id
        reasoning = "前端指定知识库，直接检索"
    else:
        intent = _get_intent_router().classify(question, route=route)
        reasoning = intent.reasoning
        if intent.need_rag and intent.kb_id is not None:
            search_results = _get_retriever().search(question, filters={"kb_id": intent.kb_id})
            context_chunks = [r.content for r in search_results] if search_results else []
            resolved_kb_id = intent.kb_id
        else:
            context_chunks = []

    # 2. 流式生成回答
    if context_chunks:
        from_knowledge_base = True
        system_prompt = _get_rag_prompt(route)
        stream = _get_llm().chat_with_context_stream(question, context_chunks, system_prompt)
    else:
        system_prompt = _get_direct_prompt(route)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        stream = _get_llm().chat_stream(messages)

    for delta in stream:
        yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"

    # 3. 发送结束事件，携带元信息
    done_payload: dict[str, Any] = {
        "delta": "",
        "done": True,
        "from_knowledge_base": from_knowledge_base,
        "reasoning": reasoning,
    }
    if resolved_kb_id is not None:
        done_payload["kb_id"] = resolved_kb_id
    yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"


def _get_direct_prompt(route: str) -> str:
    """获取直接回答时的 system prompt（不走 RAG 检索）。"""

    if route == "general":
        return _GENERAL_DIRECT_PROMPT

    if route == "agri":
        return _AGRI_DIRECT_PROMPT

    if route == "diagnose":
        return _DIAGNOSE_DIRECT_PROMPT

    return _GENERAL_DIRECT_PROMPT


def _get_rag_prompt(route: str) -> str:
    """获取 RAG 检索后的 system prompt。

    返回值中包含 {context} 占位符，由 ChatLLM._build_rag_messages 自动替换为真实检索片段。
    """

    if route == "general":
        return _GENERAL_RAG_PROMPT

    if route == "agri":
        return _AGRI_RAG_PROMPT

    if route == "diagnose":
        return _DIAGNOSE_RAG_PROMPT

    return _GENERAL_RAG_PROMPT


# ── 三个前端路由 ──────────────────────────────────────


@router.post("/general")
def chat_general(request: ChatRequest):
    """通用问答入口。"""

    return StreamingResponse(
        _handle_chat_stream(request.question, route="general", kb_id=request.kb_id),
        media_type="text/event-stream",
    )


@router.post("/agri")
def chat_agri(request: ChatRequest):
    """农业知识库入口。"""

    return StreamingResponse(
        _handle_chat_stream(request.question, route="agri", kb_id=request.kb_id),
        media_type="text/event-stream",
    )


@router.post("/diagnose")
def chat_diagnose(request: ChatRequest):
    """病虫害诊断入口。"""

    return StreamingResponse(
        _handle_chat_stream(request.question, route="diagnose", kb_id=request.kb_id),
        media_type="text/event-stream",
    )
