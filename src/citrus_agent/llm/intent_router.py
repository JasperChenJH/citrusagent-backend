"""意图路由器。

本文件负责判断用户问题是否需要检索知识库（方案A）。
先用小成本调用 LLM 做意图分类，再决定是否触发 RAG 检索。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.citrus_agent.llm.base import ChatLLM


@dataclass
class IntentResult:
    """意图判断结果。

    Attributes:
        need_rag: 是否需要检索知识库。
        kb_id: 要检索的知识库 ID，不需要检索时为 None。
        reasoning: LLM 给出的判断理由（可选，用于调试日志）。
    """

    need_rag: bool
    kb_id: int | None = None
    reasoning: str = ""


# 意图分类的通用 JSON 格式约束
_INTENT_JSON_FORMAT = (
    '请严格按照 JSON 格式输出，键名必须一致：\n'
    '{"need_rag": true/false, "kb_id": null 或 整数, "reasoning": "简要说明"}\n'
    '只输出一行 JSON，不要输出其他内容。\n'
)


class IntentRouter:
    """意图路由器。

    把用户问题发给 LLM，让 LLM 返回 JSON 格式的判断结果。
    不同路由可以传入不同的分类 prompt，实现差异化判断。

    调用方式：
        router = IntentRouter()
        result = router.classify("砂糖橘溃疡病怎么防？", route="agri")
    """

    def __init__(self, llm: ChatLLM | None = None) -> None:
        self.llm = llm or ChatLLM(temperature=0.0, max_tokens=256)

    def classify(self, user_question: str, route: str = "general") -> IntentResult:
        """判断用户意图，返回是否需要 RAG 检索。

        Args:
            user_question: 用户问题。
            route: 当前路由标识，例如 "general"、"agri"、"diagnose"。
                    不同路由使用不同的分类 prompt。

        Returns:
            IntentResult: 包含 need_rag、kb_id 和判断理由。
        """

        classification_prompt = self._get_classification_prompt(route)

        messages = [
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": user_question},
        ]

        raw = self.llm.chat(messages)
        return self._parse_intent(raw)

    def _get_classification_prompt(self, route: str) -> str:
        """根据路由获取意图分类的 system prompt。"""

        base = (
            "你是 YAGO intellect 意图分类器。请判断用户问题是否需要查询知识库来回答。\n"
            + _INTENT_JSON_FORMAT
        )

        if route == "general":
            return (
                base
                + "当前路由：通用问答。\n"
                + "分类规则：\n"
                + "- 问候、闲聊、简单常识、自我介绍 → need_rag=false\n"
                + "- 需要专业知识才能准确回答的问题 → need_rag=true, kb_id=1\n"
            )

        if route == "agri":
            return (
                base
                + "当前路由：农业知识库。\n"
                + "分类规则：\n"
                + "- 柑橘种植技术、施肥管理、土壤改良、品种特性等农业专业问题 → need_rag=true, kb_id=2\n"
                + "- 病虫害基础常识（非具体诊断）→ need_rag=true, kb_id=2\n"
                + "- 问候闲聊、非农业话题 → need_rag=false\n"
            )

        if route == "diagnose":
            return (
                base
                + "当前路由：病虫害诊断。\n"
                + "分类规则：\n"
                + "- 植物病虫害症状识别、防治方案、农药使用等诊断类问题 → need_rag=true, kb_id=3\n"
                + "- 病虫害基础知识科普 → need_rag=true, kb_id=3\n"
                + "- 问候闲聊、非诊断话题 → need_rag=false\n"
            )

        # 未知路由，默认不做 RAG
        return base + f"当前路由：{route}。默认不检索知识库。"

    def _parse_intent(self, raw: str) -> IntentResult:
        """从 LLM 返回文本中解析 JSON 意图结果。"""

        try:
            match = re.search(r"\{[^{}]*\"need_rag\"[^{}]*\}", raw)
            if not match:
                return IntentResult(need_rag=False, reasoning="LLM 未返回合法 JSON，默认不检索")

            data = json.loads(match.group())
            return IntentResult(
                need_rag=bool(data.get("need_rag", False)),
                kb_id=data.get("kb_id"),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, TypeError):
            return IntentResult(need_rag=False, reasoning="JSON 解析失败，默认不检索")
