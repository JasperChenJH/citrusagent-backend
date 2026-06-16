"""大模型调用客户端。

本文件封装大模型 API（OpenAI 兼容）的调用逻辑。上层只需要传 prompt，
不用关心 API 细节。
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from src.citrus_agent.core.config import settings


class ChatLLM:
    """LLM 聊天模型客户端。

    支持任何 OpenAI 兼容协议的模型服务（DeepSeek / MiniMax / Ollama / vLLM）。

    调用方式：
        llm = ChatLLM()
        answer = llm.chat(messages=[{"role": "user", "content": "你好"}])
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
        self.api_key = api_key or settings.deepseek_api_key
        self.base_url = base_url or settings.deepseek_base_url
        self.model = model or settings.deepseek_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Any = None

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """发送对话消息，返回完整回复文本。

        Args:
            messages: OpenAI 格式的消息列表，每条含 role 和 content。
            temperature: 温度参数，不传则用实例默认值。
            max_tokens: 最大输出 token 数，不传则用实例默认值。

        Returns:
            str: 模型回复的文本内容。
        """

        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            stream=False,
        )

        return response.choices[0].message.content or ""

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """流式对话，逐块 yield 模型回复文本。

        Args:
            messages: OpenAI 格式的消息列表。
            temperature: 温度参数。
            max_tokens: 最大输出 token 数。

        Yields:
            str: 每次 yield 一小段 delta 文本。
        """

        client = self._get_client()

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat_with_context(
        self,
        user_question: str,
        context_chunks: list[str],
        system_prompt: str | None = None,
    ) -> str:
        """带知识库上下文的问答（非流式）。"""

        messages = self._build_rag_messages(user_question, context_chunks, system_prompt)
        return self.chat(messages)

    def chat_with_context_stream(
        self,
        user_question: str,
        context_chunks: list[str],
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        """带知识库上下文的流式问答，逐块 yield。"""

        messages = self._build_rag_messages(user_question, context_chunks, system_prompt)
        yield from self.chat_stream(messages)

    def _build_rag_messages(
        self,
        user_question: str,
        context_chunks: list[str],
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """构造 RAG 问答的 messages。

        Args:
            user_question: 用户问题文本。
            context_chunks: 检索到的知识片段列表。
            system_prompt: 自定义 system prompt。如果包含 {context} 占位符，
                          会自动替换为实际检索片段；否则在末尾追加知识片段。

        Returns:
            包含 system + user 的 messages 列表。
        """

        context_text = "\n---\n".join(context_chunks) if context_chunks else ""

        if system_prompt is None:
            system_prompt = (
                "你是 YAGO intellect 知识库助手，请根据以下知识片段回答用户问题。\n"
                "如果知识片段不足以回答问题，请如实说明。\n\n"
                f"知识片段：\n{context_text}"
            )
        elif "{context}" in system_prompt:
            # 有自定义 prompt 且含 {context} 占位符：替换为实际检索片段
            system_prompt = system_prompt.replace("{context}", context_text)
        elif context_text:
            # 有自定义 prompt 但不含占位符：自动追加知识片段到末尾
            system_prompt = system_prompt + f"\n\n知识片段：\n{context_text}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ]

    def _get_client(self) -> Any:
        """懒加载 OpenAI 客户端。

        本地部署模型（localhost/127.0.0.1）允许 API Key 为空。
        """

        if self._client is not None:
            return self._client

        is_local = (
            "localhost" in self.base_url
            or "127.0.0.1" in self.base_url
            or "0.0.0.0" in self.base_url
        )

        if not is_local and (not self.api_key or self.api_key == "replace-me"):
            raise ValueError("DEEPSEEK_API_KEY 不能为空，请在 .env 中配置真实密钥")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("调用 LLM API 需要安装 openai") from exc

        api_key = self.api_key if self.api_key and self.api_key != "replace-me" else "not-needed"
        self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client
