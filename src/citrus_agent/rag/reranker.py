"""Qwen3-Reranker-4B 重排序客户端。

本文件只负责调用独立部署的 reranker HTTP 服务，不写检索逻辑。
当前默认兼容 vLLM 的 score 服务，后续如果模型服务换地址或换实现，
优先改这里，不影响 RAG API 的调用方式。
"""

from __future__ import annotations

import json
from urllib import request as url_request
from urllib.error import HTTPError, URLError

from src.citrus_agent.core.config import settings


class QwenRerankerClient:
    """Qwen3-Reranker-4B HTTP 客户端。

    Args:
        base_url: reranker 服务地址，例如 http://172.21.72.18:8002。
        model_name: vLLM 中暴露的模型名，例如 qwen3-reranker-4b。
        timeout: HTTP 请求超时时间。
        batch_size: 每次请求发送的候选片段数量。
    """

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.qwen_reranker_url).rstrip("/")
        self.model_name = model_name or settings.qwen_reranker_model
        self.timeout = timeout or settings.qwen_reranker_timeout
        self.batch_size = batch_size or settings.qwen_reranker_batch_size

    def score(self, query_text: str, documents: list[str]) -> list[float]:
        """给候选片段打相关性分数。

        Args:
            query_text: 用户问题。
            documents: 候选知识片段正文列表。

        Returns:
            list[float]: 每个候选片段对应的 rerank 分数，顺序和 documents 一致。
        """

        clean_query = query_text.strip()
        clean_documents = [document.strip() for document in documents]
        if not clean_query or not clean_documents:
            return []

        scores: list[float] = []
        for start in range(0, len(clean_documents), self.batch_size):
            batch = clean_documents[start : start + self.batch_size]
            scores.extend(self._request_score(clean_query, batch))
        return scores

    def _request_score(self, query_text: str, documents: list[str]) -> list[float]:
        """请求 vLLM score 接口。

        vLLM 的部署版本可能暴露 `/score` 或 `/v1/score`，这里优先访问 `/score`，
        如果服务返回 404 再自动尝试 `/v1/score`。
        """

        payload = {
            "model": self.model_name,
            "text_1": query_text,
            "text_2": documents,
        }

        try:
            return self._post_score("/score", payload, len(documents))
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise
            return self._post_score("/v1/score", payload, len(documents))

    def _post_score(
        self,
        path: str,
        payload: dict[str, object],
        expected_count: int,
    ) -> list[float]:
        """发送一次 HTTP 请求并解析分数。"""

        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = url_request.Request(
            url=f"{self.base_url}{path}",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with url_request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Qwen reranker 服务返回错误：HTTP {exc.code} {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"无法连接 Qwen reranker 服务：{self.base_url}") from exc

        scores = self._parse_scores(data)
        if len(scores) != expected_count:
            raise ValueError(
                f"Qwen reranker 返回数量和候选片段数量不一致：候选 {expected_count}，返回 {len(scores)}"
            )
        return scores

    def _parse_scores(self, data: dict[str, object]) -> list[float]:
        """兼容解析常见 score 响应格式。"""

        if isinstance(data.get("scores"), list):
            return [float(score) for score in data["scores"]]  # type: ignore[index]

        items = data.get("data")
        if not isinstance(items, list):
            items = data.get("results")
        if not isinstance(items, list):
            raise ValueError("Qwen reranker 响应缺少 scores/data/results 分数字段")

        score_items: list[dict[str, object]] = []
        for item in items:
            if not isinstance(item, dict) or "score" not in item:
                raise ValueError("Qwen reranker 响应中的分数项格式不正确")
            score_items.append(item)

        scores: list[float] = []
        for item in sorted(score_items, key=lambda value: int(value.get("index", 0))):
            scores.append(float(item["score"]))
        return scores
