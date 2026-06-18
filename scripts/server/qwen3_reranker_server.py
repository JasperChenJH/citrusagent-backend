"""Qwen3-Reranker-4B 独立重排服务。

这个文件用于部署在模型服务器上，通过 HTTP 接口给 query 和候选文档打相关性分数。

接口设计和项目里的 QwenRerankerClient 保持兼容：
    GET  /health
    POST /score

运行方式示例：
    uvicorn qwen3_reranker_server:app --host 0.0.0.0 --port 8000 --app-dir /app
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


def read_int_env(name: str, default: int) -> int:
    """读取整数环境变量。"""

    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def read_float_env(name: str, default: float) -> float:
    """读取浮点数环境变量。"""

    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


MODEL_NAME = os.getenv("QWEN_RERANKER_MODEL_NAME", "Qwen/Qwen3-Reranker-4B")
DEVICE = os.getenv("QWEN_RERANKER_DEVICE", "cuda")
DEFAULT_BATCH_SIZE = read_int_env("QWEN_RERANKER_BATCH_SIZE", 8)
MAX_BATCH_SIZE = read_int_env("QWEN_RERANKER_MAX_BATCH_SIZE", 64)
MAX_LENGTH = read_int_env("QWEN_RERANKER_MAX_LENGTH", 8192)
MIN_SCORE = read_float_env("QWEN_RERANKER_MIN_SCORE", 0.0)
MAX_SCORE = read_float_env("QWEN_RERANKER_MAX_SCORE", 1.0)

app = FastAPI(
    title="Qwen3 Reranker Service",
    description="Qwen3-Reranker-4B HTTP 重排服务，返回 query-document 相关性分数。",
    version="1.0.0",
)

model_lock = threading.Lock()
model_loaded_at = time.time()


class ScoreRequest(BaseModel):
    """重排打分请求体。

    text_1 是用户问题，text_2 是候选 chunk 文本列表。
    字段命名保持 vLLM score 风格，方便项目客户端复用。
    """

    model: str | None = Field(default=None, description="可选模型名，服务端当前只做记录。")
    text_1: str = Field(..., description="用户问题。")
    text_2: list[str] = Field(..., description="候选文档或 chunk 文本列表。")
    batch_size: int | None = Field(default=None, ge=1, description="可选批大小。")

    @field_validator("text_1")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """清理并校验 query。"""

        clean_value = value.strip() if isinstance(value, str) else ""
        if not clean_value:
            raise ValueError("text_1 不能为空")
        return clean_value

    @field_validator("text_2")
    @classmethod
    def validate_documents(cls, values: list[str]) -> list[str]:
        """清理并校验候选文本。"""

        if not values:
            raise ValueError("text_2 不能为空")

        clean_values: list[str] = []
        for value in values:
            clean_value = value.strip() if isinstance(value, str) else ""
            if not clean_value:
                raise ValueError("text_2 中不能包含空文本")
            clean_values.append(clean_value)
        return clean_values


def load_model():
    """加载 Qwen3-Reranker-4B CrossEncoder。

    使用 sentence-transformers 官方 CrossEncoder 方式加载 reranker。
    predict 时使用 Sigmoid，把 raw logit 转成 0 到 1 的概率型分数。
    """

    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "缺少依赖，请先安装：pip install sentence-transformers transformers accelerate"
        ) from exc

    model_kwargs: dict[str, Any] = {}
    if DEVICE.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16

    try:
        return CrossEncoder(
            MODEL_NAME,
            device=DEVICE,
            max_length=MAX_LENGTH,
            model_kwargs=model_kwargs,
        )
    except TypeError:
        # 兼容旧版 sentence-transformers，没有 model_kwargs 参数时退回普通加载。
        return CrossEncoder(
            MODEL_NAME,
            device=DEVICE,
            max_length=MAX_LENGTH,
        )


model = load_model()


@app.get("/health")
def health() -> dict[str, Any]:
    """健康检查接口。"""

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "max_length": MAX_LENGTH,
        "max_batch_size": MAX_BATCH_SIZE,
        "score_range": [MIN_SCORE, MAX_SCORE],
        "loaded_seconds": round(time.time() - model_loaded_at, 3),
    }


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    """给 query 和候选文本打相关性分数。

    返回格式：
        {
            "data": [
                {"index": 0, "score": 0.91},
                {"index": 1, "score": 0.12}
            ]
        }
    """

    if len(request.text_2) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"单次最多处理 {MAX_BATCH_SIZE} 条候选文本，当前收到 {len(request.text_2)} 条",
        )

    batch_size = request.batch_size or DEFAULT_BATCH_SIZE
    pairs = [(request.text_1, document) for document in request.text_2]

    try:
        import torch
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="缺少 torch 依赖") from exc

    with model_lock:
        scores = model.predict(
            pairs,
            batch_size=batch_size,
            activation_fn=torch.nn.Sigmoid(),
        )

    scores = to_float_list(scores)
    return {
        "model": request.model or MODEL_NAME,
        "data": [
            {
                "index": index,
                "score": clamp_score(score_value),
            }
            for index, score_value in enumerate(scores)
        ],
    }


@app.post("/v1/score")
def v1_score(request: ScoreRequest) -> dict[str, Any]:
    """兼容 /v1/score 路径。"""

    return score(request)


def to_float_list(values: Any) -> list[float]:
    """把 numpy / torch / list 分数统一转换成 Python float list。"""

    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, float):
        return [values]
    return [float(value) for value in values]


def clamp_score(value: float) -> float:
    """把分数限制在配置范围内，避免异常值影响排序。"""

    return max(MIN_SCORE, min(MAX_SCORE, float(value)))
