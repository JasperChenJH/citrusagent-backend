"""BGE-M3 Embedding 独立服务。

这个文件用于部署在模型服务器上，通过 HTTP 接口把文本转换为：
1. dense vector：语义向量，BGE-M3 默认 1024 维。
2. sparse vector：稀疏关键词向量，后续写入 Qdrant 做 hybrid search。

运行方式示例：
    uvicorn bge_m3_server:app --host 0.0.0.0 --port 8000 --app-dir /app
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


def read_bool_env(name: str, default: bool) -> bool:
    """读取布尔环境变量。"""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def read_int_env(name: str, default: int) -> int:
    """读取整数环境变量。"""

    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


MODEL_NAME = os.getenv("BGE_M3_MODEL_NAME", "BAAI/bge-m3")
DEVICE = os.getenv("BGE_M3_DEVICE", "cuda")
USE_FP16 = read_bool_env("BGE_M3_USE_FP16", True)
DEFAULT_BATCH_SIZE = read_int_env("BGE_M3_BATCH_SIZE", 8)
DEFAULT_MAX_LENGTH = read_int_env("BGE_M3_MAX_LENGTH", 8192)
MAX_BATCH_SIZE = read_int_env("BGE_M3_MAX_BATCH_SIZE", 64)
DENSE_DIM = read_int_env("BGE_M3_DENSE_DIM", 1024)

app = FastAPI(
    title="BGE-M3 Embedding Service",
    description="返回 BGE-M3 dense + sparse 向量的独立服务。",
    version="1.0.0",
)

model_lock = threading.Lock()
model_loaded_at = time.time()


class EmbedRequest(BaseModel):
    """生成向量的请求体。"""

    texts: list[str] = Field(..., description="需要向量化的文本列表。")
    batch_size: int | None = Field(default=None, ge=1, description="可选批大小。")
    max_length: int | None = Field(default=None, ge=1, description="可选最大长度。")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        """清理文本，并拒绝空文本。"""

        if not texts:
            raise ValueError("texts 不能为空")

        clean_texts: list[str] = []
        for text in texts:
            clean_text = text.strip() if isinstance(text, str) else ""
            if not clean_text:
                raise ValueError("texts 中不能包含空文本")
            clean_texts.append(clean_text)
        return clean_texts


def load_model():
    """加载 BGE-M3 模型。

    不同版本 FlagEmbedding 的初始化参数可能略有差异，所以这里做了兼容兜底。
    """

    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise RuntimeError("缺少 FlagEmbedding，请先安装：pip install FlagEmbedding") from exc

    init_errors: list[Exception] = []
    init_kwargs_list = [
        {"use_fp16": USE_FP16, "devices": DEVICE},
        {"use_fp16": USE_FP16, "device": DEVICE},
        {"use_fp16": USE_FP16},
    ]

    for init_kwargs in init_kwargs_list:
        try:
            return BGEM3FlagModel(MODEL_NAME, **init_kwargs)
        except TypeError as exc:
            init_errors.append(exc)
            continue

    raise RuntimeError(f"BGE-M3 模型初始化失败：{init_errors[-1]}")


model = load_model()


@app.get("/health")
def health() -> dict[str, Any]:
    """健康检查接口。"""

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "use_fp16": USE_FP16,
        "dense_dim": DENSE_DIM,
        "max_batch_size": MAX_BATCH_SIZE,
        "loaded_seconds": round(time.time() - model_loaded_at, 3),
    }


@app.post("/embed")
def embed(request: EmbedRequest) -> dict[str, Any]:
    """生成 dense + sparse 向量。

    返回格式专门面向 Qdrant hybrid search：
    - dense：直接写入 Qdrant named dense vector。
    - sparse.indices / sparse.values：写入 Qdrant SparseVector。
    """

    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"单次最多处理 {MAX_BATCH_SIZE} 条文本，当前收到 {len(request.texts)} 条",
        )

    batch_size = request.batch_size or DEFAULT_BATCH_SIZE
    max_length = request.max_length or DEFAULT_MAX_LENGTH

    with model_lock:
        output = model.encode(
            request.texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    dense_vectors = output.get("dense_vecs")
    lexical_weights = output.get("lexical_weights")

    if dense_vectors is None or lexical_weights is None:
        raise HTTPException(status_code=500, detail="BGE-M3 未返回 dense_vecs 或 lexical_weights")

    results = []
    for index, text in enumerate(request.texts):
        dense = to_float_list(dense_vectors[index])
        sparse = lexical_weights[index]
        indices, values = sparse_to_indices_values(sparse)

        results.append(
            {
                "index": index,
                "text_length": len(text),
                "dense": dense,
                "sparse": {
                    "indices": indices,
                    "values": values,
                },
            }
        )

    return {
        "model": MODEL_NAME,
        "dense_dim": DENSE_DIM,
        "results": results,
    }


def to_float_list(vector: Any) -> list[float]:
    """把 numpy / torch / list 向量统一转换成 Python float list。"""

    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def sparse_to_indices_values(sparse: Any) -> tuple[list[int], list[float]]:
    """把 BGE-M3 lexical_weights 转成 Qdrant 需要的 indices / values。

    FlagEmbedding 常见返回格式是：
        {"123": 0.8, "456": 0.3}
    这里统一转换成：
        indices=[123, 456]
        values=[0.8, 0.3]
    """

    if not isinstance(sparse, dict):
        raise HTTPException(status_code=500, detail="BGE-M3 sparse 结果格式不是 dict")

    pairs: list[tuple[int, float]] = []
    for token_id, weight in sparse.items():
        value = float(weight)
        if value == 0.0:
            continue
        pairs.append((int(token_id), value))

    pairs.sort(key=lambda item: item[0])
    indices = [token_id for token_id, _ in pairs]
    values = [value for _, value in pairs]
    return indices, values
