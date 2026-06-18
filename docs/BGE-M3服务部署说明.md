# BGE-M3 Embedding 服务部署说明

本文档说明如何在甲方服务器 `172.21.72.18` 上部署独立的 BGE-M3 Embedding 服务。

该服务使用：

```text
BAAI/bge-m3
```

并通过 HTTP 接口返回：

```text
dense vector + sparse vector
```

后续 RAG 入库和检索会调用该服务生成向量，再写入 Qdrant 做 hybrid search。

## 一、服务说明

本项目已经提供服务文件：

```text
scripts/server/bge_m3_server.py
```

这个文件会在服务器上作为 FastAPI 服务运行。

服务启动后提供两个接口：

```text
GET  /health
POST /embed
```

端口规划：

```text
宿主机访问：http://172.21.72.18:8001
容器内访问：http://bge-m3-embedder:8000
```

## 二、服务器前置检查

先登录服务器：

```bash
ssh 用户名@172.21.72.18
```

检查 Docker：

```bash
docker --version
```

检查 GPU：

```bash
nvidia-smi
```

检查 Docker 是否支持 GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

如果这条命令失败，说明服务器还没有配置好 `nvidia-container-toolkit`，需要先让服务器管理员处理 GPU Docker 环境。

## 三、创建服务器目录

在服务器上执行：

```bash
mkdir -p /data/rag_services
mkdir -p /data/models/huggingface
```

目录作用：

| 路径 | 作用 |
| --- | --- |
| `/data/rag_services` | 存放 `bge_m3_server.py` 服务文件 |
| `/data/models/huggingface` | 存放 HuggingFace 模型缓存，避免每次重启重复下载 |

## 四、上传 bge_m3_server.py

从你的本地电脑上传文件到服务器。

如果你在 Windows PowerShell 里操作，可以执行：

```powershell
scp G:\py_workplace\CitrusAgent\scripts\server\bge_m3_server.py 用户名@172.21.72.18:/data/rag_services/bge_m3_server.py
```

把命令里的 `用户名` 换成甲方服务器的真实登录用户。

上传后，在服务器上检查：

```bash
ls -lh /data/rag_services/bge_m3_server.py
```

如果能看到文件，说明上传成功。

## 五、创建 Docker 网络

在服务器上执行：

```bash
docker network create rag-net
```

如果提示网络已经存在，可以忽略。

这个网络用于后续让多个容器互相访问，例如：

```text
bge-m3-embedder
qwen3-reranker-4b
qdrant
backend-api
```

## 六、启动 BGE-M3 容器

如果服务器有多张 GPU，建议指定一张空闲 GPU。比如甲方服务器上 GPU 1 空闲时，使用：

```bash
sudo docker run -d \
  --name bge-m3-embedder \
  --network rag-net \
  --gpus '"device=1"' \
  --ipc=host \
  -p 8001:8000 \
  -v /data/models/huggingface:/root/.cache/huggingface \
  -v /data/rag_services/bge_m3_server.py:/app/bge_m3_server.py:ro \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e BGE_M3_MODEL_NAME=BAAI/bge-m3 \
  -e BGE_M3_DEVICE=cuda \
  -e BGE_M3_USE_FP16=true \
  -e BGE_M3_BATCH_SIZE=8 \
  -e BGE_M3_MAX_LENGTH=8192 \
  -e BGE_M3_MAX_BATCH_SIZE=64 \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
  bash -lc "pip install -U FlagEmbedding fastapi uvicorn transformers accelerate && uvicorn bge_m3_server:app --host 0.0.0.0 --port 8000 --app-dir /app"
```

容器内部只会看到分配给它的 GPU，所以容器内显示 `GPU 0` 是正常现象。

### 命令解释

| 参数 | 说明 |
| --- | --- |
| `--name bge-m3-embedder` | 容器名称 |
| `--network rag-net` | 加入 RAG 专用 Docker 网络 |
| `--gpus '"device=1"'` | 只使用宿主机 GPU 1，避免占用其他任务的 GPU |
| `--ipc=host` | 提高 PyTorch 多进程/共享内存稳定性 |
| `-p 8001:8000` | 宿主机 8001 映射到容器内 8000 |
| `/data/models/huggingface:/root/.cache/huggingface` | 挂载模型缓存目录 |
| `/data/rag_services/bge_m3_server.py:/app/bge_m3_server.py:ro` | 挂载服务文件，只读 |
| `HF_ENDPOINT=https://hf-mirror.com` | 使用 HuggingFace 镜像站，适合国内服务器 |
| `BGE_M3_MODEL_NAME=BAAI/bge-m3` | 指定模型名称 |
| `BGE_M3_DEVICE=cuda` | 使用 GPU |
| `BGE_M3_USE_FP16=true` | 使用 fp16，节省显存 |
| `BGE_M3_BATCH_SIZE=8` | 默认批处理大小 |
| `BGE_M3_MAX_LENGTH=8192` | BGE-M3 最大输入长度 |
| `BGE_M3_MAX_BATCH_SIZE=64` | 单次请求最多文本条数 |

第一次启动会下载模型和安装依赖，时间会比较久。

### 推荐的稳定启动方式：本地模型路径

如果自动下载模型遇到 `403` 或坏缓存，推荐先把模型下载到宿主机：

```text
/data/models/bge-m3
```

然后用本地路径启动：

```bash
sudo docker run -d \
  --name bge-m3-embedder \
  --network rag-net \
  --gpus '"device=1"' \
  --ipc=host \
  -p 8001:8000 \
  -v /data/models:/data/models \
  -v /data/rag_services/bge_m3_server.py:/app/bge_m3_server.py:ro \
  -e BGE_M3_MODEL_NAME=/data/models/bge-m3 \
  -e BGE_M3_DEVICE=cuda \
  -e BGE_M3_USE_FP16=true \
  -e BGE_M3_BATCH_SIZE=8 \
  -e BGE_M3_MAX_LENGTH=8192 \
  -e BGE_M3_MAX_BATCH_SIZE=64 \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
  bash -lc "pip install -U torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 && pip install -U FlagEmbedding fastapi uvicorn transformers accelerate && uvicorn bge_m3_server:app --host 0.0.0.0 --port 8000 --app-dir /app"
```

这里升级 `torch==2.6.0` 是为了避免加载 `.bin` 权重时触发 `torch.load` 安全限制。

## 七、查看容器状态

查看容器是否启动：

```bash
docker ps
```

查看日志：

```bash
docker logs -f bge-m3-embedder
```

如果模型下载成功并且服务启动成功，日志里应该能看到 uvicorn 正在监听：

```text
Uvicorn running on http://0.0.0.0:8000
```

## 八、健康检查

在服务器上执行：

```bash
curl http://127.0.0.1:8001/health
```

或者从其他机器访问：

```bash
curl http://172.21.72.18:8001/health
```

预期返回：

```json
{
  "status": "ok",
  "model": "BAAI/bge-m3",
  "device": "cuda",
  "use_fp16": true,
  "dense_dim": 1024,
  "max_batch_size": 64
}
```

## 九、测试向量生成

执行：

```bash
curl -X POST http://172.21.72.18:8001/embed \
  -H "Content-Type: application/json" \
  -d '{"texts":["砂糖橘溃疡病怎么防？"]}'
```

预期返回结构：

```json
{
  "model": "BAAI/bge-m3",
  "dense_dim": 1024,
  "results": [
    {
      "index": 0,
      "text_length": 10,
      "dense": [0.01, 0.02],
      "sparse": {
        "indices": [123, 456],
        "values": [0.8, 0.3]
      }
    }
  ]
}
```

验收标准：

```text
dense_dim = 1024
results[0].dense 是 1024 维数组
results[0].sparse.indices 非空
results[0].sparse.values 非空
indices 和 values 长度一致
```

## 十、后端配置

如果后端也在 Docker 的 `rag-net` 网络里，使用容器名访问：

```env
BGE_M3_URL=http://bge-m3-embedder:8000
```

如果后端不在 Docker 网络里，使用服务器 IP 访问：

```env
BGE_M3_URL=http://172.21.72.18:8001
```

## 十一、重启服务

停止并删除旧容器：

```bash
docker rm -f bge-m3-embedder
```

然后重新执行第六节的 `docker run` 命令。

## 十二、常见问题

### 1. docker: Error response from daemon: could not select device driver

说明 Docker 没有配置 GPU 支持。

处理方式：

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

如果第二条失败，需要安装或修复 `nvidia-container-toolkit`。

### 2. 模型下载很慢

命令里已经配置：

```bash
-e HF_ENDPOINT=https://hf-mirror.com
```

如果还是慢，可以手动提前下载模型到：

```text
/data/models/huggingface
```

### 3. 显存不够

可以降低 batch：

```bash
-e BGE_M3_BATCH_SIZE=4
-e BGE_M3_MAX_BATCH_SIZE=16
```

也可以先把最大长度降到：

```bash
-e BGE_M3_MAX_LENGTH=4096
```

### 4. 端口访问不了

先在服务器本机测试：

```bash
curl http://127.0.0.1:8001/health
```

如果本机可以，外部不行，通常是防火墙或安全组没开放 `8001`。

### 5. 每次启动都 pip install 很慢

当前命令为了简单，启动时会安装依赖。正式部署时可以后续制作自定义镜像，把依赖提前打进镜像里。

### 6. HuggingFace 镜像 403，卡在 imgs/.DS_Store

如果日志里出现：

```text
403 Forbidden
imgs/.DS_Store
```

说明镜像站下载模型仓库里的无关文件失败。处理方式是手动下载模型并忽略 `imgs` 文件夹：

```bash
sudo rm -rf /data/models/bge-m3
sudo mkdir -p /data/models/bge-m3
sudo chown -R yunxuan:yunxuan /data/models
```

```bash
sudo docker run --rm \
  -v /data/models:/data/models \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e HF_HUB_DISABLE_XET=1 \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
  bash -lc "pip install -U 'huggingface_hub<1.0' && python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-m3', local_dir='/data/models/bge-m3', local_dir_use_symlinks=False, ignore_patterns=['imgs/*','*.DS_Store','.DS_Store'])\""
```

下载完成后检查：

```bash
ls -lh /data/models/bge-m3
du -sh /data/models/bge-m3
```

至少应包含：

```text
config.json
model.safetensors 或 pytorch_model.bin
tokenizer.json
tokenizer_config.json
sentencepiece.bpe.model
```

### 7. torch.load 安全限制

如果日志里出现：

```text
Due to a serious vulnerability issue in torch.load
require users to upgrade torch to at least v2.6
```

说明当前容器里的 PyTorch 版本太低。启动命令里需要先升级：

```bash
pip install -U torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

上面的“本地模型路径”启动命令已经包含该处理。

## 十三、当前部署结果

部署成功后，你应该有：

```text
容器名：bge-m3-embedder
服务地址：http://172.21.72.18:8001
健康检查：http://172.21.72.18:8001/health
向量接口：http://172.21.72.18:8001/embed
```

该服务只负责生成向量，不写入 Qdrant，不调用大模型，也不做 rerank。
