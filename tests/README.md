# CitrusAgent 后端目录规范

本项目是农业智能问答系统后端，后续会接入：

- FastAPI 后端接口
- DeepSeek 大模型 API
- LangChain RAG 链路
- Qdrant 向量数据库
- 管理后台知识库接口
- 数据大屏接口

## 当前开发环境

按当前工作区环境统一：

```text
系统：Windows
终端：PowerShell
Python 包导入前缀：src.citrus_agent
推荐环境：conda 环境
推荐环境名：citrus-agent
```

如果 `conda` 环境，统一按下面方式创建：

```powershell
cd G:\py_workplace\CitrusAgent
conda create -n citrus-agent python=3.10 -y
conda activate citrus-agent
copy .env.example .env
```

```powershell
python -m pip install "fastapi==0.136.3" "uvicorn[standard]==0.49.0" "pydantic==2.13.4" "pydantic-settings==2.14.1" "python-dotenv==1.2.2" "langchain==1.3.1" "langchain-community==0.4.2" "langchain-openai==1.2.2" "qdrant-client==1.18.0" "SQLAlchemy==2.0.50" "pytest==9.0.3" "httpx==0.28.1"
```
## 统一导入方式

所有后端代码统一使用：

```python
from src.citrus_agent.xxx import xxx
```


## 目录结构

```text
CitrusAgent/
  README.md
  pyproject.toml
  environment.yml
  .env.example

  src/
    __init__.py
    citrus_agent/
      __init__.py

      api/
        __init__.py
        v1/
          __init__.py

      core/
        __init__.py

      services/
        __init__.py

      rag/
        __init__.py

      llm/
        __init__.py

      vectorstores/
        __init__.py

      db/
        __init__.py

      pojo/
        __init__.py

      common/
        __init__.py
```

## 每个目录放什么

`api/`

放 HTTP 接口。只负责接收请求、校验参数、调用 `services`、返回响应。不要在这里直接写复杂业务逻辑。

`api/v1/`

放第一版接口。比如后续可以新增：

- `chat.py`：聊天问答接口
- `knowledge.py`：知识库上传、删除、列表接口
- `dashboard.py`：数据大屏接口
- `admin.py`：管理后台接口

`core/`

放项目核心配置。比如：

- `config.py`：读取 `.env`
- `logging.py`：日志配置
- `security.py`：鉴权、敏感词、安全策略

`services/`

放业务编排代码。API 层应该优先调用这里。比如：

- `chat_service.py`：处理一次用户提问
- `knowledge_service.py`：处理知识库文件上传、解析、入库
- `dashboard_service.py`：处理大屏指标聚合

`rag/`

放 RAG 入库和检索相关代码。当前版本只做向量检索，不做重排。比如：

- `rag_api.py`：后端调用门面
- `retriever.py`：检索器封装

`llm/`

放大模型 API 封装。比如：

- `deepseek.py`：DeepSeek API 调用
- `base.py`：通用模型接口

API Key 必须从 `.env` 读取，不允许写死在代码里。

`vectorstores/`

放 Qdrant 和向量检索相关代码。比如：

- `qdrant.py`：Qdrant 连接、collection 管理、检索方法
- `embeddings.py`：Embedding 模型封装

`db/`

放关系型数据库相关代码。比如：

- `session.py`：数据库连接
- `models.py`：用户表、对话日志表、知识库元数据表
- `migrations/`：数据库迁移

`pojo/`

放请求对象、响应对象、业务传输对象。比如：

- `chat.py`：聊天请求、聊天响应、引用来源
- `knowledge.py`：知识库文件、解析状态、入库结果
- `user.py`：用户、角色、权限

字段名一旦和前端对接，就不要随便改。

`common/`

放通用工具。比如统一异常、响应码、时间处理、文件处理。不要把业务逻辑堆到这里。

## 推荐调用关系

统一按这个方向调用：

```text
api -> services -> rag -> llm / vectorstores
api -> services -> db
```

不要这样写：

```text
api -> llm
api -> vectorstores
api -> db 复杂查询
```

原因很简单：API 层如果直接到处调用，后面联调、测试、换模型、换数据库都会很难改。

## 注释要求

所有人写代码时必须写清楚必要注释，但不要每一行都废话式注释。

必须写注释的地方：

- 每个新增 `.py` 文件顶部写模块说明。
- 每个 service 方法写清楚“输入是什么、输出是什么、主要做什么”。
- RAG 链路、Prompt、检索策略、重排策略必须写说明。
- 数据库模型字段必须写清楚含义。
- 配置项必须说明用途，例如 `DEEPSEEK_API_KEY`、`QDRANT_COLLECTION`。
- 临时方案必须写 `TODO`，并标明负责人或原因。

推荐格式：

```python
"""聊天业务服务。

负责接收用户问题，调用 RAG 链路，并返回统一聊天结果。
"""


def answer_question(question: str) -> str:
    """根据用户问题返回知识库问答结果。

    Args:
        question: 用户输入的问题。

    Returns:
        大模型基于知识库生成的回答。
    """
```


## 新增代码流程

1. 先确认代码应该放在哪个目录。
2. 新建 `.py` 文件，并在文件顶部写模块说明。
3. 写函数或类时先确定输入和输出。
4. 如果方法会被别人调用，必须写 docstring。
5. 不确定的实现先写 `TODO`，不要随便写死。
6. 完成后在 README 或接口文档中补充调用说明。

## 环境变量

复制配置模板：

```powershell
copy .env.example .env
```

`.env` 文件只放本地配置，不要提交真实密钥。

重要配置：

```text
APP_ENV=local
PROJECT_ROOT=G:/py_workplace/CitrusAgent
PYTHONPATH=G:/py_workplace/CitrusAgent
DEEPSEEK_API_KEY=你的 DeepSeek Key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=agriculture_knowledge
```

