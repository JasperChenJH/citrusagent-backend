"""大模型客户端封装目录。

职责：
    1. 统一封装 DeepSeek 等大模型 API。
    2. 处理模型参数、超时、重试、错误提示。
    3. 给 RAG 层提供稳定调用方法。

建议文件：
    - deepseek.py：DeepSeek API 封装。
    - base.py：通用模型接口定义。

注意：
    API Key 不要写死在代码里，必须从 `.env` 或环境变量读取。
"""
