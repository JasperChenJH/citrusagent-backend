"""业务服务层。

职责：
    1. 编排一个完整业务动作。
    2. 给 API 层提供稳定方法。
    3. 隔离 API、RAG、数据库、向量库之间的直接依赖。

调用规则：
    API 层优先调用这里，不要越过 services 直接调用 rag、llm、db。

示例：
    - chat_service.py：处理一次用户提问。
    - knowledge_service.py：处理知识库文件上传、解析、入库。
    - dashboard_service.py：处理数据大屏指标聚合。
"""
