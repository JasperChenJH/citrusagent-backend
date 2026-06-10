"""项目源码根包。

本项目统一使用 `src.citrus_agent` 作为 Python 导入前缀。

示例：
    from src.citrus_agent.services import ...

团队约定：
    1. 不要在项目根目录下散放业务代码。
    2. 新增 Python 模块时统一放到 `src/citrus_agent/` 对应子目录。
    3. 跨目录调用时优先通过上层封装调用，避免互相乱引。
"""
