"""
工具模块基类与注册器
所有自定义工具需继承 BaseTool 并注册到 ToolRegistry
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseTool(ABC):
    """工具基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（供 LLM 理解用途）"""
        ...

    @property
    def parameters_schema(self) -> dict:
        """工具参数的 JSON Schema 描述"""
        return {}

    @abstractmethod
    async def execute(self, **kwargs) -> dict[str, Any]:
        """
        执行工具逻辑

        Args:
            **kwargs: 工具参数

        Returns:
            执行结果字典
        """
        ...

    def to_langchain_tool(self) -> dict:
        """
        转换为 LangChain Tool 描述格式
        供 LLM function calling 使用
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


class ToolRegistry:
    """工具注册器"""

    _tools: dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        """注册工具"""
        cls._tools[tool.name] = tool

    @classmethod
    def get_tool(cls, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return cls._tools.get(name)

    @classmethod
    def get_all_tools(cls) -> dict[str, BaseTool]:
        """获取所有已注册工具"""
        return cls._tools.copy()

    @classmethod
    def get_langchain_tools(cls) -> list[dict]:
        """获取所有工具的 LangChain 描述"""
        return [tool.to_langchain_tool() for tool in cls._tools.values()]
