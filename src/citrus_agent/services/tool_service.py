"""
工具调用服务
负责工具调用的决策与执行
"""

from typing import Any, Optional

from src.citrus_agent.tools.base import ToolRegistry
from src.citrus_agent.services.logger import logger


class ToolService:
    """工具调用服务"""

    @staticmethod
    def list_tools() -> list[dict]:
        """
        获取所有可用工具列表

        Returns:
            工具信息列表
        """
        tools = ToolRegistry.get_all_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            }
            for tool in tools.values()
        ]

    @staticmethod
    async def execute_tool(
        tool_name: str, parameters: dict
    ) -> dict[str, Any]:
        """
        执行指定工具

        Args:
            tool_name: 工具名称
            parameters: 工具参数

        Returns:
            执行结果

        Raises:
            ValueError: 工具不存在
        """
        tool = ToolRegistry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"工具不存在: {tool_name}")

        logger.info(f"执行工具: {tool_name}, 参数: {parameters}")
        result = await tool.execute(**parameters)
        logger.info(f"工具执行完成: {tool_name}")
        return result

    @staticmethod
    def get_langchain_tools() -> list[dict]:
        """
        获取 LangChain 格式的工具描述
        供 LLM function calling 使用

        Returns:
            LangChain 工具描述列表
        """
        return ToolRegistry.get_langchain_tools()
