"""
工具调用 API 路由
提供工具列表查询和工具执行接口
"""

from fastapi import APIRouter, Depends, HTTPException

from src.citrus_agent.api.deps import get_tool_service
from src.citrus_agent.common.common import ResponseBase
from src.citrus_agent.services.tool_service import ToolService

router = APIRouter(prefix="/tools", tags=["工具调用"])


@router.get("/list", response_model=ResponseBase, summary="获取可用工具列表")
async def list_tools(
    tool_service: ToolService = Depends(get_tool_service),
) -> ResponseBase:
    """
    获取所有已注册的工具列表

    返回工具名称、描述和参数 Schema
    """
    tools = tool_service.list_tools()
    return ResponseBase(data=tools)


@router.post("/execute", response_model=ResponseBase, summary="执行工具")
async def execute_tool(
    tool_name: str,
    parameters: dict,
    tool_service: ToolService = Depends(get_tool_service),
) -> ResponseBase:
    """
    执行指定工具

    Args:
        tool_name: 工具名称
        parameters: 工具参数

    Returns:
        工具执行结果
    """
    try:
        result = await tool_service.execute_tool(tool_name, parameters)
        return ResponseBase(data=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"工具执行失败: {str(e)}")
