"""
通用 Schema 定义
包含统一的响应格式、分页等公共模型
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ResponseBase(BaseModel):
    """统一响应格式"""
    code: int = Field(default=200, description="状态码")
    message: str = Field(default="success", description="状态信息")
    data: Any = Field(default=None, description="响应数据")


class PaginationRequest(BaseModel):
    """分页请求"""
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页数量")


class PaginationMeta(BaseModel):
    """分页元数据"""
    total: int = Field(description="总记录数")
    page: int = Field(description="当前页码")
    page_size: int = Field(description="每页数量")
    total_pages: int = Field(description="总页数")


class PaginatedResponse(BaseModel):
    """分页响应"""
    code: int = Field(default=200)
    message: str = Field(default="success")
    data: list[Any] = Field(default_factory=list)
    pagination: Optional[PaginationMeta] = None
