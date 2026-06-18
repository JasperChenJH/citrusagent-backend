"""
对话相关 Schema 定义
包含对话请求、响应、历史记录等模型
"""

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """对话请求"""
    question: str = Field(..., min_length=1, max_length=2000, description="用户提问内容")
    conversation_id: Optional[str] = Field(default=None, description="会话ID，为空则创建新会话")
    stream: bool = Field(default=True, description="是否流式输出")


class ChatResponse(BaseModel):
    """对话响应"""
    answer: str = Field(description="模型回答")
    conversation_id: str = Field(description="会话ID")
    sources: list[dict] = Field(default_factory=list, description="参考资料来源列表")
    tool_used: Optional[str] = Field(default=None, description="使用的工具名称（如有）")


class ChatStreamChunk(BaseModel):
    """流式对话响应片段"""
    type: str = Field(description="片段类型: content / source / tool / done")
    content: Optional[str] = Field(default=None, description="文本内容")
    sources: Optional[list[dict]] = Field(default=None, description="参考资料")
    tool_name: Optional[str] = Field(default=None, description="工具名称")
    conversation_id: Optional[str] = Field(default=None, description="会话ID")


class ChatHistoryItem(BaseModel):
    """对话历史条目"""
    role: str = Field(description="角色: user / assistant")
    content: str = Field(description="消息内容")
    sources: Optional[list[dict]] = Field(default=None, description="参考资料（仅assistant）")
    created_at: Optional[str] = Field(default=None, description="创建时间")


class ChatHistoryResponse(BaseModel):
    """对话历史响应"""
    conversation_id: str = Field(description="会话ID")
    messages: list[ChatHistoryItem] = Field(default_factory=list, description="消息列表")


class QuickCard(BaseModel):
    """快捷推荐卡片"""
    id: str = Field(description="卡片ID")
    title: str = Field(description="卡片标题")
    description: str = Field(description="卡片描述")
    icon: Optional[str] = Field(default=None, description="图标标识")
    preset_question: str = Field(description="预设问题")
