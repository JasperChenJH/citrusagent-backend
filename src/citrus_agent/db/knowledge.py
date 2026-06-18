"""
知识库相关 Schema 定义
包含文档上传、知识库条目、管理操作等模型
"""

from typing import Optional

from pydantic import BaseModel, Field


class KnowledgeDocumentCreate(BaseModel):
    """知识库文档创建请求"""
    title: str = Field(..., min_length=1, max_length=200, description="文档标题")
    source_type: str = Field(
        default="general",
        description="来源类型: general / literature / policy / structured / external",
    )
    description: Optional[str] = Field(default=None, description="文档描述")
    tags: list[str] = Field(default_factory=list, description="标签列表")


class KnowledgeDocumentResponse(BaseModel):
    """知识库文档响应"""
    id: str = Field(description="文档ID")
    kb_id: Optional[int] = Field(default=None, description="所属知识库ID")
    title: str = Field(description="文档标题")
    source_type: str = Field(description="来源类型")
    description: Optional[str] = Field(default=None, description="文档描述")
    tags: list[str] = Field(default_factory=list, description="标签列表")
    chunk_count: int = Field(default=0, description="切片数量")
    file_name: Optional[str] = Field(default=None, description="原始文件名")
    status: str = Field(default="pending", description="状态: pending/processing/ready/failed")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    created_at: str = Field(description="创建时间")
    updated_at: Optional[str] = Field(default=None, description="更新时间")


class KnowledgeChunkResponse(BaseModel):
    """知识库切片响应"""
    id: str = Field(description="切片ID")
    document_id: str = Field(description="所属文档ID")
    content: str = Field(description="切片内容")
    chunk_index: str = Field(description="切片索引")
    chunk_method: str = Field(description="切片方法")
    metadata: dict = Field(default_factory=dict, description="元数据")


class KnowledgeDocumentUpdate(BaseModel):
    """知识库文档更新请求"""
    title: Optional[str] = Field(default=None, description="文档标题")
    description: Optional[str] = Field(default=None, description="文档描述")
    tags: Optional[list[str]] = Field(default=None, description="标签列表")


class KnowledgeSearchRequest(BaseModel):
    """知识库检索请求"""
    query: str = Field(..., min_length=1, description="检索查询文本")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    kb_id: Optional[int] = Field(default=None, description="限定知识库ID，避免跨库串数据")
    source_type: Optional[str] = Field(default=None, description="按来源类型过滤")
    tags: Optional[list[str]] = Field(default=None, description="按标签过滤")


class KnowledgeSearchResult(BaseModel):
    """知识库检索结果"""
    chunk_id: str = Field(description="切片ID")
    content: str = Field(description="切片内容")
    score: float = Field(description="相似度分数")
    document_id: str = Field(description="所属文档ID")
    document_title: str = Field(description="所属文档标题")
    source_type: str = Field(default="", description="来源类型")
    metadata: dict = Field(default_factory=dict, description="元数据")
