"""
认证相关 Schema 定义
包含登录请求/响应、验证码、Token、用户信息等模型
"""

from typing import Optional

from pydantic import BaseModel, Field


class CaptchaResponse(BaseModel):
    """验证码响应"""
    captcha_id: str = Field(description="验证码ID，登录时需传回")
    captcha_image: str = Field(description="验证码图片 Base64 编码")


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., min_length=1, max_length=100, description="用户名")
    password: str = Field(..., min_length=1, max_length=128, description="密码")
    captcha_id: str = Field(..., description="验证码ID")
    captcha_code: str = Field(..., min_length=1, max_length=10, description="验证码")


class TokenResponse(BaseModel):
    """Token 响应"""
    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="bearer", description="Token 类型")
    expires_in: int = Field(description="Token 有效期（秒）")
    user_info: "UserInfoResponse" = Field(description="用户信息")


class UserInfoResponse(BaseModel):
    """用户信息响应"""
    id: int = Field(description="用户ID")
    username: str = Field(description="用户名")
    nickname: Optional[str] = Field(default=None, description="昵称")
    role: str = Field(description="角色")
    is_active: bool = Field(description="是否启用")


class ChangePasswordRequest(BaseModel):
    """修改密码请求"""
    old_password: str = Field(..., min_length=1, description="旧密码")
    new_password: str = Field(..., min_length=6, max_length=128, description="新密码")


class UserCreateRequest(BaseModel):
    """创建用户请求（预留，后续管理员使用）"""
    username: str = Field(..., min_length=1, max_length=100, description="用户名")
    password: str = Field(..., min_length=6, max_length=128, description="密码")
    nickname: Optional[str] = Field(default=None, description="昵称")
    role: str = Field(default="user", description="角色: admin/user/guest")
