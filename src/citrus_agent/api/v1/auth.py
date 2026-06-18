"""
认证相关 API 路由
提供登录、验证码、用户信息、修改密码等接口
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.citrus_agent.api.deps import get_current_user
from src.citrus_agent.core.database import get_db
from src.citrus_agent.db.db_models import User
from src.citrus_agent.db.auth import (
    LoginRequest,
    UserInfoResponse,
    ChangePasswordRequest,
    UserCreateRequest,
)
from src.citrus_agent.common.common import ResponseBase
from src.citrus_agent.services.auth_service import get_auth_service, AuthService

router = APIRouter(prefix="/auth", tags=["认证"])


@router.get("/captcha", response_model=ResponseBase, summary="获取验证码")
async def get_captcha(
    auth_service: AuthService = Depends(get_auth_service),
) -> ResponseBase:
    """
    获取图形验证码

    返回 captcha_id 和 Base64 编码的验证码图片，登录时需传回
    """
    captcha = auth_service.generate_captcha()
    return ResponseBase(data=captcha.model_dump())


@router.post("/login", response_model=ResponseBase, summary="用户登录")
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
) -> ResponseBase:
    """
    用户登录

    需先调用 /auth/captcha 获取验证码，登录时传入验证码
    验证通过后返回 JWT token 和用户信息
    """
    result = await auth_service.login(
        db=db,
        username=request.username,
        password=request.password,
        captcha_id=request.captcha_id,
        captcha_code=request.captcha_code,
    )

    if isinstance(result, dict) and "error" in result:
        return ResponseBase(code=401, message=result["error"])

    return ResponseBase(data=result.model_dump())


@router.get("/me", response_model=ResponseBase, summary="获取当前用户信息")
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
) -> ResponseBase:
    """
    获取当前登录用户信息

    需在请求头中携带 Authorization: Bearer <token>
    """
    user_info = UserInfoResponse(
        id=current_user.id,
        username=current_user.username,
        nickname=current_user.nickname,
        role=current_user.role,
        is_active=current_user.is_active,
    )
    return ResponseBase(data=user_info.model_dump())


@router.post("/change-password", response_model=ResponseBase, summary="修改密码")
async def change_password(
    request: ChangePasswordRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
) -> ResponseBase:
    """
    修改当前用户密码

    需验证旧密码后才能设置新密码
    """
    success = await auth_service.change_password(
        db=db,
        user=current_user,
        old_password=request.old_password,
        new_password=request.new_password,
    )
    if not success:
        return ResponseBase(code=400, message="旧密码错误")
    return ResponseBase(message="密码修改成功")


@router.post("/users", response_model=ResponseBase, summary="创建用户（预留）")
async def create_user(
    request: UserCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    auth_service: AuthService = Depends(get_auth_service),
) -> ResponseBase:
    """
    创建新用户（预留接口，后续由管理员使用）

    当前仅 admin 角色可调用
    """
    # 权限检查：仅 admin 可创建用户
    if not auth_service.check_permission(current_user, "admin"):
        return ResponseBase(code=403, message="无权限执行此操作")

    # 检查用户名是否已存在
    from sqlalchemy import select
    from src.citrus_agent.db.db_models import User
    existing = await db.execute(select(User).where(User.username == request.username))
    if existing.scalar_one_or_none():
        return ResponseBase(code=400, message="用户名已存在")

    user = await auth_service.create_user(
        db=db,
        username=request.username,
        password=request.password,
        nickname=request.nickname,
        role=request.role,
    )

    return ResponseBase(data={
        "id": user.id,
        "username": user.username,
        "role": user.role,
    })
