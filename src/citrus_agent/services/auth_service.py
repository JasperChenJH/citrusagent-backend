"""
认证服务模块
负责登录验证、JWT 令牌管理、验证码生成与校验、权限预留
"""

import base64
import io
import random
import string
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.citrus_agent.core.settings import get_settings
from src.citrus_agent.db.db_models import User
from src.citrus_agent.db.auth import (
    CaptchaResponse,
    TokenResponse,
    UserInfoResponse,
)
from src.citrus_agent.services.logger import logger

# 密码哈希上下文
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 内存验证码存储 {captcha_id: {"code": "xxxx", "expire": timestamp}}
# 后续可替换为 Redis
_captcha_store: dict[str, dict] = {}


class AuthService:
    """认证服务"""
    # ==================== 验证码 ====================
    def generate_captcha(self) -> CaptchaResponse:
        """
        生成图形验证码
        Returns:
            CaptchaResponse: 包含 captcha_id 和 Base64 图片
        """
        settings = get_settings()
        captcha_id = _generate_random_id()
        code = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=settings.CAPTCHA_LENGTH)
        )
        # 存储验证码
        expire = time.time() + settings.CAPTCHA_EXPIRE_SECONDS
        _captcha_store[captcha_id] = {"code": code, "expire": expire}
        # 生成简单图片
        image_base64 = _generate_captcha_image(code)
        return CaptchaResponse(captcha_id=captcha_id, captcha_image=image_base64)

    def verify_captcha(self, captcha_id: str, captcha_code: str) -> bool:
        """
        校验验证码

        Args:
            captcha_id: 验证码ID
            captcha_code: 用户输入的验证码

        Returns:
            是否验证通过
        """
        record = _captcha_store.pop(captcha_id, None)
        if not record:
            return False

        # 检查过期
        if time.time() > record["expire"]:
            return False

        # 不区分大小写
        return record["code"].upper() == captcha_code.upper()

    # ==================== 密码 ====================

    @staticmethod
    def hash_password(password: str) -> str:
        """密码哈希"""
        return _pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return _pwd_context.verify(plain_password, hashed_password)

    # ==================== JWT ====================

    @staticmethod
    def create_access_token(
        data: dict,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        创建 JWT access token

        Args:
            data: 要编码的数据（通常包含 sub, role 等）
            expires_delta: 自定义过期时间

        Returns:
            JWT token 字符串
        """
        settings = get_settings()
        to_encode = data.copy()

        expire = datetime.now(timezone.utc) + (
            expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    @staticmethod
    def decode_access_token(token: str) -> Optional[dict]:
        """
        解码并验证 JWT token

        Args:
            token: JWT token 字符串

        Returns:
            解码后的 payload，失败返回 None
        """
        settings = get_settings()
        try:
            payload = jwt.decode(
                token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
            )
            return payload
        except JWTError:
            return None

    # ==================== 登录 ====================

    async def authenticate_user(
        self, db: AsyncSession, username: str, password: str
    ) -> Optional[User]:
        """
        验证用户名密码（查询 MySQL）

        Args:
            db: 数据库会话
            username: 用户名
            password: 明文密码

        Returns:
            验证成功返回 User 对象，失败返回 None
        """
        result = await db.execute(select(User).where(User.username == username))
        user = result.scalar_one_or_none()

        if not user:
            return None
        if not user.is_active:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None

        # 更新最后登录时间
        user.last_login_at = datetime.now(timezone.utc)
        await db.flush()

        return user

    async def login(
        self,
        db: AsyncSession,
        username: str,
        password: str,
        captcha_id: str,
        captcha_code: str,
    ) -> TokenResponse | dict:
        """
        完整登录流程：验证码校验 → 用户名密码校验 → 生成 Token

        Args:
            db: 数据库会话
            username: 用户名
            password: 密码
            captcha_id: 验证码ID
            captcha_code: 验证码

        Returns:
            TokenResponse 或 {"error": "错误信息"}
        """
        settings = get_settings()

        # 1. 校验验证码
        if not self.verify_captcha(captcha_id, captcha_code):
            return {"error": "验证码错误或已过期"}

        # 2. 校验用户名密码
        user = await self.authenticate_user(db, username, password)
        if not user:
            return {"error": "用户名或密码错误"}

        # 3. 生成 JWT
        token_data = {
            "sub": str(user.id),
            "username": user.username,
            "role": user.role,
        }
        access_token = self.create_access_token(token_data)

        # 4. 构建响应
        user_info = UserInfoResponse(
            id=user.id,
            username=user.username,
            nickname=user.nickname,
            role=user.role,
            is_active=user.is_active,
        )

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_info=user_info,
        )

    # ==================== 用户信息 ====================

    async def get_current_user(self, db: AsyncSession, token: str) -> Optional[User]:
        """
        根据 token 获取当前用户

        Args:
            db: 数据库会话
            token: JWT token

        Returns:
            User 对象，失败返回 None
        """
        payload = self.decode_access_token(token)
        if not payload:
            return None

        user_id = payload.get("sub")
        if not user_id:
            return None

        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            return None

        result = await db.execute(select(User).where(User.id == user_id_int))
        return result.scalar_one_or_none()

    # ==================== 权限预留 ====================

    @staticmethod
    def check_permission(user: User, required_role: str) -> bool:
        """
        检查用户是否具有指定角色权限（预留）

        当前为简单角色匹配，后续可扩展为 RBAC/ABAC 权限体系

        Args:
            user: 用户对象
            required_role: 需要的角色

        Returns:
            是否有权限
        """
        role_hierarchy = {"admin": 3, "user": 2, "guest": 1}
        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        return user_level >= required_level

    @staticmethod
    def check_resource_permission(user: User, resource: str, action: str) -> bool:
        """
        检查用户对指定资源的操作权限（预留）

        后续可对接权限表，实现细粒度的资源级权限控制

        Args:
            user: 用户对象
            resource: 资源标识（如 "knowledge_base", "document", "admin_panel"）
            action: 操作类型（如 "read", "write", "delete"）

        Returns:
            是否有权限
        """
        # 当前简单实现：admin 拥有所有权限
        if user.role == "admin":
            return True

        # 普通用户只读权限
        if user.role == "user" and action in ("read", "write"):
            return True

        # 访客只读
        if user.role == "guest" and action == "read":
            return True

        return False

    # ==================== 用户管理预留 ====================

    async def create_user(
        self,
        db: AsyncSession,
        username: str,
        password: str,
        nickname: str | None = None,
        role: str = "user",
    ) -> User:
        """
        创建用户（预留，后续管理员使用）

        Args:
            db: 数据库会话
            username: 用户名
            password: 明文密码
            nickname: 昵称
            role: 角色

        Returns:
            创建的 User 对象
        """
        hashed = self.hash_password(password)
        user = User(
            username=username,
            hashed_password=hashed,
            nickname=nickname,
            role=role,
        )
        db.add(user)
        await db.flush()
        logger.info(f"用户已创建: {username} (role={role})")
        return user

    async def change_password(
        self, db: AsyncSession, user: User, old_password: str, new_password: str
    ) -> bool:
        """
        修改密码

        Args:
            db: 数据库会话
            user: 当前用户
            old_password: 旧密码
            new_password: 新密码

        Returns:
            是否修改成功
        """
        if not self.verify_password(old_password, user.hashed_password):
            return False

        user.hashed_password = self.hash_password(new_password)
        await db.flush()
        logger.info(f"用户 {user.username} 修改密码成功")
        return True


# ==================== 工具函数 ====================


def _generate_random_id() -> str:
    """生成随机 ID"""
    import uuid
    return uuid.uuid4().hex


def _generate_captcha_image(code: str) -> str:
    """
    生成验证码图片并返回 Base64 编码

    使用 Pillow 生成简单验证码图片，包含干扰线和噪点

    Args:
        code: 验证码文本

    Returns:
        Base64 编码的 PNG 图片
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        width, height = 120, 40
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        # 使用默认字体
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except (IOError, OSError):
            font = ImageFont.load_default()

        # 绘制验证码文字
        for i, char in enumerate(code):
            x = 10 + i * 25
            y = random.randint(2, 8)
            color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            draw.text((x, y), char, fill=color, font=font)

        # 添加干扰线
        for _ in range(3):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = random.randint(0, width), random.randint(0, height)
            draw.line((x1, y1, x2, y2), fill=(200, 200, 200), width=1)

        # 添加噪点
        for _ in range(50):
            x, y = random.randint(0, width - 1), random.randint(0, height - 1)
            draw.point((x, y), fill=(random.randint(0, 200), random.randint(0, 200), random.randint(0, 200)))

        # 转 Base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except ImportError:
        # Pillow 不可用时，返回纯文本
        logger.warning("Pillow 未安装，验证码将以纯文本形式返回")
        return base64.b64encode(code.encode("utf-8")).decode("utf-8")


# 全局单例
_auth_service: AuthService | None = None


def get_auth_service() -> AuthService:
    """获取认证服务单例"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
