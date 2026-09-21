"""Authentication service — registration, login, and token issuance."""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from cortex.core.config import Settings
from cortex.core.exceptions import ForbiddenError, UnauthorizedError
from cortex.core.security import create_access_token, verify_password
from cortex.db.models.user import User, UserRole
from cortex.schemas.auth import AuthResponse, UserLoginRequest, UserRegisterRequest
from cortex.schemas.user import UserCreate, UserRead
from cortex.services.user import UserService

logger = logging.getLogger(__name__)


def _redact_email(email: str) -> str:
    """Return a partially redacted email address for safe logging.

    Example: ``alice@example.com`` → ``a***@example.com``.
    Preserves domain for routing/debugging without exposing the full address.
    """
    if "@" not in email:
        return "***"
    local, domain = email.rsplit("@", 1)
    visible = local[:1] if local else ""
    return f"{visible}***@{domain}"


class AuthService:
    """Coordinates authentication flows using UserService and JWT helpers."""

    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        user_service: UserService | None = None,
    ) -> None:
        self._settings = settings
        self._users = user_service or UserService(session)

    async def register(self, payload: UserRegisterRequest) -> AuthResponse:
        """Register a new user and return an access token plus user profile."""
        user = await self._users.create(
            UserCreate(
                email=payload.email,
                username=payload.username,
                full_name=payload.full_name,
                password=payload.password,
                role=UserRole.USER,
                is_active=True,
                is_verified=False,
            )
        )
        logger.info(
            "Registered user id=%s email=%s",
            user.id,
            _redact_email(str(user.email)),
        )
        return self._build_auth_response(user)

    async def login(self, payload: UserLoginRequest) -> AuthResponse:
        """Authenticate with email/password and return an access token."""
        user = await self._users.get_by_email(str(payload.email))
        if user is None or not verify_password(payload.password, user.hashed_password):
            raise UnauthorizedError("Incorrect email or password")

        if not user.is_active:
            raise ForbiddenError("User account is inactive")

        logger.info(
            "User logged in id=%s email=%s",
            user.id,
            _redact_email(str(user.email)),
        )
        return self._build_auth_response(user)

    def _build_auth_response(self, user: User) -> AuthResponse:
        """Issue a JWT and package it with the public user representation."""
        token = create_access_token(
            subject=user.id,
            settings=self._settings,
            extra_claims={"email": user.email, "role": user.role.value},
        )
        return AuthResponse(
            access_token=token,
            token_type="bearer",
            expires_in=self._settings.jwt_access_token_expire_minutes * 60,
            user=UserRead.model_validate(user),
        )
