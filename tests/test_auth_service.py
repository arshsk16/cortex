"""Unit tests for authentication and user services."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from cortex.core.config import Settings
from cortex.core.exceptions import ConflictError, ForbiddenError, UnauthorizedError
from cortex.core.security import hash_password, verify_password
from cortex.db.models.user import User, UserRole
from cortex.schemas.auth import UserLoginRequest, UserRegisterRequest
from cortex.schemas.user import UserCreate, UserUpdate
from cortex.services.auth import AuthService
from cortex.services.user import UserService


@pytest.mark.asyncio
async def test_user_service_create_hashes_password() -> None:
    """UserService.create stores a bcrypt hash, not the plaintext password."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )

    service = UserService(session)
    created = await service.create(
        UserCreate(
            email="bob@example.com",
            username="bob",
            full_name="Bob Example",
            password="Password1",
        )
    )

    session.add.assert_called_once()
    user_arg: User = session.add.call_args.args[0]
    assert user_arg.email == "bob@example.com"
    assert user_arg.username == "bob"
    assert user_arg.hashed_password != "Password1"
    assert verify_password("Password1", user_arg.hashed_password)
    assert created is user_arg


@pytest.mark.asyncio
async def test_user_service_create_conflict_on_existing_email(
    sample_user: User,
) -> None:
    """Creating a user with an existing email raises ConflictError."""
    session = AsyncMock()
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=sample_user))
    )
    service = UserService(session)

    with pytest.raises(ConflictError) as exc_info:
        await service.create(
            UserCreate(
                email=sample_user.email,
                username="different",
                full_name="Other User",
                password="Password1",
            )
        )
    assert exc_info.value.code == "conflict"


@pytest.mark.asyncio
async def test_user_service_update_password(sample_user: User) -> None:
    """Updating password replaces the stored hash."""
    session = AsyncMock()
    session.get = AsyncMock(return_value=sample_user)
    session.execute = AsyncMock(
        return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
    )
    session.flush = AsyncMock()
    session.refresh = AsyncMock()

    service = UserService(session)
    await service.update(sample_user.id, UserUpdate(password="NewPass99"))

    assert verify_password("NewPass99", sample_user.hashed_password)
    assert not verify_password("Password1", sample_user.hashed_password)


@pytest.mark.asyncio
async def test_auth_service_register_returns_token(
    test_settings: Settings,
    sample_user: User,
) -> None:
    """AuthService.register returns a JWT and public user payload."""
    user_service = AsyncMock(spec=UserService)
    user_service.create = AsyncMock(return_value=sample_user)
    auth = AuthService(
        session=AsyncMock(),
        settings=test_settings,
        user_service=user_service,
    )

    result = await auth.register(
        UserRegisterRequest(
            email="alice@example.com",
            username="alice",
            full_name="Alice Example",
            password="Password1",
        )
    )

    assert result.access_token
    assert result.token_type == "bearer"
    assert result.user.email == sample_user.email
    assert result.expires_in == test_settings.jwt_access_token_expire_minutes * 60
    user_service.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_auth_service_login_success(
    test_settings: Settings,
    sample_user: User,
) -> None:
    """Valid credentials return an AuthResponse."""
    user_service = AsyncMock(spec=UserService)
    user_service.get_by_email = AsyncMock(return_value=sample_user)
    auth = AuthService(
        session=AsyncMock(),
        settings=test_settings,
        user_service=user_service,
    )

    result = await auth.login(
        UserLoginRequest(email=sample_user.email, password="Password1")
    )
    assert result.user.id == sample_user.id
    assert result.access_token


@pytest.mark.asyncio
async def test_auth_service_login_invalid_password(
    test_settings: Settings,
    sample_user: User,
) -> None:
    """Wrong password raises UnauthorizedError."""
    user_service = AsyncMock(spec=UserService)
    user_service.get_by_email = AsyncMock(return_value=sample_user)
    auth = AuthService(
        session=AsyncMock(),
        settings=test_settings,
        user_service=user_service,
    )

    with pytest.raises(UnauthorizedError):
        await auth.login(
            UserLoginRequest(email=sample_user.email, password="WrongPass1")
        )


@pytest.mark.asyncio
async def test_auth_service_login_inactive_user(
    test_settings: Settings,
    sample_user: User,
) -> None:
    """Inactive users cannot log in."""
    sample_user.is_active = False
    sample_user.hashed_password = hash_password("Password1")
    user_service = AsyncMock(spec=UserService)
    user_service.get_by_email = AsyncMock(return_value=sample_user)
    auth = AuthService(
        session=AsyncMock(),
        settings=test_settings,
        user_service=user_service,
    )

    with pytest.raises(ForbiddenError):
        await auth.login(
            UserLoginRequest(email=sample_user.email, password="Password1")
        )


@pytest.mark.asyncio
async def test_auth_service_login_unknown_email(test_settings: Settings) -> None:
    """Unknown email raises UnauthorizedError (no user enumeration detail)."""
    user_service = AsyncMock(spec=UserService)
    user_service.get_by_email = AsyncMock(return_value=None)
    auth = AuthService(
        session=AsyncMock(),
        settings=test_settings,
        user_service=user_service,
    )

    with pytest.raises(UnauthorizedError):
        await auth.login(
            UserLoginRequest(email="missing@example.com", password="Password1")
        )


def test_user_role_values() -> None:
    """UserRole enum exposes the expected persisted values."""
    assert UserRole.USER.value == "user"
    assert UserRole.ADMIN.value == "admin"
