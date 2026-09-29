"""Authentication HTTP endpoints: register, login, logout, me."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request, status

from cortex.api.deps import (
    AuthServiceDep,
    CurrentActiveUserDep,
    SettingsDep,
    TokenBlocklistDep,
)
from cortex.core.exceptions import UnauthorizedError
from cortex.core.security import decode_access_token
from cortex.schemas.auth import AuthResponse, UserLoginRequest, UserRegisterRequest
from cortex.schemas.user import UserRead

router = APIRouter(tags=["Authentication"])


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description=(
        "Create a new user account and return a JWT access token together with "
        "the public user profile."
    ),
    responses={
        201: {"description": "User registered successfully"},
        409: {"description": "Email or username already exists"},
        422: {"description": "Validation error"},
    },
)
async def register(
    payload: UserRegisterRequest,
    auth_service: AuthServiceDep,
) -> AuthResponse:
    """Register a new user and issue an access token."""
    return await auth_service.register(payload)


@router.post(
    "/login",
    response_model=AuthResponse,
    status_code=status.HTTP_200_OK,
    summary="Log in",
    description="Authenticate with email and password; returns a JWT access token.",
    responses={
        200: {"description": "Login successful"},
        401: {"description": "Invalid credentials"},
        403: {"description": "Account inactive"},
        422: {"description": "Validation error"},
    },
)
async def login(
    payload: UserLoginRequest,
    auth_service: AuthServiceDep,
) -> AuthResponse:
    """Authenticate a user and issue an access token."""
    return await auth_service.login(payload)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Log out",
    description=(
        "Revoke the current access token by adding its ``jti`` to the Redis "
        "blocklist. Subsequent requests with this token will be rejected with "
        "401 even before the token expires naturally. "
        "If Redis is unavailable the call succeeds silently (fail-open)."
    ),
    responses={
        204: {"description": "Successfully logged out"},
        401: {"description": "Missing or invalid bearer token"},
    },
)
async def logout(
    request: Request,
    current_user: CurrentActiveUserDep,
    settings: SettingsDep,
    blocklist: TokenBlocklistDep,
) -> None:
    """Revoke the current bearer token."""
    # Extract raw token from the Authorization header.
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise UnauthorizedError("Not authenticated")
    raw_token = auth_header[7:].strip()

    # Decode to read jti + exp (signature already verified by CurrentActiveUserDep).
    payload = decode_access_token(raw_token, settings)
    jti: str | None = payload.get("jti")
    if not jti:
        # Token predates jti support -- nothing to revoke.
        return

    # Calculate remaining lifetime so the blocklist entry auto-expires with the token.
    exp_ts: int | float | None = payload.get("exp")
    if exp_ts is not None:
        remaining = int(exp_ts) - int(datetime.now(UTC).timestamp())
    else:
        remaining = settings.jwt_access_token_expire_minutes * 60

    if blocklist is not None:
        await blocklist.revoke(jti, ttl_seconds=remaining)


@router.get(
    "/me",
    response_model=UserRead,
    status_code=status.HTTP_200_OK,
    summary="Get current user",
    description="Return the profile of the currently authenticated user.",
    responses={
        200: {"description": "Current user profile"},
        401: {"description": "Missing or invalid bearer token"},
        403: {"description": "Account inactive"},
    },
)
async def read_current_user(
    current_user: CurrentActiveUserDep,
) -> UserRead:
    """Return the authenticated, active user's public profile."""
    return UserRead.model_validate(current_user)
