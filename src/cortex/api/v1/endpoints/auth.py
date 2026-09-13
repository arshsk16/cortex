"""Authentication HTTP endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from cortex.api.deps import AuthServiceDep, CurrentActiveUserDep
from cortex.schemas.auth import (
    AuthResponse,
    UserLoginRequest,
    UserRegisterRequest,
)
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
