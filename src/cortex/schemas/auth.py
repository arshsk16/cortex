"""Pydantic schemas for authentication flows."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from cortex.schemas.user import UserRead


class UserRegisterRequest(BaseModel):
    """Public registration payload."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    username: str = Field(..., min_length=3, max_length=64)
    full_name: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        from cortex.schemas.user import UserBase

        return UserBase.validate_username(value)

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str) -> str:
        from cortex.schemas.user import UserBase

        return UserBase.validate_full_name(value)

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        from cortex.schemas.user import UserCreate

        return UserCreate.validate_password_strength(value)


class UserLoginRequest(BaseModel):
    """Login payload using email + password."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class TokenResponse(BaseModel):
    """JWT access token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(
        ...,
        description="Access token lifetime in seconds",
    )


class AuthResponse(BaseModel):
    """Combined token + user payload returned after register/login."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserRead
