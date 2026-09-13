"""Pydantic schemas for user resources."""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from cortex.db.models.user import UserRole

_USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,64}$")


class UserBase(BaseModel):
    """Shared user fields used by create and response schemas."""

    email: EmailStr = Field(..., description="Unique email address")
    username: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Unique username",
    )
    full_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Display name",
    )

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        """Restrict usernames to alphanumeric characters and underscores."""
        if not _USERNAME_PATTERN.fullmatch(value):
            msg = (
                "Username must be 3-64 characters and contain only letters, "
                "numbers, and underscores"
            )
            raise ValueError(msg)
        return value

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str) -> str:
        """Reject blank full names after stripping whitespace."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("full_name must not be empty")
        return normalized


class UserCreate(UserBase):
    """Payload for creating a user (internal / service layer)."""

    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Plaintext password",
    )
    role: UserRole = Field(default=UserRole.USER)
    is_active: bool = Field(default=True)
    is_verified: bool = Field(default=False)

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, value: str) -> str:
        """Enforce a minimal password policy suitable for Phase 1."""
        if value.isspace() or not value.strip():
            raise ValueError("Password must not be blank")
        if not any(char.isalpha() for char in value) or not any(
            char.isdigit() for char in value
        ):
            raise ValueError("Password must contain at least one letter and one digit")
        return value


class UserUpdate(BaseModel):
    """Partial update payload for user profile fields."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr | None = None
    username: str | None = Field(default=None, min_length=3, max_length=64)
    full_name: str | None = Field(default=None, min_length=1, max_length=255)
    password: str | None = Field(default=None, min_length=8, max_length=128)
    role: UserRole | None = None
    is_active: bool | None = None
    is_verified: bool | None = None

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not _USERNAME_PATTERN.fullmatch(value):
            msg = (
                "Username must be 3-64 characters and contain only letters, "
                "numbers, and underscores"
            )
            raise ValueError(msg)
        return value

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("full_name must not be empty")
        return normalized

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if value.isspace() or not value.strip():
            raise ValueError("Password must not be blank")
        if not any(char.isalpha() for char in value) or not any(
            char.isdigit() for char in value
        ):
            raise ValueError("Password must contain at least one letter and one digit")
        return value


class UserRead(UserBase):
    """Public user representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
