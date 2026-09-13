"""Unit tests for password hashing and JWT helpers."""

from __future__ import annotations

import pytest

from cortex.core.config import Settings
from cortex.core.exceptions import UnauthorizedError
from cortex.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_and_verify_password() -> None:
    """Hashed passwords verify correctly and reject wrong passwords."""
    hashed = hash_password("Password1")
    assert hashed != "Password1"
    assert verify_password("Password1", hashed) is True
    assert verify_password("wrong-password", hashed) is False


def test_create_and_decode_access_token(test_settings: Settings) -> None:
    """Access tokens round-trip subject and custom claims."""
    token = create_access_token(
        subject="user-123",
        settings=test_settings,
        extra_claims={"email": "alice@example.com"},
    )
    payload = decode_access_token(token, test_settings)
    assert payload["sub"] == "user-123"
    assert payload["email"] == "alice@example.com"
    assert payload["type"] == "access"


def test_decode_invalid_token_raises(test_settings: Settings) -> None:
    """Malformed tokens raise UnauthorizedError."""
    with pytest.raises(UnauthorizedError):
        decode_access_token("not-a-valid-jwt", test_settings)


def test_decode_token_with_wrong_secret_raises(test_settings: Settings) -> None:
    """Tokens signed with a different secret are rejected."""
    other = test_settings.model_copy(
        update={"jwt_secret_key": "another-secret-key-that-is-32chars!!"}
    )
    token = create_access_token(subject="user-123", settings=other)
    with pytest.raises(UnauthorizedError):
        decode_access_token(token, test_settings)
