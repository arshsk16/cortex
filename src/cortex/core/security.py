"""Password hashing and JWT helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from cortex.core.config import Settings
from cortex.core.exceptions import UnauthorizedError

# bcrypt has a 72-byte input limit; truncate defensively before hashing/verify.
_BCRYPT_MAX_PASSWORD_BYTES = 72
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _prepare_password(password: str) -> str:
    """Normalize password input for bcrypt's byte-length constraint."""
    return password.encode("utf-8")[:_BCRYPT_MAX_PASSWORD_BYTES].decode(
        "utf-8",
        errors="ignore",
    )


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt via passlib."""
    return _pwd_context.hash(_prepare_password(password))


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True when ``plain_password`` matches ``hashed_password``."""
    return _pwd_context.verify(_prepare_password(plain_password), hashed_password)


def create_access_token(
    *,
    subject: str,
    settings: Settings,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a signed JWT access token for the given subject (user id)."""
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "exp": expire,
        "type": "access",
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    """Decode and validate a JWT access token.

    Raises
    ------
    UnauthorizedError
        If the token is invalid, expired, or not an access token.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as exc:
        raise UnauthorizedError("Could not validate credentials") from exc

    if payload.get("type") != "access":
        raise UnauthorizedError("Invalid token type")

    subject = payload.get("sub")
    if not subject or not isinstance(subject, str):
        raise UnauthorizedError("Could not validate credentials")

    return payload
