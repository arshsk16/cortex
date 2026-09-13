"""User persistence service — CRUD operations against the users table."""

from __future__ import annotations

import logging

from sqlalchemy import Select, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from cortex.core.exceptions import ConflictError, NotFoundError
from cortex.core.security import hash_password
from cortex.db.models.user import User
from cortex.schemas.user import UserCreate, UserUpdate

logger = logging.getLogger(__name__)


class UserService:
    """Application service for user CRUD.

    Receives an async SQLAlchemy session via constructor injection so call sites
    (routes, AuthService, tests) remain decoupled from persistence details.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, data: UserCreate) -> User:
        """Persist a new user after uniqueness checks."""
        await self._ensure_unique(email=data.email, username=data.username)

        user = User(
            email=str(data.email).lower(),
            username=data.username,
            full_name=data.full_name,
            hashed_password=hash_password(data.password),
            role=data.role,
            is_active=data.is_active,
            is_verified=data.is_verified,
        )
        self._session.add(user)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            logger.warning("Integrity error while creating user: %s", exc)
            raise ConflictError(
                "A user with this email or username already exists",
                details={"email": str(data.email), "username": data.username},
            ) from exc

        await self._session.refresh(user)
        return user

    async def get_by_id(self, user_id: str) -> User | None:
        """Return a user by primary key, or None if missing."""
        return await self._session.get(User, user_id)

    async def get_by_id_or_raise(self, user_id: str) -> User:
        """Return a user by id or raise NotFoundError."""
        user = await self.get_by_id(user_id)
        if user is None:
            raise NotFoundError(
                "User not found",
                details={"user_id": user_id},
            )
        return user

    async def get_by_email(self, email: str) -> User | None:
        """Return a user by email (case-insensitive), or None."""
        statement: Select[tuple[User]] = select(User).where(
            func.lower(User.email) == email.lower()
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def get_by_username(self, username: str) -> User | None:
        """Return a user by username (exact match), or None."""
        statement: Select[tuple[User]] = select(User).where(User.username == username)
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    async def list_users(self, *, skip: int = 0, limit: int = 50) -> list[User]:
        """Return a page of users ordered by creation time descending."""
        statement: Select[tuple[User]] = (
            select(User).order_by(User.created_at.desc()).offset(skip).limit(limit)
        )
        result = await self._session.execute(statement)
        return list(result.scalars().all())

    async def update(self, user_id: str, data: UserUpdate) -> User:
        """Apply a partial update to an existing user."""
        user = await self.get_by_id_or_raise(user_id)
        updates = data.model_dump(exclude_unset=True)

        if "email" in updates and updates["email"] is not None:
            updates["email"] = str(updates["email"]).lower()
            await self._ensure_unique(
                email=updates["email"],
                username=None,
                exclude_user_id=user_id,
            )

        if "username" in updates and updates["username"] is not None:
            await self._ensure_unique(
                email=None,
                username=updates["username"],
                exclude_user_id=user_id,
            )

        if "password" in updates and updates["password"] is not None:
            updates["hashed_password"] = hash_password(updates.pop("password"))

        for field, value in updates.items():
            setattr(user, field, value)

        try:
            await self._session.flush()
        except IntegrityError as exc:
            raise ConflictError(
                "A user with this email or username already exists",
            ) from exc

        await self._session.refresh(user)
        return user

    async def delete(self, user_id: str) -> None:
        """Hard-delete a user by id."""
        user = await self.get_by_id_or_raise(user_id)
        await self._session.delete(user)
        await self._session.flush()

    async def _ensure_unique(
        self,
        *,
        email: str | None,
        username: str | None,
        exclude_user_id: str | None = None,
    ) -> None:
        """Raise ConflictError when email or username is already taken."""
        clauses = []
        if email is not None:
            clauses.append(func.lower(User.email) == email.lower())
        if username is not None:
            clauses.append(User.username == username)
        if not clauses:
            return

        statement: Select[tuple[User]] = select(User).where(or_(*clauses))
        if exclude_user_id is not None:
            statement = statement.where(User.id != exclude_user_id)

        result = await self._session.execute(statement)
        existing = result.scalar_one_or_none()
        if existing is None:
            return

        details: dict[str, str] = {}
        if email is not None and existing.email.lower() == email.lower():
            details["email"] = email
        if username is not None and existing.username == username:
            details["username"] = username
        raise ConflictError(
            "A user with this email or username already exists",
            details=details,
        )
