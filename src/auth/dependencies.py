"""FastAPI dependencies for authentication."""

from __future__ import annotations

import os
from collections.abc import Callable, Coroutine
from typing import Any, cast

from fastapi import Depends, HTTPException, Request, status

from .models import User

ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "0195f48b-d000-7000-8000-000000000001")

_ANONYMOUS_USER = User(
    sub=DEFAULT_TENANT_ID,
    preferred_username="anonymous",
    tenant_id=DEFAULT_TENANT_ID,
    roles=["admin"],
)


async def get_current_user(request: Request) -> User:
    """Return the authenticated user, or raise 401."""
    if not ENABLE_AUTH:
        return _ANONYMOUS_USER

    user = getattr(request.state, "user", None)
    if isinstance(user, User):
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(role: str) -> Callable[..., Coroutine[Any, Any, User]]:
    async def checker(user: User = Depends(get_current_user)) -> User:
        if not user.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required",
            )
        return user

    return cast("Callable[..., Coroutine[Any, Any, User]]", checker)
