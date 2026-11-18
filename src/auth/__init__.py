"""Auth module — license-token middleware + dependencies."""

from .dependencies import get_current_user, require_role
from .middleware import AuthMiddleware
from .models import User

__all__ = [
    "AuthMiddleware",
    "User",
    "get_current_user",
    "require_role",
]
