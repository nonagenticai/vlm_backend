"""User model — hybrid shape supporting license-token + legacy claims."""

from __future__ import annotations

from pydantic import BaseModel, Field


class User(BaseModel):
    """User identity carried on ``request.state.user``."""

    sub: str = Field(..., description="Subject identifier")
    preferred_username: str = Field(default="anonymous", description="Username")
    email: str | None = None
    name: str | None = None

    realm_access: dict[str, list[str]] = Field(default_factory=dict)
    resource_access: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    roles: list[str] = Field(default_factory=list)

    tenant_id: str | None = Field(default=None, description="Tenant identifier")
    organization_id: str | None = Field(default=None, description="Organization ID")

    def model_post_init(self, __context: object) -> None:
        if self.tenant_id is None:
            self.tenant_id = self.organization_id or self.sub

    @property
    def effective_tenant_id(self) -> str:
        return self.tenant_id or self.organization_id or self.sub

    def has_role(self, role: str) -> bool:
        all_roles = self.roles + self.realm_access.get("roles", [])
        return role in all_roles

    def has_realm_role(self, role: str) -> bool:
        roles = self.realm_access.get("roles", [])
        return role in roles or role in self.roles

    def has_client_role(self, client_id: str, role: str) -> bool:
        client_roles = self.resource_access.get(client_id, {}).get("roles", [])
        return role in client_roles

    def get_all_roles(self) -> list[str]:
        realm_roles: list[str] = self.realm_access.get("roles", []) + self.roles
        client_roles: list[str] = []
        for client_data in self.resource_access.values():
            client_roles.extend(client_data.get("roles", []))
        return realm_roles + client_roles

    def is_admin(self) -> bool:
        return self.has_realm_role("admin") or self.has_realm_role("platform-admin")
