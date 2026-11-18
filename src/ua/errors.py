"""JSON-RPC / MCP error codes (design.md 1.2)."""
from __future__ import annotations

from pydantic import BaseModel

CODES = {
    -32700: "PARSE_ERROR", -32600: "INVALID_REQUEST", -32601: "METHOD_NOT_FOUND",
    -32602: "INVALID_PARAMS", -32603: "INTERNAL_ERROR",
    -32001: "TOOL_TIMEOUT", -32002: "RBAC_DENIED", -32003: "IDEMPOTENCY_CONFLICT",
    -32004: "BREAKER_OPEN", -32006: "TENANT_MISMATCH", -32007: "PII_BLOCKED",
    -32008: "INJECTION_BLOCKED", -32009: "BUDGET_EXHAUSTED", -32010: "UPSTREAM_UNAVAILABLE",
    -32011: "NOT_FOUND",
}


class ErrorObj(BaseModel):
    code: int
    name: str
    message: str
    retryable: bool = False


class UAError(Exception):
    def __init__(self, code: int, message: str, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable

    @property
    def name(self) -> str:
        return CODES.get(self.code, "UNKNOWN")

    def to_obj(self) -> ErrorObj:
        return ErrorObj(code=self.code, name=self.name, message=self.message, retryable=self.retryable)


class Block(UAError):
    """Raised by a plugin to block the request (fail-closed)."""
