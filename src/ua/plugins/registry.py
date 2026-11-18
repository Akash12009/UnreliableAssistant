"""Priority-ordered, hot-toggleable lifecycle plugin registry on top of pluggy."""
from __future__ import annotations

import importlib.metadata
import time
from dataclasses import dataclass, field
from typing import Any

import pluggy

from ..config import load_mcp_config
from ..errors import Block, UAError
from ..telemetry import span
from .spec import LifecycleSpec

STAGE_HOOK = {"PRE_EXECUTION": "pre_execution", "TOOL_DISPATCH": "tool_dispatch",
              "POST_EXECUTION": "post_execution"}
LOCKED = {"pii_redactor", "injection_shield"}   # cannot be hot-disabled outside dev mode


@dataclass
class RequestContext:
    tenant_id: str
    user_id: str
    session_id: str
    role: str
    raw_query: str
    query: str = ""                       # redacted + normalised working query
    traceparent: str = ""
    flags: dict[str, Any] = field(default_factory=dict)
    redactions: dict[str, int] = field(default_factory=dict)
    injection_score: float = 0.0
    blocked: Block | None = None
    t0: float = field(default_factory=time.perf_counter)

    def __post_init__(self):
        self.query = self.query or self.raw_query

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000


@dataclass
class PluginEntry:
    name: str
    stage: str
    priority: int
    fail_closed: bool
    enabled: bool
    obj: Any
    cfg: dict


class LifecycleRegistry:
    def __init__(self, config: dict | None = None, dev_mode: bool = True):
        self.pm = pluggy.PluginManager("ua")
        self.pm.add_hookspecs(LifecycleSpec)
        self.dev_mode = dev_mode
        self.entries: list[PluginEntry] = []
        cfg = config or load_mcp_config()
        eps = {ep.name: ep for ep in importlib.metadata.entry_points(group="ua.plugins")}
        for p in cfg.get("plugins", []):
            cls = self._resolve(p["name"], eps)
            obj = cls(**{k: v for k, v in p.items() if k not in ("name", "stage", "priority", "fail_closed")})
            self.pm.register(obj, name=p["name"])
            self.entries.append(PluginEntry(p["name"], p["stage"], p.get("priority", 100),
                                            bool(p.get("fail_closed", False)), True, obj, p))
        self.entries.sort(key=lambda e: (e.stage, e.priority))

    @staticmethod
    def _resolve(name, eps):
        if name in eps:
            try:
                return eps[name].load()
            except Exception:
                pass
        from . import builtin  # fallback when package entry points are not installed
        return {"pii_redactor": builtin.PiiRedactor, "injection_shield": builtin.InjectionShield,
                "circuit_breaker": builtin.CircuitBreakerPlugin, "ragas_auditor": builtin.RagasAuditor,
                "otel_exporter": builtin.OtelExporter}[name]

    # ---- control plane (Streamlit Tab 3)
    def get(self, name: str):
        return next((e.obj for e in self.entries if e.name == name), None)

    def set_enabled(self, name: str, enabled: bool) -> bool:
        for e in self.entries:
            if e.name == name:
                if not enabled and name in LOCKED and not self.dev_mode:
                    return False
                e.enabled = enabled
                return True
        return False

    def status(self) -> list[dict]:
        return [dict(name=e.name, stage=e.stage, priority=e.priority, fail_closed=e.fail_closed,
                     enabled=e.enabled) for e in self.entries]

    # ---- execution
    def _call_one(self, hook: str, entry: PluginEntry, **kw):
        others = [e.obj for e in self.entries if e is not entry]
        caller = self.pm.subset_hook_caller(hook, remove_plugins=others)
        return caller(**kw)

    def run(self, stage: str, **kw):
        hook = STAGE_HOOK[stage]
        for e in self.entries:
            if e.stage != stage or not e.enabled:
                continue
            try:
                self._call_one(hook, e, **kw)
            except Block:
                raise
            except UAError:
                raise
            except Exception as exc:
                if e.fail_closed:   # FMEA #7/#8: fail closed
                    code = -32007 if e.name == "pii_redactor" else -32008
                    raise Block(code, f"{e.name} failed closed: {type(exc).__name__}") from exc

    def pre(self, ctx: RequestContext) -> None:
        with span("hook.pre"):
            self.run("PRE_EXECUTION", ctx=ctx)

    def dispatch(self, call) -> None:
        self.run("TOOL_DISPATCH", call=call)

    def observe(self, call, result) -> None:
        for e in self.entries:
            if e.stage == "TOOL_DISPATCH" and e.enabled:
                try:
                    self._call_one("tool_result", e, call=call, result=result)
                except Exception:
                    pass

    def post(self, ctx, draft) -> None:
        with span("hook.post"):
            self.run("POST_EXECUTION", ctx=ctx, draft=draft)
