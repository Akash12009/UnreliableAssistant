"""W3C TraceContext, in-process span recorder, percentile tracking, optional OTLP export.

Span names follow design.md Gate 10: hook.pre, cache.lookup, triage.classify, mcp.rag.bm25,
mcp.rag.biencoder, mcp.rag.rrf, mcp.rag.rerank, agent.react.step, llm.generate, hook.post.ragas.
"""
from __future__ import annotations

import contextlib
import contextvars
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np

_current_trace: contextvars.ContextVar["TraceCtx | None"] = contextvars.ContextVar("trace", default=None)


def new_traceparent() -> str:
    return f"00-{secrets.token_hex(16)}-{secrets.token_hex(8)}-01"


def parse_traceparent(tp: str) -> tuple[str, str]:
    _, trace_id, span_id, _ = tp.split("-")
    return trace_id, span_id


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_ns: int
    end_ns: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6


@dataclass
class TraceCtx:
    trace_id: str
    spans: list[Span] = field(default_factory=list)
    stack: list[str] = field(default_factory=list)


class Recorder:
    """Thread-safe rolling latency histograms + recent traces (bounded, drop-oldest)."""

    def __init__(self, per_name: int = 20000, traces: int = 200):
        self._lat: dict[str, deque] = defaultdict(lambda: deque(maxlen=per_name))
        self._traces: deque[list[Span]] = deque(maxlen=traces)
        self._counters: dict[str, float] = defaultdict(float)
        self._ts: dict[str, deque] = {}
        self._lock = threading.Lock()
        self.sinks: list[Callable[[Span], None]] = []   # e.g. otel_exporter plugin

    def record(self, span: Span) -> None:
        with self._lock:
            self._lat[span.name].append(span.ms)
        for s in self.sinks:
            try:
                s(span)
            except Exception:  # exporters must never break the request path (FMEA #10)
                pass

    def finish_trace(self, spans: list[Span]) -> None:
        with self._lock:
            self._traces.append(spans)

    def sample(self, name: str, value: float) -> None:
        """Timestamped sample for windowed alert rules (runbooks)."""
        with self._lock:
            self._ts.setdefault(name, deque(maxlen=200000)).append((time.time(), value))

    def window(self, name: str, seconds: float) -> list[float]:
        cut = time.time() - seconds
        with self._lock:
            return [v for t, v in self._ts.get(name, []) if t >= cut]

    def incr(self, name: str, v: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += v

    def counter(self, name: str) -> float:
        return self._counters.get(name, 0.0)

    def percentile(self, name: str, q: float) -> float | None:
        with self._lock:
            data = list(self._lat.get(name, []))
        return float(np.percentile(data, q)) if data else None

    def count(self, name: str) -> int:
        return len(self._lat.get(name, []))

    def recent_traces(self, n: int = 20) -> list[list[Span]]:
        with self._lock:
            return list(self._traces)[-n:]

    def reset(self) -> None:
        with self._lock:
            self._lat.clear(); self._traces.clear(); self._counters.clear(); self._ts.clear()


RECORDER = Recorder()


def start_trace(traceparent: str | None = None) -> TraceCtx:
    trace_id = parse_traceparent(traceparent)[0] if traceparent else secrets.token_hex(16)
    t = TraceCtx(trace_id=trace_id)
    _current_trace.set(t)
    return t


def current_trace() -> TraceCtx | None:
    return _current_trace.get()


@contextlib.contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    t = _current_trace.get()
    trace_id = t.trace_id if t else secrets.token_hex(16)
    parent = t.stack[-1] if t and t.stack else None
    s = Span(name, trace_id, secrets.token_hex(8), parent, time.perf_counter_ns(), attrs=dict(attrs))
    if t:
        t.stack.append(s.span_id)
    try:
        yield s
    finally:
        s.end_ns = time.perf_counter_ns()
        if t:
            t.stack.pop()
            t.spans.append(s)
        RECORDER.record(s)


def add_span(name: str, ms: float, **attrs: Any) -> None:
    """Record a span measured elsewhere (e.g. inside an MCP server, reported via stage_ms)."""
    t = _current_trace.get()
    now = time.perf_counter_ns()
    s = Span(name, t.trace_id if t else secrets.token_hex(16), secrets.token_hex(8),
             t.stack[-1] if t and t.stack else None, now - int(ms * 1e6), now, dict(attrs))
    if t:
        t.spans.append(s)
    RECORDER.record(s)
