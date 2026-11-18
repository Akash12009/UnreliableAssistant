"""MCP host: JSON-RPC 2.0 client for the three MCP servers.

Modes
  inproc : services called in-process through bulkheaded thread pools; the JSON-RPC envelope is still built and
           parsed so serialization overhead (KPI-30) is real. Used by the Streamlit workbench and local load runs.
  mcp    : real MCP sessions (stdio for local workers, Streamable HTTP for distributed services) via `mcp.ClientSession`.
Both implement the same `call(tool, args, meta)` contract, so the harness is transport-agnostic.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from .config import ROOT, SETTINGS, load_mcp_config, scaled
from .contracts import TOOLS, Meta
from .errors import UAError
from .security import sign_meta, verify_manifest
from .telemetry import RECORDER, add_span

SERVER_POOL = {"hybrid-rag": "rag", "graphify-vault": "rag", "state-store": "db"}


@dataclass
class ServerStatus:
    name: str
    transport: str
    state: str = "down"            # up | down | degraded
    calls: int = 0
    errors: int = 0
    last_ms: float = 0.0
    last_error: str = ""
    manifest: str = ""
    restarts: int = 0


class McpHost:
    def __init__(self, mode: str = "inproc", services: dict | None = None, config: dict | None = None):
        self.mode = mode
        self.cfg = config or load_mcp_config()
        self.services = services or {}
        self.pools = {"rag": ThreadPoolExecutor(SETTINGS.rag_pool, thread_name_prefix="bulk-rag"),
                      "db": ThreadPoolExecutor(SETTINGS.db_pool, thread_name_prefix="bulk-db")}
        self.status = {n: ServerStatus(n, c.get("transport", "stdio")) for n, c in self.cfg["mcpServers"].items()}
        self._sessions: dict[str, Any] = {}
        self._runners: dict[str, asyncio.Task] = {}
        self._stop: dict[str, asyncio.Event] = {}
        self._id = 0
        self.fail_injected: set[str] = set()       # chaos hook: simulate a hung/crashed server (FMEA #3)
        self.latency_injected: dict[str, float] = {}
        self.bulkhead_open: set[str] = set()

    # ---------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        if self.mode == "inproc":
            for n in self.status:
                self.status[n].state = "up" if n in self.services else "down"
            return
        for name in self.cfg["mcpServers"]:
            await self._start_server(name)

    def _params(self, name: str):
        from mcp import StdioServerParameters
        c = self.cfg["mcpServers"][name]
        cmd, args = c["command"], list(c.get("args", []))
        # config names `uv run <script>` (spec). Use the active interpreter unless UA_USE_UV=1 (avoids a re-resolve).
        if cmd == "uv" and os.environ.get("UA_USE_UV") != "1":
            cmd, args = sys.executable, args[1:] if args[:1] == ["run"] else args
        args = [str(ROOT / a) if a.endswith(".py") and not os.path.isabs(a) else a for a in args]
        return StdioServerParameters(command=cmd, args=args, env={**os.environ, "PYTHONUNBUFFERED": "1"})

    async def _start_server(self, name: str) -> None:
        st, c = self.status[name], self.cfg["mcpServers"][name]
        if c.get("transport", "stdio") == "stdio":
            ok, why = verify_manifest(name)
            st.manifest = why
            if not ok:                       # signed-manifest enforcement (Gate 13): refuse to launch tampered servers
                st.state, st.last_error = "down", f"manifest check failed: {why}"
                return
        ready = asyncio.Event()
        stop = asyncio.Event()
        self._stop[name] = stop

        async def runner():
            from mcp import ClientSession
            try:
                async with contextlib.AsyncExitStack() as stack:
                    if c.get("transport", "stdio") == "stdio":
                        from mcp.client.stdio import stdio_client
                        r, w = await stack.enter_async_context(stdio_client(self._params(name)))
                    else:
                        from mcp.client.streamable_http import streamablehttp_client
                        r, w, _ = await stack.enter_async_context(streamablehttp_client(c["url"]))
                    sess = await stack.enter_async_context(ClientSession(r, w))
                    await asyncio.wait_for(sess.initialize(), 30)
                    self._sessions[name] = sess
                    st.state = "up"
                    ready.set()
                    await stop.wait()
            except Exception as e:
                st.state, st.last_error = "down", f"{type(e).__name__}: {str(e)[:200]}"
            finally:
                self._sessions.pop(name, None)
                st.state = "down"
                ready.set()

        self._runners[name] = asyncio.create_task(runner())
        await ready.wait()

    async def stop(self) -> None:
        for ev in self._stop.values():
            ev.set()
        await asyncio.gather(*self._runners.values(), return_exceptions=True)
        for p in self.pools.values():
            p.shutdown(wait=False, cancel_futures=True)

    async def restart(self, name: str) -> None:
        """Supervisor action from the MCP-hang runbook: restart the worker."""
        if name in self._stop:
            self._stop[name].set()
            await asyncio.gather(self._runners[name], return_exceptions=True)
        self.status[name].restarts += 1
        self.fail_injected.discard(name)
        self.bulkhead_open.discard(name)
        if self.mode == "mcp":
            await self._start_server(name)
        else:
            self.status[name].state = "up" if name in self.services else "down"

    # ---------------------------------------------------------------- calls
    async def call(self, tool: str, args: dict, meta: Meta, timeout_ms: float | None = None) -> dict:
        server = TOOLS[tool]["server"]
        st = self.status[server]
        timeout = scaled(timeout_ms or SETTINGS.mcp_timeout_ms) / 1000
        if server in self.bulkhead_open:
            raise UAError(-32010, f"{server} bulkhead open (isolated after repeated timeouts)", retryable=True)
        if st.state == "down" and self.mode == "mcp":
            raise UAError(-32010, f"{server} unavailable: {st.last_error or 'not started'}", retryable=True)
        meta = meta.model_copy(update={"auth": sign_meta(meta)})
        # ---- JSON-RPC 2.0 envelope (serialization cost is measured; KPI-30)
        t_ser = time.perf_counter()
        self._id += 1
        wire = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": "tools/call",
                           "params": {"name": tool, "arguments": {"args": args, "meta": meta.model_dump(mode="json")}}})
        ser_ms = (time.perf_counter() - t_ser) * 1000
        t0 = time.perf_counter()
        try:
            if server in self.fail_injected:
                await asyncio.sleep(timeout + 0.05)
                raise asyncio.TimeoutError()
            if server in self.latency_injected:
                await asyncio.sleep(self.latency_injected[server] / 1000)
            if self.mode == "inproc":
                t_p = time.perf_counter()
                params = json.loads(wire)["params"]["arguments"]
                ser_ms += (time.perf_counter() - t_p) * 1000
                loop = asyncio.get_running_loop()
                svc = self.services[server]
                res = await asyncio.wait_for(loop.run_in_executor(
                    self.pools[SERVER_POOL[server]], svc.call, tool, params["args"], params["meta"]), timeout)
            else:
                sess = self._sessions[server]
                raw = await asyncio.wait_for(sess.call_tool(tool, {"args": args, "meta": meta.model_dump(mode="json")},
                                                            read_timeout_seconds=timedelta(seconds=timeout)), timeout)
                res = self._parse(raw)
        except (asyncio.TimeoutError, TimeoutError):
            st.errors += 1
            st.last_error = f"timeout>{timeout * 1000:.0f}ms"
            RECORDER.incr(f"mcp.timeout.{server}")
            if RECORDER.counter(f"mcp.timeout.{server}") % 5 == 0 and self.mode == "inproc":
                pass
            raise UAError(-32001, f"{tool} timed out after {timeout * 1000:.0f} ms", retryable=TOOLS[tool]["tier"].value == "TIER_1_READ")
        rt_ms = (time.perf_counter() - t0) * 1000
        t_p = time.perf_counter()
        json.dumps(res)                                 # response-side serialization
        ser_ms += (time.perf_counter() - t_p) * 1000
        server_ms = float(res.get("server_ms", 0.0))
        st.calls += 1
        st.last_ms = rt_ms
        overhead = ser_ms + max(0.0, rt_ms - server_ms) if self.mode == "mcp" else ser_ms
        add_span("mcp.jsonrpc.overhead", overhead, server=server, tool=tool)
        for k, v in (res.get("data") or {}).get("stage_ms", {}).items() if isinstance(res.get("data"), dict) else []:
            if k != "total":
                add_span({"bm25": "mcp.rag.bm25", "biencoder": "mcp.rag.biencoder", "rrf": "mcp.rag.rrf", "rerank": "mcp.rag.rerank"}.get(k, f"mcp.rag.{k}"), v)
        res["_rt_ms"], res["_overhead_ms"] = rt_ms, overhead
        return res

    @staticmethod
    def _parse(raw) -> dict:
        if getattr(raw, "structuredContent", None):
            sc = raw.structuredContent
            return sc.get("result", sc) if set(sc) == {"result"} else sc
        for c in raw.content:
            if getattr(c, "text", None):
                try:
                    return json.loads(c.text)
                except Exception:
                    return dict(ok=False, error=dict(code=-32603, name="INTERNAL_ERROR", message=c.text[:300], retryable=False))
        return dict(ok=False, error=dict(code=-32603, name="INTERNAL_ERROR", message="empty MCP result", retryable=False))

    def snapshot(self) -> list[dict]:
        return [dict(server=s.name, transport=s.transport, state=s.state, calls=s.calls, errors=s.errors,
                     last_ms=round(s.last_ms, 2), last_error=s.last_error, manifest=s.manifest, restarts=s.restarts,
                     bulkhead="OPEN" if s.name in self.bulkhead_open else "closed") for s in self.status.values()]
