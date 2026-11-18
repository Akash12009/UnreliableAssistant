"""Runtime assembly shared by the FastAPI service and the Streamlit workbench."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .cache import MemoryBackend, RedisBackend, SemanticCache
from .config import SETTINGS
from .embedder import Embedder
from .harness import ToolExecutionHarness
from .loop import HotMemory
from .mcp_host import McpHost
from .orchestrator import Orchestrator
from .plugins.registry import LifecycleRegistry
from .runbooks import Runbooks
from .services import GraphService, RagService, StateService
from .store import StateStore
from .telemetry import RECORDER
from .triage import Triage


@dataclass
class Runtime:
    mode: str
    host: McpHost
    plugins: LifecycleRegistry
    harness: ToolExecutionHarness
    cache: SemanticCache
    embedder: Embedder
    orch: Orchestrator
    runbooks: Runbooks
    services: dict = field(default_factory=dict)

    async def close(self):
        await self.host.stop()


async def build_runtime(mode: str | None = None, sqlite_path: str | None = None, warm: bool = True) -> Runtime:
    mode = mode or os.environ.get("UA_TRANSPORT_MODE", "inproc")
    services: dict = {}
    if mode == "inproc":
        rag = RagService()
        services = {"hybrid-rag": rag, "graphify-vault": GraphService(), "state-store": StateService(StateStore(sqlite_path or SETTINGS.sqlite_path))}
        encoder = rag.engine.encoder
    else:
        from .rag.encoders import load_models
        encoder, _ = load_models()
    host = McpHost(mode, services)
    await host.start()
    plugins = LifecycleRegistry(dev_mode=os.environ.get("UA_DEV_MODE", "1") == "1")
    emb = Embedder(encoder)
    if warm:
        emb.warm(SETTINGS.vault_dir)
    backend = RedisBackend(SETTINGS.redis_url) if SETTINGS.redis_url else MemoryBackend()
    cache = SemanticCache(backend, embed=emb.query)
    harness = ToolExecutionHarness(host, plugins)
    llm = None
    if os.environ.get("UA_LLM", "").lower() == "groq":
        from .llm_groq import GroqLLM
        llm = GroqLLM()
        if not llm.configured:
            raise RuntimeError("UA_LLM=groq but GROQ_API_KEY is not set")
    orch = Orchestrator(harness, plugins, cache, Triage(), emb, HotMemory(SETTINGS.loop.hot_turns), llm=llm)
    otel = plugins.get("otel_exporter")
    if otel is not None and getattr(otel, "_tracer", None):
        RECORDER.sinks.append(otel.export)
    rb = Runbooks()
    rt = Runtime(mode, host, plugins, harness, cache, emb, orch, rb, services)
    rb.rt = rt
    return rt
