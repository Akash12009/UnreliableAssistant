import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
async def rt():
    from ua.config import SETTINGS
    from ua.orchestrator import QueryRequest
    from ua.runtime import build_runtime
    # Tests assert behaviour, not CPU speed: generous timeouts + a warm-up so a cold ONNX session cannot trip the (correct, by-design)
    # timeout->abstain degradation in an unrelated test.
    SETTINGS.budget_scale = 20
    r = await build_runtime(sqlite_path=f"{tempfile.mkdtemp()}/t.db")
    for q in ("What is the unit price of SKU-007?", "Who signs off on big purchases of the Tundra Gauze?"):
        await r.orch.handle(QueryRequest("warmup", "w", q))
    yield r
    await asyncio.sleep(0.2)
    await r.close()


def meta(tenant="tA", user="u1", session="s1", **kw):
    from ua.contracts import Meta
    from ua.telemetry import new_traceparent
    return Meta(traceparent=new_traceparent(), tenant_id=tenant, user_id=user, session_id=session, **kw)
