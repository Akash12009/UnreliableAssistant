"""RedisBackend against a real Redis (skipped if none). Start one: redis-server --port 6390 --save ''"""
import os
import time

import numpy as np
import pytest

URL = os.environ.get("UA_TEST_REDIS", "redis://127.0.0.1:6390/0")


@pytest.fixture(scope="module")
def backend():
    import redis
    try:
        redis.Redis.from_url(URL, socket_connect_timeout=0.3).ping()
    except Exception:
        pytest.skip("no redis server")
    from ua.cache import RedisBackend
    b = RedisBackend(URL)
    b.r.flushdb()
    yield b
    b.r.flushdb()


def test_isolation_and_semantic_scoping_on_real_redis(backend):
    from ua.cache import SemanticCache
    rng = np.random.default_rng(1)
    v = rng.normal(size=64).astype(np.float32)
    v /= np.linalg.norm(v)
    near = v + 0.01 * rng.normal(size=64).astype(np.float32)
    near /= np.linalg.norm(near)
    vecs = {"price of x": v, "what is the price of x": near}
    c = SemanticCache(backend, embed=lambda q: vecs.get(q, rng.normal(size=64).astype(np.float32) / 8))
    c.store("A", "u1", "price of x", {"answer": "A-private"}, "user")
    assert c.lookup("A", "u1", "price of x")["answer"] == "A-private"                    # exact
    assert c.lookup("A", "u1", "what is the price of x")["_cache"].startswith("semantic")  # semantic, same namespace
    assert c.lookup("A", "u2", "price of x") is None                                       # other user
    assert c.lookup("B", "u1", "price of x") is None                                       # other tenant
    c.store("A", "u1", "shared fact", {"answer": "pub"}, "tenant")
    assert c.lookup("A", "u9", "shared fact")["answer"] == "pub" and c.lookup("B", "u9", "shared fact") is None


def test_ttl_jitter_and_tenant_flush(backend):
    from ua.cache import SemanticCache
    c = SemanticCache(backend)
    for i in range(30):
        c.store("T1", "u", f"q{i}", {"a": i}, "user")
    c.store("T2", "u", "q", {"a": 1}, "user")
    ttls = [backend.r.ttl(k) for k in backend.r.scan_iter(match="{T1}:u:u:sc:*")]
    assert 250 <= min(ttls) and max(ttls) <= 346 and max(ttls) - min(ttls) >= 20          # 300 s +/- 45 s jitter
    assert backend.flush_tenant("T1") >= 30 and c.lookup("T2", "u", "q") is not None       # other tenant untouched


def test_redis_down_is_a_miss_not_an_error():
    from ua.cache import RedisBackend, SemanticCache
    c = SemanticCache(RedisBackend("redis://127.0.0.1:1/0"))                               # nothing listens on :1
    assert c.lookup("A", "u", "anything") is None and c.errors >= 1                        # FMEA #1: treat as miss
    c.store("A", "u", "q", {"a": 1}, "user")                                               # must not raise
