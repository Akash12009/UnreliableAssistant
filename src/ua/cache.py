"""Tenant-isolated semantic cache (ADR-003).

Key layout (Redis-Cluster friendly, `{tenant}` hash tag co-locates a tenant's keys):
    {tenant}:u:<user>:sc:<sha256(tenant:user:norm_query)>     user-scope  (personalised / higher risk)
    {tenant}:t:sc:<sha256(tenant:norm_query)>                 tenant-scope (shared, risk-tier-1 content only)
Semantic matching (cosine >= 0.95, or 0.92 under the latency-breach runbook) only ever happens INSIDE one
namespace; there is no global scan. TTL is jittered (300s +/- 45s) against stampedes.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from .config import SETTINGS
from .textnorm import norm_query


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def user_key(tenant: str, user: str, nq: str) -> str:
    return f"{{{tenant}}}:u:{user}:sc:{_sha(f'{tenant}:{user}:{nq}')}"


def tenant_key(tenant: str, nq: str) -> str:
    return f"{{{tenant}}}:t:sc:{_sha(f'{tenant}:{nq}')}"


def ns_user(tenant: str, user: str) -> str:
    return f"{{{tenant}}}:u:{user}"


def ns_tenant(tenant: str) -> str:
    return f"{{{tenant}}}:t"


class MemoryBackend:
    """Thread-safe in-process backend with the same semantics as Redis (TTL, namespaces)."""

    def __init__(self):
        self.kv: dict[str, tuple[float, dict]] = {}
        self.index: dict[str, OrderedDict[str, np.ndarray]] = {}
        self.lock = threading.RLock()
        self.cpu_hint = 0.0

    def get(self, key: str) -> dict | None:
        with self.lock:
            v = self.kv.get(key)
            if not v:
                return None
            if v[0] < time.time():
                self.kv.pop(key, None)
                return None
            return v[1]

    def set(self, key: str, val: dict, ttl: float, ns: str, emb: np.ndarray | None, cap: int) -> None:
        with self.lock:
            self.kv[key] = (time.time() + ttl, val)
            if emb is not None:
                idx = self.index.setdefault(ns, OrderedDict())
                idx[key] = emb
                idx.move_to_end(key)
                while len(idx) > cap:
                    old, _ = idx.popitem(last=False)
                    self.kv.pop(old, None)

    def neighbours(self, ns: str) -> list[tuple[str, np.ndarray]]:
        with self.lock:
            return list(self.index.get(ns, {}).items())

    def flush_tenant(self, tenant: str) -> int:
        pref = f"{{{tenant}}}:"
        with self.lock:
            ks = [k for k in self.kv if k.startswith(pref)]
            for k in ks:
                self.kv.pop(k, None)
            for ns in [n for n in self.index if n.startswith(pref)]:
                self.index.pop(ns, None)
        return len(ks)

    def all_keys(self) -> list[str]:
        with self.lock:
            return list(self.kv)

    def size(self) -> int:
        return len(self.kv)

    def ping(self) -> bool:
        return True


class RedisBackend:
    """Redis (Cluster-safe: every op touches keys sharing one `{tenant}` hash tag)."""

    def __init__(self, url: str):
        import redis
        self.r = redis.Redis.from_url(url, socket_timeout=0.05, socket_connect_timeout=0.05)

    def get(self, key):
        raw = self.r.get(key)
        return json.loads(raw) if raw else None

    def set(self, key, val, ttl, ns, emb, cap):
        p = self.r.pipeline(transaction=False)
        p.set(key, json.dumps(val), ex=max(1, int(ttl)))
        if emb is not None:
            ik = f"{ns}:idx"          # same hash tag as key -> same slot
            p.hset(ik, key, emb.astype(np.float32).tobytes())
            p.expire(ik, int(ttl) * 4)
        p.execute()
        if emb is not None and self.r.hlen(f"{ns}:idx") > cap:
            for k in list(self.r.hkeys(f"{ns}:idx"))[: self.r.hlen(f"{ns}:idx") - cap]:
                self.r.hdel(f"{ns}:idx", k)

    def neighbours(self, ns):
        out = []
        for k, v in self.r.hgetall(f"{ns}:idx").items():
            out.append((k.decode(), np.frombuffer(v, dtype=np.float32)))
        return out

    def flush_tenant(self, tenant):
        n = 0
        for k in self.r.scan_iter(match=f"{{{tenant}}}:*", count=500):
            self.r.delete(k); n += 1
        return n

    def all_keys(self):
        return [k.decode() for k in self.r.scan_iter(match="*", count=1000)]

    def size(self):
        return self.r.dbsize()

    def ping(self):
        try:
            return bool(self.r.ping())
        except Exception:
            return False


class SemanticCache:
    def __init__(self, backend=None, embed=None):
        self.backend = backend or MemoryBackend()
        self.embed = embed                  # callable str -> unit-norm np.ndarray (query embedding)
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self._lock = threading.Lock()

    def _threshold(self) -> float:
        c = SETTINGS.cache
        return c.cosine_aggressive if SETTINGS.aggressive_cache else c.cosine_normal

    def lookup(self, tenant: str, user: str, query: str) -> dict | None:
        """Exact then semantic, user namespace first, then tenant namespace. Never crosses tenants."""
        nq = norm_query(query)
        try:
            for key, ns in ((user_key(tenant, user, nq), ns_user(tenant, user)),
                            (tenant_key(tenant, nq), ns_tenant(tenant))):
                v = self.backend.get(key)
                if v is not None:
                    return self._hit(v, "exact")
            if self.embed is not None:
                q = self.embed(nq)
                thr = self._threshold()
                for ns in (ns_user(tenant, user), ns_tenant(tenant)):
                    best, best_s = None, thr
                    for k, e in self.backend.neighbours(ns):
                        s = float(np.dot(q, e))
                        if s >= best_s:
                            best, best_s = k, s
                    if best:
                        v = self.backend.get(best)
                        if v is not None:
                            return self._hit(v, f"semantic:{best_s:.3f}")
        except Exception:   # FMEA #1: Redis down -> treat as miss, never serve unscoped data
            with self._lock:
                self.errors += 1
        with self._lock:
            self.misses += 1
        return None

    def _hit(self, v: dict, kind: str) -> dict:
        with self._lock:
            self.hits += 1
        return {**v, "_cache": kind}

    def store(self, tenant: str, user: str, query: str, value: dict, scope: str) -> str:
        """scope: 'tenant' only for shared, non-personalised, risk-tier-1 answers (ADR-003)."""
        nq = norm_query(query)
        c = SETTINGS.cache
        ttl = c.ttl_s + random.uniform(-c.ttl_jitter_s, c.ttl_jitter_s)
        if scope == "tenant":
            key, ns = tenant_key(tenant, nq), ns_tenant(tenant)
        else:
            key, ns = user_key(tenant, user, nq), ns_user(tenant, user)
        emb = self.embed(nq) if self.embed is not None else None
        try:
            self.backend.set(key, {**value, "tenant_id": tenant, "scope": scope}, ttl, ns, emb,
                             c.max_entries_per_namespace)
        except Exception:
            with self._lock:
                self.errors += 1
        return key

    def hit_ratio(self) -> float:
        t = self.hits + self.misses
        return self.hits / t if t else 0.0
