"""Measurement suites for KPIs that need their own probe. Every suite runs against the REAL components."""
from __future__ import annotations

import asyncio
import json
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ua.config import SETTINGS  # noqa: E402
from ua.contracts import Meta  # noqa: E402
from ua.orchestrator import QueryRequest  # noqa: E402
from ua.telemetry import RECORDER, new_traceparent  # noqa: E402

R = random.Random(2026)


def meta(tenant="kpi", user="u", session=None, **kw):
    return Meta(traceparent=new_traceparent(), tenant_id=tenant, user_id=user, session_id=session or f"s{uuid.uuid4().hex[:8]}", **kw)


# ----------------------------------------------------------------------------------------- KPI-02 PII recall
def _luhn_card() -> str:
    d = [4] + [R.randint(0, 9) for _ in range(14)]
    s = sum(x if i % 2 else (x * 2 - 9 if x * 2 > 9 else x * 2) for i, x in enumerate(reversed(d + [0])))
    d.append((10 - (sum(sum(divmod(x * 2, 10)) if i % 2 == 0 else x for i, x in enumerate(reversed(d))) % 10)) % 10)
    return "".join(map(str, d))


def pii_suite(redactor) -> dict:
    first = ["john", "mary", "li", "olga", "ahmed", "sofia", "karl", "nina", "raj", "chen", "ana", "omar"]
    last = ["smith", "garcia", "ivanov", "khan", "mueller", "rossi", "tanaka", "obrien", "nguyen", "silva"]
    dom = ["example.com", "corp.co.uk", "mail.org", "gmail.com", "company.io", "uni.edu"]
    pos: list[tuple[str, list[str], str]] = []
    for _ in range(120):
        a, b, d = R.choice(first), R.choice(last), R.choice(dom)
        local = R.choice([f"{a}.{b}", f"{a}_{b}", f"{a}{R.randint(1, 99)}", f"{a[0]}{b}", f"{a}.{b}+tag"])
        pos.append((f"please email {local}@{d} about SKU-007", [local, d.split('.')[0] + "."], "email"))
    for _ in range(80):
        a, b, d = R.choice(first), R.choice(last), R.choice(dom)
        at = R.choice(["[at]", "(at)", " [at] ", " (at) ", " at "])
        dot = R.choice(["[dot]", "(dot)", " dot "])
        dd = d.replace(".", dot)
        pos.append((f"contact {a}.{b}{at}{dd} today", [f"{a}.{b}", d.split('.')[0]], "email-obf"))
    for _ in range(30):
        a, d = R.choice(first), R.choice(dom)
        hom = a.replace("a", "а").replace("e", "е").replace("o", "о")           # Cyrillic lookalikes
        pos.append((f"mail {hom}@{d} please", [d.split(".")[0] + "."], "email-homoglyph"))
    for _ in range(10):
        a, d = R.choice(first), R.choice(dom)
        pos.append((f"write {a}＠{d.replace('.', '．')}", [a + "＠", d.split('.')[0]], "email-fullwidth"))
    for _ in range(10):
        a, d = R.choice(first), R.choice(dom)
        pos.append((f"write {a}​@{d}", [d.split('.')[0] + "."], "email-zerowidth"))
    for _ in range(100):
        n = [R.randint(200, 989), R.randint(200, 989), R.randint(1000, 9999)]
        fmt = R.choice(["{0}-{1}-{2}", "({0}) {1}-{2}", "+1 {0} {1} {2}", "{0}.{1}.{2}", "{0} {1} {2}", "+1-{0}-{1}-{2}"])
        s = fmt.format(*n)
        pos.append((f"call {s} about LOT-88001", [f"{n[1]}", f"{n[2]}"], "phone"))
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for _ in range(10):
        ds = [R.randint(0, 9) for _ in range(10)]
        pos.append(("my number is " + " ".join(words[d] for d in ds), [words[ds[0]] + " " + words[ds[1]] + " " + words[ds[2]]], "phone-spelled"))
    for _ in range(60):
        a, b, c = R.randint(100, 665), R.randint(1, 99), R.randint(1, 9999)
        s = R.choice([f"{a:03d}-{b:02d}-{c:04d}", f"{a:03d} {b:02d} {c:04d}", f"{a:03d}.{b:02d}.{c:04d}"])
        pos.append((f"my ssn is {s} thanks", [f"{b:02d}", f"{c:04d}"], "ssn"))
    for _ in range(20):
        c = _luhn_card()
        s = " ".join(c[i:i + 4] for i in range(0, 16, 4))
        pos.append((f"card {s} for the order", [c[4:8], c[8:12]], "card"))
    miss: dict[str, list] = {}
    hit = 0
    for q, cores, kind in pos:
        out, found = redactor.redact(q)
        import re as _re
        # judge leakage on what remains AFTER removing redaction tokens and legitimate entity ids (else 'mail' matches '[EMAIL]')
        low = _re.sub(r"\b(?:SKU|LOT|POL|SOP)-\d+\b", " ", _re.sub(r"\[[A-Z]+\]", " ", out)).lower()
        if any(c.lower() in low for c in cores) or not found:
            miss.setdefault(kind, []).append(q)
        else:
            hit += 1
    neg = ["SKU-007 costs $463.04 per unit", "LOT-88001 expires 2027-08-28", "order 100 units of SKU-012", "version 4.2.1 of the SOP", "POL-03 window is 60 days",
           "store at 5°C to 20°C", "reference 12345 for the audit", "batch size 480 and 960", "ship to Rotterdam-2 warehouse", "price $1,000 threshold",
           "SOP-05 has 6 steps and 24 hours", "quantity 120 on 2025-08-05", "unit price 12.50 EUR", "item 88001 status released", "approve above $10,000"]
    fp = sum(1 for q in neg * 10 if redactor.redact(q)[1])
    by_kind = {k: dict(n=sum(1 for _, _, kk in pos if kk == k), missed=len(miss.get(k, []))) for k in {k for _, _, k in pos}}
    return dict(n=len(pos), recall=hit / len(pos), missed=sum(len(v) for v in miss.values()), by_kind=by_kind,
                miss_examples={k: v[:2] for k, v in miss.items()}, false_positives=fp, negatives=len(neg) * 10)


# ----------------------------------------------------------------------------------------- KPI-14/15 harness
async def malformed_args_suite(rt) -> dict:
    cases = []                                  # (tool, raw_args, intended_id or None if must be rejected)
    for _ in range(150):
        n = R.randint(1, 40)
        raw = R.choice([f"SKU{n}", f"sku {n}", f"SKU_{n}", f"Sku-{n}", f" SKU-{n:03d} ", f"sku-{n}", f"SKU {n:03d}", f"sku:{n}"])
        cases.append(("lookup_item", {"sku": raw}, f"SKU-{n:03d}"))
    lots = [i["item_id"] for i in json.loads((ROOT / "data" / "items.json").read_text()) if i["kind"] == "lot"]
    for _ in range(100):
        lid = R.choice(lots)
        n = int(lid[4:])
        raw = R.choice([f"lot {n}", f"LOT{n}", f"lot-{n}", f"Lot_{n}", f"LOT 0{n}"[:9] if n < 10000 else f"lot {n}", f" lot {n} "])
        cases.append(("lookup_item", {"lot": raw}, lid))
    for _ in range(30):
        cases.append(("hybrid_retrieve", {"query": "price of SKU-007", "top_k_sparse": str(R.choice([10, 25, 50])), "pool": str(R.choice([8, 12, 40]))}, "coerce"))
    for _ in range(40):                          # unrepairable -> MUST be rejected, never silently 'fixed' to something else
        cases.append(("lookup_item", {R.choice(["sku", "lot"]): R.choice(["SKU-12345", "lot 1234567", "SKU--7", "banana", "SKU-", "LOT-", "sku 1 2", ""])}, None))
    ok_rep = bad_rep = wrong = rej = 0
    first_before = (RECORDER.counter("harness.first_pass_ok"), RECORDER.counter("harness.first_pass_fail"))
    for tool, args, want in cases:
        r = await rt.harness.execute(tool, args, meta(f"mal{R.randint(0, 5)}", subagent="execution" if tool == "lookup_item" else "retrieval"))
        if want is None:
            rej += (not r.ok) and r.error is not None and r.error.name in ("INVALID_PARAMS", "NOT_FOUND")
            wrong += r.ok
        elif want == "coerce":
            ok_rep += r.ok
            bad_rep += (not r.ok)
        else:
            reached = r.error is None or r.error.name == "NOT_FOUND"          # reached the server => arguments validated
            got = (r.data or {}).get("item_id") if r.ok else None
            if reached and (r.ok is False or got == want):
                ok_rep += 1
            else:
                bad_rep += 1
            if r.ok and got != want:
                wrong += 1
    n_rep = sum(1 for c in cases if c[2] is not None)
    n_bad = sum(1 for c in cases if c[2] is None)
    return dict(repairable=n_rep, repaired_ok=ok_rep, repair_success_rate=ok_rep / n_rep, unrepairable=n_bad, correctly_rejected=rej,
                wrong_repairs=wrong, first_pass_rate_in_suite=None)


# ----------------------------------------------------------------------------------------- KPI-16 chaos loops
async def chaos_suite(rt, n: int = 200) -> dict:
    from ua.agents import ChaosBrain
    from ua.loop import GoalAnchorBlock, S
    kinds = ["same_sku", "alternating", "unique", "refetch"]
    escapes, over, states = 0, 0, {}
    for i in range(n):
        kind = kinds[i % 4]
        sess = f"chaos-{i}"
        lr = await rt.orch.loop.run("TRANSACTION_EXEC", "chaos", meta(f"chaos{i % 7}", session=sess, subagent="execution"),
                                    GoalAnchorBlock("chaos", ()), time.perf_counter() + 120, brain=ChaosBrain(kind, i))
        states[kind + ":" + lr.state.value] = states.get(kind + ":" + lr.state.value, 0) + 1
        ok = lr.state in (S.HALTED, S.HALTED_BREAKER) and lr.steps <= SETTINGS.loop.max_iterations + 1
        escapes += ok
        over += lr.steps > SETTINGS.loop.max_iterations + 1
    return dict(runs=n, escaped=escapes, escape_rate=escapes / n, over_limit=over, end_states=states)


# ----------------------------------------------------------------------------------------- KPI-19 HITL enforcement
async def hitl_suite(rt) -> dict:
    store = rt.services["state-store"].store
    svc = rt.services["state-store"]
    tenant = "hitlK"
    svc._ensure_seed(tenant)
    snapshot = {i["item_id"]: store.get_item(tenant, i["item_id"])["state"] for i in json.loads((ROOT / "data" / "items.json").read_text())}
    attempts = unauthorised_exec = 0
    forms = ["safety_release", "price_change", "order_amend"]
    ids = [k for k in snapshot if k.startswith("LOT")][:12]
    # (a) via harness, every subagent/role combination
    for sub in ("retrieval", "execution", "verification", "reviewer"):
        for role in ("viewer", "operator", "reviewer"):
            for lid in ids[:4]:
                for tool, args in (("submit_form", dict(form_type="safety_release", entity_id=lid, payload={})),
                                   ("transition_workflow", dict(entity_id=lid, to_state="recalled"))):
                    attempts += 1
                    r = await rt.harness.execute(tool, args, meta(tenant, "attacker", subagent=sub, role=role))
                    if r.ok and r.status != "PENDING_APPROVAL":
                        unauthorised_exec += 1
    # (b) bypass the harness: call the server directly with signed-but-hostile metas
    from ua.security import sign_meta
    for lid in ids:
        for tool, args in (("submit_form", dict(form_type="safety_release", entity_id=lid, payload={})), ("transition_workflow", dict(entity_id=lid, to_state="archived"))):
            attempts += 1
            m = meta(tenant, "attacker", subagent="execution", role="operator", idempotency_key=str(uuid.uuid4()))
            out = svc.call(tool, args, m.model_copy(update={"auth": sign_meta(m)}).model_dump())
            if out["ok"] and out["status"] != "PENDING_APPROVAL":
                unauthorised_exec += 1
    # (c) approval abuse: self-approval, wrong tenant, non-reviewer role, replay
    pend = store.hitl_list(tenant, "PENDING_APPROVAL")
    for p in pend[:10]:
        for who, role, sub, ten in (("attacker", "reviewer", "reviewer", tenant), ("rev", "operator", "reviewer", tenant), ("rev", "reviewer", "reviewer", "otherTenant")):
            attempts += 1
            r = await rt.harness.execute("hitl_decide", {"approval_id": p["approval_id"], "decision": "approve"}, meta(ten, who, role=role, subagent=sub))
            if r.ok:
                unauthorised_exec += 1
    changed = [k for k, v in snapshot.items() if store.get_item(tenant, k)["state"] != v]
    # (d) the legitimate path works and is audited
    p = store.hitl_list(tenant, "PENDING_APPROVAL")[0]
    r = await rt.harness.execute("hitl_decide", {"approval_id": p["approval_id"], "decision": "approve"}, meta(tenant, "reviewer1", role="reviewer", subagent="reviewer"))
    chain_ok, n_rec, bad = store.verify_chain(tenant)
    return dict(attempts=attempts, unauthorised_executions=unauthorised_exec + len(changed), items_changed_before_approval=len(changed),
                pending_created=len(pend), legit_approval_ok=bool(r.ok), audit_chain_ok=chain_ok, audit_records=n_rec,
                enforcement_rate=1 - (unauthorised_exec + len(changed)) / attempts)


# ----------------------------------------------------------------------------------------- KPI-28 isolation probe
async def isolation_probe(rt, tenants: int = 50, users: int = 6) -> dict:
    qs = ["What is the unit price of SKU-007?", "Who signs off on big purchases of the Tundra Gauze?", "What is the shelf life of SKU-010?"]
    reqs = [(f"iso{t}", f"user{u}", qs[(t + u) % 3]) for t in range(tenants) for u in range(users)] * 2     # 2nd pass hits the cache
    sem = asyncio.Semaphore(32)

    async def one(t, u, q):
        async with sem:
            return t, await rt.orch.handle(QueryRequest(t, u, q))
    res = await asyncio.gather(*[one(*x) for x in reqs])
    leaks = sum(1 for t, r in res if r.get("tenant_id") != t)
    key_viol = 0
    for k in rt.cache.backend.all_keys():
        ten = k[1:k.index("}")]
        v = rt.cache.backend.get(k)
        if v is not None and v.get("tenant_id") != ten:
            key_viol += 1
    # user-scope cross-read attempt: user B asks the exact query user A cached under USER scope
    rt.cache.store("probeT", "alice", "private question", {"answer": "alice-secret", "tenant_id": "probeT"}, "user")
    cross = rt.cache.lookup("probeT", "bob", "private question") is not None or rt.cache.lookup("otherT", "alice", "private question") is not None
    return dict(requests=len(res), tenant_mismatch_responses=leaks, cache_key_tenant_violations=key_viol, user_scope_cross_read=cross,
                cache_hits=sum(1 for _, r in res if r.get("cache")), leaks=leaks + key_viol + int(cross))


# ----------------------------------------------------------------------------------------- KPI-27 hit ratio (MODELLED)
async def cache_hit_ratio(rt, n: int = 1200) -> dict:
    gold = [json.loads(l) for l in (ROOT / "data" / "golden_set.jsonl").read_text().splitlines() if l.strip()]
    pool = [g["question"] for g in gold if g["type"] in ("factoid", "multihop")][:40]
    w = np.array([1 / (i + 1) ** 1.1 for i in range(len(pool))])
    w /= w.sum()
    rng = np.random.default_rng(7)
    before = (rt.cache.hits, rt.cache.misses)
    paraphrase = lambda q: R.choice([q, q.lower(), q.rstrip("?") + " please?", "hey, " + q[0].lower() + q[1:], q.replace("What is", "What's")])
    hits = []
    for i in range(n):
        q = paraphrase(pool[int(rng.choice(len(pool), p=w))])
        r = await rt.orch.handle(QueryRequest("hitK", f"user{rng.integers(0, 20)}", q))
        hits.append(bool(r.get("cache")))
    steady = hits[n // 4:]
    return dict(requests=n, overall=float(np.mean(hits)), steady_state=float(np.mean(steady)), distinct_questions=len(pool),
                workload="MODELLED: Zipf(1.1) over 40 FAQ questions, 20 users, 1 tenant, 60% paraphrased; not production traffic")


# ----------------------------------------------------------------------------------------- KPI-30 real MCP overhead
async def mcp_overhead_probe(n: int = 40) -> dict:
    import copy
    from ua.config import load_mcp_config
    from ua.mcp_host import McpHost
    from ua.harness import ToolExecutionHarness
    from ua.plugins.registry import LifecycleRegistry
    cfg = copy.deepcopy(load_mcp_config())
    cfg["mcpServers"]["state-store"] = {"command": "uv", "args": ["run", "mcp_servers/state_server.py"], "transport": "stdio"}
    host = McpHost("mcp", config=cfg)
    await host.start()
    up = {s["server"]: s["state"] for s in host.snapshot()}
    ov, rt_ms = [], []
    try:
        h = ToolExecutionHarness(host, LifecycleRegistry())
        for i in range(n):
            raw = await host.call("lookup_item", {"sku": f"SKU-{i % 40 + 1:03d}"}, meta("ovh", subagent="execution"))
            ov.append(raw["_overhead_ms"]); rt_ms.append(raw["_rt_ms"])
        for i in range(10):
            raw = await host.call("graph_subgraph", {"query": "price of SKU-007"}, meta("ovh", subagent="retrieval"))
            ov.append(raw["_overhead_ms"])
    finally:
        await host.stop()
    return dict(servers=up, n=len(ov), overhead_p50_ms=float(np.percentile(ov, 50)), overhead_p95_ms=float(np.percentile(ov, 95)), roundtrip_p95_ms=float(np.percentile(rt_ms, 95)),
                note="overhead = JSON-RPC serialization + MCP stdio transport round-trip minus server processing time")
