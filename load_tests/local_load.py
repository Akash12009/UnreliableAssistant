"""Local closed-loop load runner against a running API (the only load test executable in this environment).

Honest scope: this measures the REAL server at the concurrency this machine can drive (default 200 VUs); it does not
and cannot certify 50,000 VUs. It reports the same metrics as k6 (p50/p95/p99, RPS, 5xx rate, tenant leaks) plus
server RSS over time (psutil) for the soak KPI.

usage: python load_tests/local_load.py --base http://127.0.0.1:8000 --vus 200 --duration 60 --think 1.0 --pid <uvicorn pid>
"""
import argparse
import asyncio
import json
import random
import time

import httpx
import numpy as np

QUERIES = ["Verify batch LOT-88001 safety protocol and return triage status", "What is the unit price of SKU-007?",
           "Who signs off on big purchases of the Tundra Gauze?", "What is the shelf life of SKU-010?", "Summarize policy POL-03.",
           "Which supplier provides SKU-012?", "What is the status of LOT-88008?", "How many days is the return window under POL-02?"]


async def vu(i, a, client, stop, lat, errs, leaks, codes):
    tenant = f"tenant_{i % a.tenants}"
    rnd = random.Random(i)
    await asyncio.sleep(rnd.random() * a.ramp)          # ramp-up spread
    n = 0
    while time.time() < stop:
        q = QUERIES[(n + i) % len(QUERIES)]
        n += 1
        t = time.perf_counter()
        try:
            r = await client.post(f"{a.base}/api/v1/agent/query", json={"tenant_id": tenant, "user_id": f"user_{i}", "query": q},
                                  headers={"X-Tenant-ID": tenant}, timeout=30)
            ms = (time.perf_counter() - t) * 1000
            codes[r.status_code] = codes.get(r.status_code, 0) + 1
            if r.status_code >= 500:
                errs.append(r.status_code)
            elif r.status_code == 200:
                lat.append(ms)
                if r.json().get("tenant_id") != tenant:
                    leaks.append(1)
        except Exception as e:
            errs.append(type(e).__name__)
        await asyncio.sleep(a.think)


async def rss_sampler(pid, stop, out):
    import psutil
    p = psutil.Process(pid)
    while time.time() < stop:
        try:
            out.append((time.time(), p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))))
        except Exception:
            pass
        await asyncio.sleep(2)


async def main(a):
    lat, errs, leaks, codes, rss = [], [], [], {}, []
    stop = time.time() + a.duration + a.ramp
    limits = httpx.Limits(max_connections=a.vus + 10, max_keepalive_connections=a.vus + 10)
    async with httpx.AsyncClient(limits=limits) as client:
        # warm-up so first-request model/JIT costs aren't counted
        for q in QUERIES[:3]:
            await client.post(f"{a.base}/api/v1/agent/query", json={"tenant_id": "warm", "user_id": "w", "query": q}, timeout=60)
        t0 = time.time()
        tasks = [asyncio.create_task(vu(i, a, client, stop, lat, errs, leaks, codes)) for i in range(a.vus)]
        if a.pid:
            tasks.append(asyncio.create_task(rss_sampler(a.pid, stop, rss)))
        await asyncio.gather(*tasks)
        el = time.time() - t0
    n = len(lat) + len(errs)
    res = dict(vus=a.vus, duration_s=round(el, 1), think_s=a.think, requests=n, rps=round(n / el, 1),
               p50_ms=round(float(np.percentile(lat, 50)), 1) if lat else None, p95_ms=round(float(np.percentile(lat, 95)), 1) if lat else None,
               p99_ms=round(float(np.percentile(lat, 99)), 1) if lat else None, http_5xx_rate=round(len(errs) / max(1, n), 5),
               tenant_leaks=len(leaks), status_codes=codes)
    if lat:
        counts, edges = np.histogram(lat, bins=40, range=(0, float(np.percentile(lat, 99.5)) * 1.2))
        res["hist"] = dict(edges=[round(float(x), 1) for x in edges], counts=[int(c) for c in counts])
    if len(rss) > 5:
        base = np.mean([r for _, r in rss[2:7]])
        res["rss_mb_start"], res["rss_mb_end"] = round(base / 1e6, 1), round(np.mean([r for _, r in rss[-5:]]) / 1e6, 1)
        res["rss_growth_pct"] = round((res["rss_mb_end"] - res["rss_mb_start"]) / res["rss_mb_start"] * 100, 2)
    print(json.dumps(res, indent=1))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--vus", type=int, default=200)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--ramp", type=int, default=10)
    ap.add_argument("--think", type=float, default=1.0)
    ap.add_argument("--tenants", type=int, default=50)
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--out", default="")
    asyncio.run(main(ap.parse_args()))
