# The Unreliable Assistant → Enterprise Grade

An MCP-enabled, plugin-extensible agentic assistant with 3-stage hybrid RAG, a bounded ReAct loop, tenant-isolated caching,
hash-chained audit, HITL-gated mutations, a red-team suite, a 36-KPI scorecard and a six-tab developer workbench.
Built from `Unreliable_Assistant_MidCheck_Enterprise_Blueprint_v2.docx` (18 gates). Authoritative design: **[design.md](design.md)**
(locked; deviations in its Appendix C). Latest KPI results: **[docs/KPI_SCORECARD.md](docs/KPI_SCORECARD.md)**.

> ## Read this first — what this is and is not
> * **It is NOT certified.** The scorecard has FAILs (see below). Several are hardware-bound; several are real quality gaps.
> * **By default no LLM is connected.** Generation is a deterministic *extractive* generator and planning is rule-based, both behind
>   `LLM`/`Brain` interfaces. A **Groq** generator/answerability-judge is implemented (`UA_LLM=groq`, see "Using Groq") but was only tested against a mocked endpoint — **never against the live API**. Planning stays rule-based. Faithfulness 1.0 / hallucination 0 are true *by construction*, not evidence about an LLM.
> * **The corpus is synthetic** (template-generated). Golden-set numbers are inflated; a held-out paraphrase set shows the gap.
> * **50,000 VUs were never run.** This machine has 4 vCPUs, no GPU. The k6/JMeter scripts exist and are unexecuted at scale;
>   KPI-31..36 are measured at ~50 VUs and labelled "scaled".
> * "RAGAS" metrics are deterministic RAGAS-*style* proxies (no judge LLM offline).

## Quick start
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.lock.txt   # Python 3.14 here; spec says 3.11+, code is 3.11-compatible
scripts/download_models.sh        # ONNX: bge-small-en-v1.5, ms-marco-MiniLM-L-6 (dev profile). INT8 variants: see eval/README
scripts/seed_dev.sh               # corpus -> index -> graph.json -> triage model -> signed MCP manifests
.venv/bin/python -m pytest -q     # 44 tests (3 need a local Redis on :6390, else skipped)
.venv/bin/streamlit run app.py    # developer workbench (6 tabs)
UA_SQLITE=data/state.db PYTHONPATH=src .venv/bin/uvicorn ua.api:app --port 8000   # API: POST /api/v1/agent/query
.venv/bin/python -m eval.run_kpis # full 36-KPI scorecard (~15 min) -> docs/KPI_SCORECARD.md
```
INT8 models are produced with `onnxruntime.quantization.quantize_dynamic` (see eval/README, ADR-002 acceptance).
Real MCP: `UA_TRANSPORT_MODE=mcp` makes the API talk to the three servers over stdio / Streamable HTTP
(`UA_TRANSPORT=streamable-http UA_PORT=8081 python mcp_servers/state_server.py`). Default `inproc` uses the same harness and JSON-RPC envelopes.

## Using Groq (LLM generator + answerability judge)
```bash
export GROQ_API_KEY=...            # never commit it; the code reads only this env var and never logs/returns it
export UA_LLM=groq                 # optional: UA_GROQ_MODEL (default llama-3.1-8b-instant), UA_GROQ_RPM (client rate limit, default 30),
                                   #           UA_GROQ_TIMEOUT_S (8), UA_GROQ_BASE_URL (default https://api.groq.com/openai/v1)
.venv/bin/streamlit run app.py     # or uvicorn ua.api:app ...
UA_LLM=groq .venv/bin/python -m eval.ragas_eval        # golden + held-out with the real model (287 calls; ~10 min at 30 RPM)
UA_LLM=groq .venv/bin/python -m eval.run_kpis          # full scorecard; header/notes record which generator produced the numbers
```
One call per answer: evidence (compressed, injection-scanned, passed as *untrusted data*) -> answer with `[ID]` citations or `ABSTAIN: reason`.
`ABSTAIN` is the answerability judge intended to fix KPI-23. Every LLM answer still passes the claim-support verifier; a vetoed draft is
retried once in quote-only mode, then abstains. 429 -> one bounded retry; any other Groq failure -> extractive fallback (`llm_fallback: true`).
**Data leaves the machine:** only the PII-redacted question and the retrieved evidence snippets are sent. Check Groq's data terms before using real content.
Model ids change; if `UA_GROQ_MODEL` is rejected the system falls back to extractive and counts `llm.err.http_4xx`.

## Gate → implementation map
| Gate | Delivered | Where |
|---|---|---|
| 0 stack / MCP / plugins | 3 MCP servers (FastMCP, `mcp<2`), Pluggy lifecycle registry with priorities + hot toggle, `mcp_config.json` (spec) + `mcp_config.prod.json` | `mcp_servers/`, `src/ua/plugins/`, `src/ua/mcp_host.py` |
| 1 design.md | 7 sections, 36 KPIs, 12-row FMEA, ADRs, spec-contradiction table | `design.md` |
| 2 workflow image | Graphviz, 300 DPI, 4 required groups; shown in workbench Tab 1 | `docs/workflow_orchestration.png`, `scripts/generate_workflow_diagram.py` |
| 3, 8 RAG + RAGAS | BM25Okapi+Porter inverted index → FAISS HNSW → RRF(k=60) → cross-encoder; 250-Q golden set; RAGAS-style gate runner | `src/ua/rag/`, `eval/ragas_eval.py`, `eval/ir_bench.py` |
| 4 harness + loops | strict Pydantic v2 → regex auto-repair → RBAC tiers → UUIDv4 idempotency; explicit state machine, max 5 iterations, per-step/total budgets, no-progress halt | `src/ua/harness.py`, `src/ua/loop.py` |
| 5 context | immutable `GoalAnchorBlock` at prompt[0] every step, ≤180-token extractive observation compression, last-3-turn hot memory + `session_facts` | `src/ua/loop.py`, `src/ua/agents.py` |
| 6, 7 triage/subagents | linear classifier on shared embedding (0.3 ms); retrieval (read-only) / execution (no RAG) / verification subagents with MCP allow-lists | `src/ua/triage.py`, `src/ua/contracts.py::TOOLS` |
| 9 patterns | idempotent circuit breaker → `HALTED_BREAKER`; bulkhead pools (64/128/256); deterministic abstention; async HITL (`PENDING_APPROVAL`, separation of duties) | `plugins/builtin.py`, `mcp_host.py`, `store.py` |
| 10, 11 observability / IR | W3C traceparent spans for every named hop, Prometheus `/metrics`, OTLP exporter, 4 runbooks with alert rules | `telemetry.py`, `runbooks.py`, `api.py` |
| 12 KPIs | 36 measured KPIs with PASS/FAIL/scaled/modelled verdicts | `eval/run_kpis.py` |
| 13, 14 governance / red team | per-tenant SHA-256 audit chain, zero raw PII persisted, HMAC-signed identity across MCP, signed server manifests; 100-prompt suite | `store.py`, `security.py`, `eval/redteam.py` |
| 15, 16 load | k6 (spec thresholds reconciled), JMeter distributed plan (XML valid, **not executed**), local load runner (executed) | `load_tests/` |
| 17 tokens | Obsidian vault + `graphify build ./src ./knowledge_vault --output graph.json`, depth-≤2 subgraph | `src/ua/graphify.py`, `knowledge_vault/` |
| 18 workbench | 6 tabs driving the real runtime (smoke-tested headlessly) | `app.py` |

## Verified behaviours (automated tests, `tests/`)
Injection blocked before retrieval · PII redacted before any tool/audit write · Tier-3 never executes without a *different* reviewer's approval ·
forged tenant/role rejected across the MCP boundary (`-32006`) · cache never crosses tenant/user (also on a real Redis) · unknown entity ⇒ abstain ·
hung RAG server ⇒ deterministic abstain · reranker failure ⇒ stricter fallback gate · infinite SKU retry ⇒ `HALTED_BREAKER` ·
audit tampering detected · BM25 matches the `rank_bm25` oracle · Redis outage ⇒ cache miss, not error.

## Known gaps (release blockers) — see docs/KPI_SCORECARD.md for current numbers
1. **Abstention on missing-attribute questions (KPI-23)** — ~73 % vs 98 % gate. No deterministic guard generalised (four attempts documented in
   `eval/README.md`). Needs an NLI/LLM judge in the Verification subagent.
2. **Latency (KPI-08/09/25/32/33/35)** — CPU-bound: the reranker alone is ~290 ms vs a 45 ms budget. Needs GPU serving (the `spec` profile
   is wired but unbenchmarked) and a cluster for 8,500 RPS.
3. **Answer relevancy generalisation (KPI-21)** — 0.97 on templated golden vs ~0.60 on held-out paraphrases. The extractive generator is a stand-in.
4. **Injection block rate (KPI-03)** — 98.3 % on a 60-prompt suite (95 % CI 91–99.7 %); gate 99.5 % cannot be statistically demonstrated with 60 prompts.
5. **Triage (KPI-04)** 95.7–97.8 % depending on test construction; **drift (KPI-18)** cosine ≈ 0.74 vs 0.85 (bge-small, short extractive answers).
6. Never run: 50K-VU k6/JMeter, GPU profile, PostgreSQL/pgvector (SQLite used), Prometheus/Tempo/Grafana/OTel collector (exporter wired, no sink), real LLM.

## Layout
```
design.md  docs/(workflow png, KPI_SCORECARD.md)  app.py  mcp_config*.json  graph.json  mcp_manifests/
mcp_servers/   rag_server.py graphify_server.py state_server.py (FastMCP)
src/ua/        config contracts harness mcp_host services store cache security telemetry runbooks runtime api
               orchestrator loop agents triage verify embedder graphify textnorm  plugins/  rag/{engine,index,bm25,encoders,guards}
eval/          ir_bench ragas_eval redteam(+data) kpi_suites run_kpis gen_bench metrics README.md results/
load_tests/    k6_50k_load.js jmeter_50k_plan.jmx local_load.py
scripts/       build_corpus build_index train_triage sign_manifests generate_workflow_diagram download_models seed_dev
tests/         security_plugins core pipeline redis_backend
```

## Security notes
Dev defaults for `UA_MCP_SECRET`, `UA_MANIFEST_KEY`, `UA_ADMIN_TOKEN` are insecure placeholders — set real secrets and `UA_DEV_MODE=0`
(which locks `pii_redactor`/`injection_shield` against hot-disable and removes the admin endpoints). Tenant auth at the gateway is assumed:
the API trusts `tenant_id` from the (authenticated) request and cross-checks `X-Tenant-ID`; wire it to your IdP before production.
Lazy per-tenant item seeding in `StateService` is demo behaviour.
