"""State store: session_facts, hash-chained audit log, HITL approvals, items.

SQLite for local dev (design Appendix A #14); the SQL is portable to PostgreSQL. EVERY query carries
`tenant_id` in its predicate (design 1.3 invariant 1). Audit payloads are PII-checked before persistence.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS session_facts(
  tenant_id TEXT NOT NULL, session_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, ts REAL NOT NULL,
  PRIMARY KEY(tenant_id, session_id, key));
CREATE TABLE IF NOT EXISTS audit_log(
  tenant_id TEXT NOT NULL, seq INTEGER NOT NULL, ts REAL NOT NULL, event TEXT NOT NULL,
  payload TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL,
  PRIMARY KEY(tenant_id, seq));
CREATE TABLE IF NOT EXISTS hitl(
  approval_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, tool TEXT NOT NULL, args TEXT NOT NULL,
  requested_by TEXT NOT NULL, status TEXT NOT NULL, decided_by TEXT, created REAL NOT NULL,
  decided REAL, expires REAL NOT NULL, idem TEXT);
CREATE TABLE IF NOT EXISTS items(
  tenant_id TEXT NOT NULL, item_id TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL, data TEXT NOT NULL,
  PRIMARY KEY(tenant_id, item_id));
CREATE TABLE IF NOT EXISTS idempotency(
  tenant_id TEXT NOT NULL, idem TEXT NOT NULL, args_hash TEXT NOT NULL, result TEXT NOT NULL, ts REAL NOT NULL,
  PRIMARY KEY(tenant_id, idem));
"""
GENESIS = "0" * 64
HITL_TTL_S = 24 * 3600


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class StateStore:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL") if path != ":memory:" else None
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()
        from .plugins.builtin import PiiRedactor
        self._pii = PiiRedactor()

    # ------------------------------------------------------------ session facts
    def put_fact(self, tenant_id: str, session_id: str, key: str, value: str) -> None:
        clean, found = self._pii.redact(value)      # zero raw PII persistence
        with self.lock:
            self.db.execute("INSERT INTO session_facts VALUES(?,?,?,?,?) ON CONFLICT(tenant_id,session_id,key) "
                            "DO UPDATE SET value=excluded.value, ts=excluded.ts",
                            (tenant_id, session_id, key, clean, time.time()))

    def get_facts(self, tenant_id: str, session_id: str) -> dict[str, str]:
        with self.lock:
            rows = self.db.execute("SELECT key,value FROM session_facts WHERE tenant_id=? AND session_id=?",
                                   (tenant_id, session_id)).fetchall()
        return dict(rows)

    # ------------------------------------------------------------ items
    def upsert_item(self, tenant_id: str, item_id: str, kind: str, state: str, data: dict) -> None:
        with self.lock:
            self.db.execute("INSERT INTO items VALUES(?,?,?,?,?) ON CONFLICT(tenant_id,item_id) "
                            "DO UPDATE SET state=excluded.state, data=excluded.data",
                            (tenant_id, item_id, kind, state, canonical(data)))

    def get_item(self, tenant_id: str, item_id: str) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT item_id,kind,state,data FROM items WHERE tenant_id=? AND item_id=?",
                                (tenant_id, item_id)).fetchone()
        return None if not r else dict(item_id=r[0], kind=r[1], state=r[2], data=json.loads(r[3]))

    def set_item_state(self, tenant_id: str, item_id: str, state: str) -> bool:
        with self.lock:
            cur = self.db.execute("UPDATE items SET state=? WHERE tenant_id=? AND item_id=?",
                                  (state, tenant_id, item_id))
        return cur.rowcount > 0

    # ------------------------------------------------------------ audit hash chain
    def audit_append(self, tenant_id: str, event: str, detail: dict) -> dict:
        payload = canonical(detail)
        redacted, found = self._pii.redact(payload)
        if found:   # never persist raw PII; store the redacted form and note it
            payload = redacted
        with self.lock:
            last = self.db.execute("SELECT seq,hash FROM audit_log WHERE tenant_id=? ORDER BY seq DESC LIMIT 1",
                                   (tenant_id,)).fetchone()
            seq, prev = (last[0] + 1, last[1]) if last else (1, GENESIS)
            ts = time.time()
            body = canonical({"seq": seq, "ts": round(ts, 6), "event": event, "payload": payload})
            h = hashlib.sha256((prev + body).encode()).hexdigest()
            self.db.execute("INSERT INTO audit_log VALUES(?,?,?,?,?,?,?)",
                            (tenant_id, seq, round(ts, 6), event, payload, prev, h))
        return {"seq": seq, "hash": h}

    def verify_chain(self, tenant_id: str) -> tuple[bool, int, int | None]:
        """Returns (ok, n_records, first_bad_seq)."""
        with self.lock:
            rows = self.db.execute("SELECT seq,ts,event,payload,prev_hash,hash FROM audit_log "
                                   "WHERE tenant_id=? ORDER BY seq", (tenant_id,)).fetchall()
        prev = GENESIS
        for seq, ts, event, payload, ph, h in rows:
            body = canonical({"seq": seq, "ts": ts, "event": event, "payload": payload})
            if ph != prev or hashlib.sha256((prev + body).encode()).hexdigest() != h:
                return False, len(rows), seq
            prev = h
        return True, len(rows), None

    def audit_tail(self, tenant_id: str, n: int = 50) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT seq,ts,event,payload,hash FROM audit_log WHERE tenant_id=? "
                                   "ORDER BY seq DESC LIMIT ?", (tenant_id, n)).fetchall()
        return [dict(seq=r[0], ts=r[1], event=r[2], payload=r[3], hash=r[4]) for r in rows]

    # ------------------------------------------------------------ HITL
    def hitl_create(self, tenant_id: str, tool: str, args: dict, requested_by: str, idem: str | None) -> str:
        aid = str(uuid.uuid4())
        clean, _ = self._pii.redact(canonical(args))
        with self.lock:
            self.db.execute("INSERT INTO hitl VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (aid, tenant_id, tool, clean, requested_by, "PENDING_APPROVAL", None,
                             time.time(), None, time.time() + HITL_TTL_S, idem))
        self.audit_append(tenant_id, "hitl.pending", {"approval_id": aid, "tool": tool, "by": requested_by})
        return aid

    def hitl_list(self, tenant_id: str, status: str | None = "PENDING_APPROVAL") -> list[dict]:
        self.hitl_expire(tenant_id)
        q = "SELECT approval_id,tool,args,requested_by,status,decided_by,created FROM hitl WHERE tenant_id=?"
        a: list = [tenant_id]
        if status:
            q += " AND status=?"; a.append(status)
        with self.lock:
            rows = self.db.execute(q + " ORDER BY created DESC", a).fetchall()
        return [dict(approval_id=r[0], tool=r[1], args=json.loads(r[2]), requested_by=r[3], status=r[4],
                     decided_by=r[5], created=r[6]) for r in rows]

    def hitl_expire(self, tenant_id: str) -> int:
        with self.lock:
            cur = self.db.execute("UPDATE hitl SET status='EXPIRED' WHERE tenant_id=? AND status='PENDING_APPROVAL' "
                                  "AND expires<?", (tenant_id, time.time()))
        return cur.rowcount

    def hitl_decide(self, tenant_id: str, approval_id: str, decision: str, reviewer: str) -> dict | None:
        """Atomically decide. Requester may not approve their own request (separation of duties)."""
        with self.lock:
            r = self.db.execute("SELECT tool,args,requested_by,status FROM hitl WHERE tenant_id=? AND approval_id=?",
                                (tenant_id, approval_id)).fetchone()
            if not r or r[3] != "PENDING_APPROVAL":
                return None
            if r[2] == reviewer and decision == "approve":
                raise PermissionError("requester cannot approve own request")
            status = "APPROVED" if decision == "approve" else "REJECTED"
            self.db.execute("UPDATE hitl SET status=?, decided_by=?, decided=? WHERE tenant_id=? AND approval_id=?",
                            (status, reviewer, time.time(), tenant_id, approval_id))
        self.audit_append(tenant_id, f"hitl.{status.lower()}", {"approval_id": approval_id, "by": reviewer,
                                                               "tool": r[0]})
        return dict(approval_id=approval_id, tool=r[0], args=json.loads(r[1]), status=status)

    # ------------------------------------------------------------ idempotency
    def idem_get(self, tenant_id: str, idem: str) -> tuple[str, str] | None:
        with self.lock:
            return self.db.execute("SELECT args_hash,result FROM idempotency WHERE tenant_id=? AND idem=?",
                                   (tenant_id, idem)).fetchone()

    def idem_put(self, tenant_id: str, idem: str, args_hash: str, result: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO idempotency VALUES(?,?,?,?,?)",
                            (tenant_id, idem, args_hash, result, time.time()))
