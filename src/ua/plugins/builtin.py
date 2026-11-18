"""Built-in lifecycle plugins: pii_redactor, injection_shield, circuit_breaker, ragas_auditor, otel_exporter."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import defaultdict

from ..errors import Block
from ..textnorm import deobfuscate_symbols, leet_fold, normalize
from .spec import hookimpl

# =============================================================== PII redactor (PRE, fail-closed)
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-']+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
_EMAIL_WORDS = re.compile(
    r"\b([A-Za-z0-9._%+\-]{2,})\s+(?:at|@)\s+([A-Za-z0-9\-]{2,})\s+(?:dot|\.)\s+([A-Za-z]{2,}(?:\s+(?:dot|\.)\s+[A-Za-z]{2,})*)\b", re.I)
_EMAIL_AT_WORD = re.compile(r"\b([A-Za-z0-9._%+\-']{2,})\s+at\s+([A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,})\b", re.I)   # 'name at domain.tld'
_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}[\s.\-]?(?!00)\d{2}[\s.\-]?(?!0000)\d{4}\b")
_PHONE = re.compile(r"(?<![\w\-])(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]?\d{3}[\s.\-]?\d{4}(?![\w\-])")
_SPELLED = {w: str(i) for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])}
_SPELLED_RUN = re.compile(r"\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine)[\s\-]+){6,}"
                          r"(?:zero|one|two|three|four|five|six|seven|eight|nine)\b", re.I)
_CARD = re.compile(r"\b(?:\d[ \-]?){13,19}\b")
# entity ids that look like phone numbers must not be redacted
_KEEP = re.compile(r"\b(?:SKU|LOT|POL|SOP)[\s\-]?\d+\b", re.I)


def _luhn(num: str) -> bool:
    d = [int(c) for c in num if c.isdigit()][::-1]
    return len(d) >= 13 and (sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])) % 10 == 0


class PiiRedactor:
    """Regex + normalisation redactor. Replaces PII with typed tokens; never stores originals."""

    def __init__(self, **_):
        self.stats = defaultdict(int)

    def redact(self, text: str) -> tuple[str, dict[str, int]]:
        counts: dict[str, int] = defaultdict(int)
        t = deobfuscate_symbols(normalize(text))
        protected: list[str] = []

        def protect(m):
            protected.append(m.group(0))
            return f"\x00{len(protected) - 1}\x00"
        t = _KEEP.sub(protect, t)

        def sub(rx, tag, s, check=None):
            def r(m):
                if check and not check(m.group(0)):
                    return m.group(0)
                counts[tag] += 1
                return f"[{tag}]"
            return rx.sub(r, s)

        t = sub(_EMAIL, "EMAIL", t)
        t = _EMAIL_WORDS.sub(lambda m: (counts.__setitem__("EMAIL", counts["EMAIL"] + 1) or "[EMAIL]"), t)
        t = _EMAIL_AT_WORD.sub(lambda m: (counts.__setitem__("EMAIL", counts["EMAIL"] + 1) or "[EMAIL]"), t)
        t = sub(_CARD, "CARD", t, check=_luhn)
        t = sub(_SSN, "SSN", t)
        t = _SPELLED_RUN.sub(lambda m: (counts.__setitem__("PHONE", counts["PHONE"] + 1) or "[PHONE]"), t)
        t = sub(_PHONE, "PHONE", t)
        t = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], t)
        for k, v in counts.items():
            self.stats[k] += v
        return t, dict(counts)

    @hookimpl
    def pre_execution(self, ctx):
        ctx.query, ctx.redactions = self.redact(ctx.raw_query)
        if ctx.redactions:
            ctx.flags["pii_redacted"] = True


# =============================================================== Injection shield (PRE, fail-closed)
_RULES: list[tuple[str, float, str]] = [
    (r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions?|prompts?|rules?|messages?)", 1.0, "override"),
    (r"disregard (all |any |the )?(previous|prior|above|your) (instructions?|rules?|guidelines?)", 1.0, "override"),
    (r"forget (everything|all|your)\b", 0.9, "override"),
    (r"ignore (the |all |your |any |these |those )?(safety |security |content )?(guidelines?|rules?|restrictions?|instructions?|polic(?:y|ies)|filters?)", 0.95, "override"),
    (r"(output|print|show|repeat|display|list)\W+(me\W+)?(everything|all|the (text|words|content))\W+(above|before|prior)", 0.9, "extract"),
    (r"hidden (messages?|instructions?|text|prompt)", 0.7, "extract"),
    (r"(reveal|show|print|repeat|output|display|tell me|leak|dump)\W+(me\W+)?(your |the |all )?(system|hidden|initial|original|secret)\W*(prompt|instructions?|message|rules?)", 1.0, "extract"),
    (r"(what|which) (is|are|were) your (system |initial |original )?(prompt|instructions)", 0.8, "extract"),
    (r"repeat (the )?(text|words|everything) (above|before)", 0.8, "extract"),
    (r"\byou are now\b|\bnew (persona|role|identity)\b|\bact as (an? )?(unrestricted|unfiltered|jailbroken)", 0.9, "persona"),
    (r"\b(dan|do anything now|developer mode|jailbreak(ed)?|god mode)\b", 0.9, "jailbreak"),
    (r"without (any )?(restrictions?|filters?|limitations?|guardrails?|safety)", 0.7, "jailbreak"),
    (r"(pretend|imagine|roleplay|role-play) (that )?(you|there) (have|has|are|is|were) no (rules|restrictions|guidelines)", 0.9, "jailbreak"),
    (r"(bypass|disable|turn off|override) (the )?(safety|guardrails?|filters?|hitl|approval|moderation|circuit)", 0.95, "bypass"),
    (r"(skip|without) (the )?(human )?(approval|review|hitl)", 0.8, "bypass"),
    (r"<\s*/?\s*(system|assistant|instructions?)\s*>|\[\s*(system|inst)\s*\]|###\s*(system|instruction)", 0.9, "delimiter"),
    (r"(new|updated) (system )?instructions?\s*:", 0.8, "delimiter"),
    (r"(sudo|admin|root) (mode|override|access)", 0.7, "bypass"),
    # tool-argument overflow / SQLi / traversal fuzzing
    (r"('|\")\s*(or|and)\s+('?\d+'?\s*=\s*'?\d+|'[^']*'\s*=\s*'[^']*|true)", 1.0, "sqli"),
    (r";\s*(drop|delete|truncate|alter|update|insert)\s", 1.0, "sqli"),
    (r"\bunion\s+(all\s+)?select\b", 1.0, "sqli"),
    (r"(--|#|/\*)\s*$", 0.5, "sqli"),
    (r"\.\./|\.\.\\|/etc/passwd|%2e%2e%2f", 0.9, "traversal"),
    (r"(\$\(|`)[^)`]*(\)|`)|\|\s*(sh|bash|nc|curl|wget)\b", 0.9, "cmdi"),
    (r"(.)\1{200,}", 0.9, "overflow"),
]
_COMPILED = [(re.compile(p, re.I | re.S), w, tag) for p, w, tag in _RULES]
# crescendo / social-engineering cues (low weight individually, accumulate per session)
_SOFT = [
    (re.compile(p, re.I), w) for p, w in [
        (r"\bhypothetically\b|\bfor (a|my) (story|novel|research|school|class)\b", 0.25),
        (r"\b(just|only) (this once|between us)\b|\bdon'?t tell\b|\boff the record\b", 0.3),
        (r"\b(as|since) (you|we) (already )?(agreed|said|discussed)\b|\bearlier you (said|agreed|promised)\b", 0.3),
        (r"\b(i am|i'm) (the |an? )?(admin|administrator|developer|ceo|cto|auditor|security (team|officer)|your creator)\b", 0.35),
        (r"\b(urgent|emergency|critical)\b.{0,40}\b(override|bypass|approve|skip|immediately)\b", 0.35),
        (r"\b(internal|confidential|secret)\b.{0,30}\b(prompt|config(uration)?|credentials?|keys?|tokens?|passwords?)\b", 0.35),
        (r"\b(step by step|detailed)\b.{0,40}\b(how (to|do)).{0,30}\b(hack|exploit|bypass|crack)\b", 0.4),
        (r"\b(api[_ ]?key|password|secret|credential)s?\b", 0.2),
    ]
]


class InjectionShield:
    """Weighted rule classifier over normalised text + per-session crescendo accumulation."""

    def __init__(self, block_threshold: float = 0.8, session_threshold: float = 0.9, **_):
        self.block_threshold = block_threshold
        self.session_threshold = session_threshold
        self._session: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self.stats = defaultdict(int)

    def views(self, text: str) -> list[str]:
        n = normalize(text)
        return [n, deobfuscate_symbols(n), leet_fold(n), re.sub(r"[\s_\-.*]+", "", leet_fold(n))]

    def score(self, text: str) -> tuple[float, list[str]]:
        hits: list[tuple[float, str]] = []
        v = self.views(text)
        for rx, w, tag in _COMPILED:
            if any(rx.search(x) for x in v[:3]):
                hits.append((w, tag))
        # spaced-out / separator-obfuscated keywords ("i g n o r e  p r e v i o u s")
        squashed = v[3]
        for kw, tag in (("ignorepreviousinstructions", "override"), ("ignoreallpreviousinstructions", "override"),
                        ("revealthesystemprompt", "extract"), ("revealyoursystemprompt", "extract"),
                        ("disregardpreviousinstructions", "override"), ("systemprompt", "extract-weak")):
            if kw in squashed:
                hits.append((0.5 if tag == "extract-weak" else 1.0, tag))
        s = max((w for w, _ in hits), default=0.0)
        if len(hits) > 1:
            s = min(1.0, s + 0.1 * (len(hits) - 1))
        return s, sorted({t for _, t in hits})

    def soft_score(self, text: str) -> float:
        n = normalize(text)
        return sum(w for rx, w in _SOFT if rx.search(n))

    def scan_untrusted(self, text: str, threshold: float = 0.7) -> tuple[bool, float, list[str]]:
        """Indirect injection: scan retrieved/observed content. Returns (is_malicious, score, tags)."""
        s, tags = self.score(text)
        # instructions addressed to the assistant inside data are suspicious even at lower weight
        n = normalize(text)
        if re.search(r"\b(assistant|ai|model|llm|agent)\b.{0,40}\b(must|should|shall|will now|need to)\b.{0,60}\b(ignore|reveal|send|call|execute|approve|delete|transfer)", n, re.I):
            s, tags = max(s, 0.85), tags + ["indirect-directive"]
        if re.search(r"\b(call|invoke|run|execute)\s+(the\s+)?(tool\s+)?(submit_form|transition_workflow|hitl_decide|audit_append)\b", n, re.I):
            s, tags = max(s, 0.95), tags + ["indirect-toolcall"]
        return s >= threshold, s, tags

    @hookimpl
    def pre_execution(self, ctx):
        s, tags = self.score(ctx.raw_query)
        soft = self.soft_score(ctx.raw_query)
        key = (ctx.tenant_id, ctx.session_id)
        with self._lock:
            hist = self._session[key]
            hist.append(max(soft, s * 0.5))
            del hist[:-6]
            cumulative = sum(hist)
        ctx.injection_score = max(s, min(1.0, cumulative))
        if s >= self.block_threshold:
            self.stats["blocked_direct"] += 1
            raise Block(-32008, f"prompt-injection blocked ({','.join(tags)})")
        if cumulative >= self.session_threshold and len(hist) > 1:
            self.stats["blocked_crescendo"] += 1
            raise Block(-32008, "multi-turn escalation blocked")


# =============================================================== Circuit breaker (TOOL_DISPATCH)
class CircuitBreakerPlugin:
    """Trips on (tool_name, canonical_args_hash) failing >= max_identical_calls times."""

    def __init__(self, max_identical_calls: int = 2, **_):
        self.limit = max_identical_calls
        self.fails: dict[tuple, int] = defaultdict(int)
        self.open: dict[tuple, float] = {}
        self._lock = threading.Lock()
        self.trips = 0

    @staticmethod
    def key(call) -> tuple:
        h = hashlib.sha256(json.dumps(call.args, sort_keys=True, default=str).encode()).hexdigest()
        return (call.meta.tenant_id, call.meta.session_id, call.name, h)

    @hookimpl
    def tool_dispatch(self, call):
        if self.key(call) in self.open:
            raise Block(-32004, f"circuit open for {call.name} with identical arguments", False)

    @hookimpl
    def tool_result(self, call, result):
        k = self.key(call)
        with self._lock:
            if result.ok:
                self.fails.pop(k, None)
                return
            self.fails[k] += 1
            if self.fails[k] >= self.limit and k not in self.open:
                self.open[k] = time.time()
                self.trips += 1

    def reset(self) -> None:
        with self._lock:
            self.fails.clear(); self.open.clear()

    def snapshot(self) -> dict:
        return {"open": [{"tool": k[2], "session": k[1]} for k in self.open], "trips": self.trips}


# =============================================================== RAGAS auditor (POST)
class RagasAuditor:
    """Online gate = lexical claim-support (NLI proxy). Full RAGAS runs async/offline (ADR-006)."""

    def __init__(self, min_faithfulness: float = 0.92, **_):
        self.min_faithfulness = min_faithfulness
        self.scores: list[float] = []

    @hookimpl
    def post_execution(self, ctx, draft):
        from ..verify import claim_support  # local import to avoid cycles
        if draft.abstained or not draft.chunks:
            draft.faithfulness = 1.0
            return
        faith, unsupported = claim_support(draft.text, [c["text"] for c in draft.chunks])
        draft.faithfulness = faith
        draft.unsupported = unsupported
        self.scores.append(faith)
        del self.scores[:-5000]
        if faith < self.min_faithfulness:
            draft.veto = f"faithfulness {faith:.2f} < {self.min_faithfulness}"


# =============================================================== OTel exporter (ALL_SPANS)
class OtelExporter:
    """Bridges recorded spans to the OTLP collector with a non-blocking batch processor."""

    def __init__(self, endpoint: str = "http://localhost:4317", enabled: bool = True, **_):
        self.endpoint = endpoint
        self.exported = 0
        self._tracer = None
        if not enabled:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            prov = TracerProvider(resource=Resource.create({"service.name": "unreliable-assistant"}))
            prov.add_span_processor(BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint, insecure=True, timeout=1),
                max_queue_size=4096, schedule_delay_millis=2000, export_timeout_millis=1000))
            self._tracer = prov.get_tracer("ua")
        except Exception:
            self._tracer = None

    def export(self, span) -> None:
        if not self._tracer:
            return
        s = self._tracer.start_span(span.name, start_time=span.start_ns)
        for k, v in span.attrs.items():
            if isinstance(v, (str, int, float, bool)):
                s.set_attribute(k, v)
        s.end(end_time=span.end_ns)
        self.exported += 1
