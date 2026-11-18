import hashlib
import json

import pytest
from conftest import meta
from ua.cache import MemoryBackend, SemanticCache, tenant_key, user_key
from ua.config import SETTINGS
from ua.harness import det_uuid4, repair_id
from ua.store import StateStore
from ua.verify import claim_support


def test_repair_examples_from_spec():
    assert repair_id("SKU99", "SKU") == "SKU-099"
    assert repair_id("lot 44", "LOT") == "LOT-00044"
    assert repair_id("SKU-1234", "SKU") == "SKU-1234"      # cannot be repaired -> unchanged -> validation fails
    assert repair_id("hello", None) == "hello"


def test_det_uuid4_is_valid_v4_and_deterministic():
    import re
    from ua.contracts import UUID4_RE
    k = det_uuid4("t", "s", "tool", "{}")
    assert re.match(UUID4_RE, k) and k == det_uuid4("t", "s", "tool", "{}") and k != det_uuid4("t", "s", "tool", "{1}")


def test_audit_hash_chain_detects_tampering():
    st = StateStore(":memory:")
    for i in range(5):
        st.audit_append("t1", "e", {"i": i})
    assert st.verify_chain("t1") == (True, 5, None)
    st.db.execute("UPDATE audit_log SET payload='{\"i\":99}' WHERE tenant_id='t1' AND seq=3")
    ok, n, bad = st.verify_chain("t1")
    assert not ok and bad == 3
    assert st.verify_chain("t2") == (True, 0, None)         # chains are per tenant


def test_audit_never_stores_raw_pii():
    st = StateStore(":memory:")
    st.audit_append("t", "e", {"note": "contact bob@example.com"})
    assert "bob@example.com" not in st.audit_tail("t")[0]["payload"]
    st.put_fact("t", "s", "k", "ssn 123-45-6789")
    assert "123-45-6789" not in st.get_facts("t", "s")["k"]


def test_tenant_isolation_in_state_store():
    st = StateStore(":memory:")
    st.upsert_item("A", "SKU-001", "sku", "active", {})
    assert st.get_item("A", "SKU-001") and st.get_item("B", "SKU-001") is None
    st.put_fact("A", "s", "k", "v")
    assert st.get_facts("B", "s") == {}


def test_hitl_requires_approval_and_separation_of_duties():
    st = StateStore(":memory:")
    aid = st.hitl_create("A", "submit_form", {"entity_id": "LOT-88001"}, "alice", None)
    with pytest.raises(PermissionError):
        st.hitl_decide("A", aid, "approve", "alice")          # requester can't self-approve
    assert st.hitl_decide("B", aid, "approve", "bob") is None  # other tenant can't decide
    assert st.hitl_decide("A", aid, "approve", "bob")["status"] == "APPROVED"
    assert st.hitl_decide("A", aid, "approve", "carol") is None  # already decided


def test_cache_keys_isolated_by_tenant_and_user():
    assert tenant_key("A", "q") != tenant_key("B", "q")
    assert user_key("A", "u1", "q") != user_key("A", "u2", "q")
    assert user_key("A", "u1", "q").startswith("{A}:")       # Redis-cluster hash tag co-locates a tenant
    c = SemanticCache(MemoryBackend())
    c.store("A", "u1", "what is x", {"answer": "secret-A"}, "user")
    assert c.lookup("A", "u1", "what is x")["answer"] == "secret-A"
    assert c.lookup("A", "u2", "what is x") is None           # other user
    assert c.lookup("B", "u1", "what is x") is None           # other tenant
    c.store("A", "u1", "shared q", {"answer": "pub"}, "tenant")
    assert c.lookup("A", "u2", "shared q")["answer"] == "pub" and c.lookup("B", "u2", "shared q") is None


def test_cache_ttl_is_jittered():
    c = SemanticCache(MemoryBackend())
    import time
    exp = []
    for i in range(40):
        c.store("A", "u", f"q{i}", {"a": 1}, "user")
    exp = [v[0] - time.time() for v in c.backend.kv.values()]
    assert 255 <= min(exp) and max(exp) <= 346 and max(exp) - min(exp) > 20


def test_claim_support_catches_fabricated_numbers():
    ev = ["Unit price $463.04; MOQ 100 units. Supplied by Atlas Biochem."]
    assert claim_support("Unit price $463.04.", ev)[0] == 1.0
    f, bad = claim_support("Unit price $999.99.", ev)
    assert f == 0.0 and bad
    f, bad = claim_support("It ships from SKU-999.", ev)
    assert f == 0.0


def test_goal_anchor_is_immutable_and_first():
    from ua.loop import GoalAnchorBlock
    a = GoalAnchorBlock("price of SKU-007", ("SKU-007",))
    with pytest.raises(Exception):
        a.original_intent = "x"
    assert a.render().startswith("[GOAL] price of SKU-007")


def test_bm25_matches_rank_bm25_oracle():
    from rank_bm25 import BM25Okapi
    from ua.rag.bm25 import BM25Index, tokenize
    docs = ["unit price of the flask is 12", "shelf life of reagent is 24 months", "supplier of the film is acme", "flask storage temperature range"]
    mine = BM25Index(docs).scores("flask price", 1.5, 0.75)
    ref = BM25Okapi([tokenize(d) for d in docs], k1=1.5, b=0.75).get_scores(tokenize("flask price"))
    assert pytest.approx(list(mine), rel=1e-6, abs=1e-9) == list(ref)
