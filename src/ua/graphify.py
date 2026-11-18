"""Graphify: unified structural graph of Python modules/Pydantic schemas + Obsidian policy notes (Gate 17).

CLI:   python -m ua.graphify build ./src ./knowledge_vault --output graph.json
Runtime: Graph.load('graph.json').subgraph(query, depth<=2, max_tokens) -> minimal connected subgraph text.

Traversal follows OUT-edges only (note -> [[linked notes]]). Following in-edges would fan out through shared
policies (one policy is linked from many products) and defeat the token savings.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

from .rag.index import parse_note
from .textnorm import approx_tokens

ID_RX = re.compile(r"\b(?:SKU-\d{3}|LOT-\d{5}|POL-\d{2}|SOP-\d{2})\b", re.I)


def _first_sentence_body(body: str, limit: int = 600) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^#.*\n", "", body, flags=re.M)).strip()[:limit]


def build(src_dirs: list[str], vault_dir: str, output: str) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    # ---- obsidian notes
    for f in sorted(Path(vault_dir).glob("*.md")):
        n = parse_note(f)
        nodes[f"note:{n['id']}"] = dict(id=f"note:{n['id']}", kind="note", ref=n["id"], title=n["title"], domain=n["domain"],
                                        risk_tier=n["risk_tier"], version=n["version"], text=_first_sentence_body(n["body"]))
    for nid, nd in list(nodes.items()):
        if nd["kind"] != "note":
            continue
        for l in parse_note(Path(vault_dir) / f"{nd['ref']}.md")["links"]:
            if f"note:{l}" in nodes:
                edges.append(dict(src=nid, dst=f"note:{l}", rel="wikilink"))
    # ---- python modules / classes / pydantic schemas
    for sd in src_dirs:
        for f in sorted(Path(sd).rglob("*.py")):
            mod = ".".join(f.relative_to(sd).with_suffix("").parts)
            tree = ast.parse(f.read_text())
            mid = f"py:{mod}"
            nodes[mid] = dict(id=mid, kind="module", ref=mod, title=mod, text=(ast.get_docstring(tree) or "").split("\n")[0])
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    bases = [ast.unparse(b) for b in node.bases]
                    cid = f"py:{mod}.{node.name}"
                    nodes[cid] = dict(id=cid, kind="schema" if "BaseModel" in bases else "class", ref=node.name,
                                      title=node.name, text=(ast.get_docstring(node) or "").split("\n")[0])
                    edges.append(dict(src=mid, dst=cid, rel="defines"))
                elif isinstance(node, ast.ImportFrom) and node.module:
                    tgt = node.module if node.level == 0 else ".".join(mod.split(".")[:max(0, len(mod.split(".")) - node.level)] + node.module.split("."))
                    edges.append(dict(src=mid, dst=f"py:{tgt}", rel="imports"))
    valid = set(nodes)
    edges = [e for e in edges if e["dst"] in valid]
    g = dict(nodes=list(nodes.values()), edges=edges, built_from=[*src_dirs, vault_dir])
    Path(output).write_text(json.dumps(g, indent=1))
    return g


class Graph:
    def __init__(self, g: dict):
        self.nodes = {n["id"]: n for n in g["nodes"]}
        self.out: dict[str, list[str]] = {}
        for e in g["edges"]:
            if e["rel"] == "wikilink":
                self.out.setdefault(e["src"], []).append(e["dst"])
        self.by_ref = {n["ref"].upper(): n["id"] for n in g["nodes"] if n["kind"] == "note"}
        self.names = {re.sub(r"^(?:SKU|LOT|POL|SOP)-\d+\s*", "", n["title"]).lower(): n["id"]
                      for n in g["nodes"] if n["kind"] == "note" and n["domain"] in ("catalog", "sop")}

    @classmethod
    def load(cls, path: str) -> "Graph":
        return cls(json.loads(Path(path).read_text()))

    def resolve_anchors(self, query: str, hint: list[str] | None = None) -> list[str]:
        anchors = [self.by_ref[m.upper()] for m in ID_RX.findall(query) if m.upper() in self.by_ref]
        ql = query.lower()
        anchors += [nid for nm, nid in self.names.items() if len(nm) >= 8 and nm in ql]
        if not anchors and hint:
            anchors = [self.by_ref[h.upper()] for h in hint if h.upper() in self.by_ref][:1]
        return list(dict.fromkeys(anchors))

    def subgraph(self, query: str, anchor: str | None = None, depth: int = 2, max_tokens: int = 320,
                 hint: list[str] | None = None) -> dict:
        anchors = [self.by_ref[anchor.upper()]] if anchor and anchor.upper() in self.by_ref else self.resolve_anchors(query, hint)
        depth = min(depth, 2)
        seen: dict[str, int] = {a: 0 for a in anchors}
        frontier = list(anchors)
        for d in range(1, depth + 1):
            nxt = []
            for n in frontier:
                for m in self.out.get(n, []):
                    if m not in seen:
                        seen[m] = d
                        nxt.append(m)
            frontier = nxt
        lines, used, kept = [], 0, []
        for nid, d in sorted(seen.items(), key=lambda x: x[1]):         # nearest first; prune the farthest to fit budget
            n = self.nodes[nid]
            txt = n["text"] if d <= 1 else n["text"][:160]
            line = f"[{n['ref']}] {txt}"
            t = approx_tokens(line)
            if used + t > max_tokens and kept:
                break
            lines.append(line)
            kept.append(n["ref"])
            used += t
        edges = [(self.nodes[a]["ref"], self.nodes[b]["ref"]) for a in seen for b in self.out.get(a, []) if b in seen]
        return dict(anchors=[self.nodes[a]["ref"] for a in anchors], nodes=kept, edges=edges, text="\n".join(lines),
                    tokens=used, depth_reached=max(seen.values(), default=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="graphify")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("src")
    b.add_argument("vault")
    b.add_argument("--output", default="graph.json")
    a = ap.parse_args()
    g = build([a.src], a.vault, a.output)
    print(f"graph.json: {len(g['nodes'])} nodes, {len(g['edges'])} edges")
