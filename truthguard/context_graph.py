"""The 3-plane context graph — built on dg-core's DecisionMemory.

  x  CHAT SPINE      every answered question is a node, linked `follows`
                     chronologically (the conversation timeline)
  y+ KNOWLEDGE PLANE spine node --grounds--> provenance chunks it cited
  y- CODE PLANE      spine node --references--> code chunks it used

Spine nodes ARE decision-audit nodes (confidence / outcome / decay / supersede
come free from dg-core). Follow-up questions can retrieve context by graph
neighborhood instead of replaying chat history.

  python -m truthguard.context_graph        # print the current graph
"""
import os
import pickle

import networkx as nx

from . import config

_PATH = lambda d: os.path.join(d or config.STORAGE_DIR, "context_graph.pkl")


class ContextGraph:
    def __init__(self, storage_dir: str = None):
        self.storage_dir = storage_dir or config.STORAGE_DIR
        self.g = nx.DiGraph()
        self._last_spine = None
        self.load()

    # ── record one turn (called by the controller after every response) ─────
    def record_turn(self, question: str, response: dict, chunks_used: list,
                    session: str = "live", agent_id: str = None,
                    namespace: str = None, episode_id: str = None) -> str:
        """Record one turn on the spine.

        agent_id / namespace exist so a shared graph can say *who* concluded
        something and *which tenant* it belongs to. Both default from the
        environment, so a single-agent setup behaves exactly as before while a
        fleet gets attribution without any caller changing.
        """
        import os
        agent_id = agent_id or os.getenv("TG_AGENT_ID", "default")
        namespace = namespace or os.getenv("TG_NAMESPACE", "default")

        nid = f"t{self.g.graph.get('n_turns', 0) + 1}"
        self.g.graph["n_turns"] = self.g.graph.get("n_turns", 0) + 1
        self.g.add_node(nid, plane="spine", question=question[:200],
                        kind=response["kind"],
                        session=session,
                        agent_id=agent_id,
                        namespace=namespace,
                        text=(response.get("text") or "")[:300],
                        confidence=response.get("confidence"),
                        band=response.get("band"))
        # link the conclusion to the run that produced it (z-plane)
        if episode_id and self.g.has_node(episode_id):
            self.g.add_edge(episode_id, nid, relation="produced")
        if self._last_spine:
            self.g.add_edge(self._last_spine, nid, relation="follows")
        self._last_spine = nid

        for c in chunks_used:
            plane = "code" if c.get("content_type") == "code" else "knowledge"
            cid = f"{plane[0]}:{c['id']}"
            if not self.g.has_node(cid):
                self.g.add_node(cid, plane=plane,
                                source=f"{c['source_file']} p{c['page']}",
                                extraction=c.get("extraction"))
            self.g.add_edge(nid, cid,
                            relation="references" if plane == "code" else "grounds")
            # y− Level 2: resolve code identifiers to REAL symbols (GitNexus)
            if plane == "code":
                try:
                    from . import code_link
                    for ident in code_link.extract_identifiers(c["text"]):
                        sym = code_link.resolve_symbol(ident)
                        if sym:
                            sid = f"sym:{sym['symbol']}"
                            if not self.g.has_node(sid):
                                self.g.add_node(sid, plane="code_symbol",
                                                source=sym["file"],
                                                extraction="gitnexus")
                            self.g.add_edge(nid, sid, relation="references_symbol")
                except Exception:
                    pass    # symbol linking must never break recording
        self.save()
        return nid

    # ── neighborhood context pack (the "context window is a neighborhood") ──
    def neighborhood(self, spine_id: str = None, hops_back: int = 2) -> dict:
        sid = spine_id or self._last_spine
        if sid is None or not self.g.has_node(sid):
            return {"spine": [], "knowledge": [], "code": []}
        spine, cur = [sid], sid
        for _ in range(hops_back):
            prevs = [u for u, _, d in self.g.in_edges(cur, data=True)
                     if d.get("relation") == "follows"]
            if not prevs:
                break
            cur = prevs[0]
            spine.append(cur)
        knowledge, code, entities = set(), set(), set()
        for s in spine:
            for _, v, d in self.g.out_edges(s, data=True):
                rel = d.get("relation")
                nd = self.g.nodes[v]
                name = nd.get("source") or nd.get("label") or str(v)
                plane = nd.get("plane", "")
                if rel == "grounds" or plane == "knowledge":
                    knowledge.add(name)
                elif plane == "entity":
                    entities.add(name)
                elif rel in ("references", "references_symbol", "edited") \
                        or plane in ("code", "code_symbol", "code_file"):
                    code.add(name)
        return {"spine": [dict(self.g.nodes[s], id=s) for s in spine],
                "knowledge": sorted(knowledge), "code": sorted(code),
                "entities": sorted(entities)}

    def episode_entity_index(self) -> dict:
        """Bidirectional episode<->entity index (Zep): {episode->[entities],
        entity->[episodes]} built from grounds/references edges. O(1) citation
        traceability — any fact traces back to the source turn/chunk it came from,
        and any turn lists what it grounded on."""
        ep2ent, ent2ep = {}, {}
        for u, v, e in self.g.edges(data=True):
            if e.get("relation") in ("grounds", "references", "references_symbol"):
                pu = self.g.nodes[u].get("plane")
                if pu in ("spine", "chat"):
                    ep2ent.setdefault(u, []).append(v)
                    ent2ep.setdefault(v, []).append(u)
        return {"episode_to_entity": ep2ent, "entity_to_episode": ent2ep}

    def hierarchy(self, node: str) -> dict:
        """Explicit three-tier lineage (Zep): episode (raw turn/chunk, immutable)
        -> entity (derived triple/symbol) -> community (clustered summary)."""
        tier = {"code": "entity", "code_symbol": "entity", "entity": "entity",
                "knowledge": "episode", "spine": "episode", "chat": "episode",
                "x_community": "community", "y_community": "community",
                "doc_community": "community", "code_community": "community"}
        out = {"node": node, "tier": tier.get(self.g.nodes[node].get("plane"), "?"),
               "parents": [], "children": []}
        for _, v, e in self.g.out_edges(node, data=True):
            if e.get("relation") == "member_of":
                out["parents"].append(v)
        for u, _, e in self.g.in_edges(node, data=True):
            if e.get("relation") == "member_of":
                out["children"].append(u)
        return out

    def save(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        with open(_PATH(self.storage_dir), "wb") as f:
            pickle.dump({"g": self.g, "last": self._last_spine}, f)

    def load(self):
        p = _PATH(self.storage_dir)
        if os.path.exists(p):
            with open(p, "rb") as f:
                blob = pickle.load(f)
            self.g, self._last_spine = blob["g"], blob["last"]

    def dump(self) -> str:
        lines = [f"3-PLANE CONTEXT GRAPH — {self.g.graph.get('n_turns', 0)} turns"]
        spine = [n for n, d in self.g.nodes(data=True) if d["plane"] == "spine"]
        for s in sorted(spine):
            d = self.g.nodes[s]
            lines.append(f"\n(x) {s} [{d['kind']}"
                         + (f" conf={d['confidence']}" if d.get("confidence") is not None else "")
                         + f"] {d['question'][:70]}")
            for _, v, e in self.g.out_edges(s, data=True):
                if e["relation"] not in ("grounds", "references", "references_symbol"):
                    continue
                nd = self.g.nodes[v]
                arrow = {"grounds": "y+ grounds   ", "references": "y- references",
                         "references_symbol": "y- SYMBOL    "}[e["relation"]]
                lines.append(f"    {arrow} -> {nd['source']} ({nd.get('extraction','')})")
        return "\n".join(lines)


if __name__ == "__main__":
    print(ContextGraph().dump())
