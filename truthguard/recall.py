"""x-plane recall — DG's DecisionMemory.query recipe applied to the chat spine.

Embed the question -> cosine against every spine turn's question (and every
topic-community summary) -> top matches -> neighborhood walk (the chunks/code
each matched turn grounded on). This is how a NEW chat retrieves context from
OLD chats: O(neighborhood), not O(history).

Run:  python -m truthguard.recall "what did we decide about the travel limit?"
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from . import config
from .context_graph import ContextGraph

_EMBED = None


def _embedder():
    global _EMBED
    if _EMBED is None:
        from sentence_transformers import SentenceTransformer
        _EMBED = SentenceTransformer(config.EMBED_MODEL)
    return _EMBED



def _visible(d: dict, namespace: str) -> bool:
    """Is this node readable from `namespace`?

    Namespaces were enforced on writes but not on reads, so an agent could not
    write into another tenant's space yet could recall its conversations. Nodes
    carrying no namespace are shared substrate (documents, code, communities)
    and stay visible; nodes that declare one are scoped to it.
    """
    ns = d.get("namespace")
    return ns is None or ns == namespace


def _ns(explicit: str = None) -> str:
    import os
    return explicit or os.getenv("TG_NAMESPACE", "default")


def recall(question: str, top_k: int = 3, storage_dir: str = None,
           namespace: str = None) -> dict:
    """DG recipe: similarity over past turns + communities, then neighborhood."""
    cg = ContextGraph(storage_dir)
    g = cg.g
    namespace = _ns(namespace)
    turns = [(n, d) for n, d in g.nodes(data=True)
             if d.get("plane") == "spine" and _visible(d, namespace)]
    comms = [(n, d) for n, d in g.nodes(data=True) if d.get("plane") == "x_community"]
    if not turns:
        return {"matches": [], "communities": [], "note": "spine is empty"}

    em = _embedder()
    q = em.encode([question], normalize_embeddings=True)[0]

    # 1) beam over topic-community summaries (DG's community layer)
    comm_hits = []
    if comms:
        cvecs = em.encode([d.get("summary", "") or d.get("source", "")
                           for _, d in comms], normalize_embeddings=True)
        sims = np.asarray(cvecs) @ q
        for i in np.argsort(sims)[::-1][:2]:
            n, d = comms[i]
            comm_hits.append({"community": n, "summary": d.get("summary", ""),
                              "similarity": round(float(sims[i]), 3),
                              "n_turns": g.in_degree(n)})

    # 2) DG DecisionMemory.query, complete: lazy decay -> active only ->
    #    score = sim x confidence -> access bookkeeping on retrieval
    from datetime import datetime, timedelta
    today = datetime.now().date().isoformat()
    if g.graph.get("last_decay_run") != today:            # decay_confidence, lazy 1x/day
        cutoff = datetime.now() - timedelta(days=90)
        for n, d in turns:
            ref = d.get("last_accessed") or ""
            try:
                last = datetime.fromisoformat(ref) if ref else datetime.now()
            except ValueError:
                last = datetime.now()
            if last < cutoff and d.get("is_active", True):
                d["confidence"] = max(0.0, round((d.get("confidence") or 1.0) - 0.1, 3))
                if d["confidence"] < 0.2:
                    d["is_active"] = False
        g.graph["last_decay_run"] = today

    active = [(n, d) for n, d in turns if d.get("is_active", True)]
    tvecs = em.encode([d["question"] for _, d in active], normalize_embeddings=True)
    sims = np.asarray(tvecs) @ q
    scored = sorted(
        ((float(sims[i]) * float(active[i][1].get("confidence") or 1.0), float(sims[i]), i)
         for i in range(len(active))), reverse=True)
    matches = []
    for score, sim, i in scored[:top_k]:
        if sim < 0.35:                 # DG's DECISION_SIMILARITY_THRESHOLD spirit
            continue
        n, d = active[i]
        d["access_count"] = d.get("access_count", 0) + 1   # access_decision()
        d["last_accessed"] = datetime.now().isoformat()
        nb = cg.neighborhood(spine_id=n, hops_back=1)
        matches.append({
            "turn": n,
            "similarity": round(sim, 3),
            "score": round(score, 3),
            "confidence": (d.get("confidence") or 1.0),
            "access_count": d["access_count"],
            "question": d["question"],
            "kind": d.get("kind"),
            "session": d.get("session", "live"),
            "status": d.get("status", "current"),
            "superseded_by": d.get("superseded_by"),
            "answer_preview": (d.get("text") or "")[:200],
            "grounded_on": nb["knowledge"][:5],
            "code_touched": nb["code"][:5],
            "entities": nb.get("entities", [])[:5],
        })
    cg.save()
    return {"matches": matches, "communities": comm_hits,
            **recall_planes(question, q, g, em)}


def recall_planes(question: str, qvec, g, em, top_k: int = 4,
                  namespace: str = None) -> dict:
    """Cross-plane recall: y+ entities/communities, y- code nodes, and the
    actual document passages (chunk store) — one query, all three planes."""
    out = {"entities": [], "code": [], "doc_passages": [], "code_passages": []}

    # actual code BODIES from the codebase digest (AST-extracted source text)
    try:
        from . import code_digest
        out["code_passages"] = code_digest.search(qvec, k=top_k)
    except Exception:
        pass

    def _search(nodes):
        if not nodes:
            return []
        texts = [(d.get("label") or d.get("summary") or d.get("source") or str(n))[:300]
                 for n, d in nodes]
        sims = np.asarray(em.encode(texts, normalize_embeddings=True)) @ qvec
        hits, seen = [], set()
        for i in np.argsort(sims)[::-1]:
            if sims[i] < 0.30 or len(hits) >= top_k:
                break
            key = texts[i]
            if key in seen:            # duplicate nodes with the same label
                continue
            seen.add(key)
            hits.append((nodes[i], float(sims[i])))
        return hits

    ents = [(n, d) for n, d in g.nodes(data=True)
            if d.get("plane") in ("entity", "doc_community", "y_community", "knowledge")
            and _visible(d, _ns(namespace))]
    for (n, d), s in _search(ents):
        out["entities"].append({"node": n, "similarity": round(s, 3),
                                "label": d.get("label") or d.get("summary", ""),
                                "source": d.get("source", "")})

    code = [(n, d) for n, d in g.nodes(data=True)
            if d.get("plane") in ("code", "code_symbol", "code_file")
            and _visible(d, _ns(namespace))]
    for (n, d), s in _search(code):
        out["code"].append({"node": n, "similarity": round(s, 3),
                            "label": d.get("label") or str(n),
                            "file": d.get("source") or d.get("file", "")})

    # actual document CONTENT from the chunk store (what `ask` retrieves over)
    try:
        from .chunk_store import ChunkStore
        store = ChunkStore()
        for cid, s in store.vector_search(question, k=top_k):
            if s < 0.30:
                continue
            c = store.by_id.get(cid, {})
            out["doc_passages"].append({
                "chunk": cid, "similarity": round(float(s), 3),
                "source": c.get("source_file", ""), "heading": f"p.{c.get('page', '?')}",
                "text": (c.get("text") or "")[:400]})
    except Exception:
        pass
    return out


def format_recall(r: dict) -> str:
    if not any(r.get(k) for k in ("matches", "communities", "entities", "code",
                                  "doc_passages", "code_passages")):
        return "Nothing relevant found on any plane."
    out = []
    for p in r.get("code_passages", []):
        out.append(f"[code {p['file']}:{p['lineno']} sim={p['similarity']}] {p['kind']} {p['symbol']}\n"
                   + "\n".join("  | " + ln for ln in p["text"].splitlines()[:12]))
    for p in r.get("doc_passages", []):
        out.append(f"[document {p['source']} sim={p['similarity']}] {p['heading']}\n  \"{p['text'][:250]}\"")
    for e in r.get("entities", []):
        out.append(f"[y+ entity sim={e['similarity']}] {e['label'][:90]}" +
                   (f" (from {e['source']})" if e.get("source") else ""))
    for c in r.get("code", []):
        out.append(f"[y- code sim={c['similarity']}] {c['label'][:90]}" +
                   (f" ({c['file']})" if c.get("file") else ""))
    for c in r["communities"]:
        out.append(f"[topic sim={c['similarity']}] {c['summary'][:80]} ({c['n_turns']} turns)")
    for m in r["matches"]:
        tag = f" SUPERSEDED by {m['superseded_by']}" if m.get("status") == "superseded" else ""
        out.append(f"\n[{m['turn']} sim={m['similarity']} {m['kind']} @{m.get('session','live')}{tag}] Q: {m['question'][:90]}")
        if m["answer_preview"]:
            out.append(f"  A: {m['answer_preview'][:140]}")
        if m.get("entities"):
            out.append(f"  entities: {', '.join(m['entities'])}")
        if m["grounded_on"]:
            out.append(f"  grounded on: {', '.join(m['grounded_on'])}")
        if m["code_touched"]:
            out.append(f"  code: {', '.join(m['code_touched'])}")
    return "\n".join(out)


def _fit_budget(parts: list, budget_tokens: int) -> str:
    """Hard token budget with priority ranking (top memory systems cap at 2-4k;
    unbounded context measurably degrades answer quality). Sections are kept in
    evidence-first order and the first section to overflow is truncated; lower
    priority sections are dropped entirely rather than half-shown."""
    PRIORITY = ["## Document evidence", "## Code", "## Known facts",
                "## Relevant past conversation", "## Topics"]
    header, sections, cur = parts[0], {}, None
    for line in parts[1:]:
        key = next((p for p in PRIORITY if line.strip().startswith(p)), None)
        if key:
            cur = key
            sections[cur] = [line]
        elif cur:
            sections[cur].append(line)
    budget_chars = budget_tokens * 4                 # ~4 chars/token
    out, used = [header], len(header)
    for key in PRIORITY:
        block = sections.get(key)
        if not block:
            continue
        text = "\n".join(block)
        if used + len(text) <= budget_chars:
            out.append(text)
            used += len(text)
        else:
            room = budget_chars - used
            if room > 300:                            # only if usefully large
                out.append(text[:room].rsplit("\n", 1)[0] + "\n  …[truncated to budget]")
            break
    return "\n".join(out)


def get_context(question: str, storage_dir: str = None, namespace: str = None,
                budget_tokens: int = None) -> str:
    """Context ROUTER: one call -> a ready-to-inject context block with the
    best of every plane (doc passages, code bodies, entities, compiled topic
    truths, past turns + their reference points). Any MCP client can prepend
    this to its prompt — full context transfer across models and chats.

    Bounded by CONTEXT_BUDGET_TOKENS (default 4000): evidence first, noise
    dropped. Unbounded dumps measurably hurt answer accuracy."""
    budget_tokens = budget_tokens or getattr(config, "CONTEXT_BUDGET_TOKENS", 4000)
    r = recall(question, top_k=3, storage_dir=storage_dir, namespace=namespace)
    parts = [f"### TruthGuard context for: {question}"]
    if r.get("communities"):
        parts.append("\n## Topics already discussed (compiled truths)")
        for c in r["communities"]:
            parts.append(f"- {c['summary'][:200]} [{c['n_turns']} past turns]")
    if r.get("matches"):
        parts.append("\n## Relevant past conversation")
        for m in r["matches"]:
            parts.append(f"- [{m['kind']} conf={m['confidence']}] Q: {m['question'][:120]}")
            if m.get("answer_preview"):
                parts.append(f"  A: {m['answer_preview'][:200]}")
            if m.get("grounded_on"):
                parts.append(f"  (grounded on: {', '.join(m['grounded_on'][:3])})")
    if r.get("doc_passages"):
        parts.append("\n## Document evidence")
        for p in r["doc_passages"]:
            parts.append(f"[{p['source']} {p['heading']}] \"{p['text'][:350]}\"")
    code_hits = r.get("code_passages") or []
    if not code_hits and (r.get("doc_passages") or r.get("entities")):
        # abstract question -> bridge to code via the evidence just found:
        # doc passages name the mechanisms; re-query the digest with them
        try:
            from . import code_digest
            probes = [question + " " + " ".join(
                          e["label"] for e in r.get("entities", [])[:3])]
            probes += [p["text"][:300] for p in r.get("doc_passages", [])[:2]]
            best = {}
            for pv in _embedder().encode(probes, normalize_embeddings=True):
                for h in code_digest.search(pv, k=2, min_sim=0.25):
                    key = (h["symbol"], h["file"])
                    if key not in best or h["similarity"] > best[key]["similarity"]:
                        best[key] = h
            # keyword bridge: doc evidence names mechanisms ("assessment gate")
            # whose stems ARE the symbol names (assess, _answerability)
            import re as _re, json as _json, os as _os
            jp = _os.path.join(config.STORAGE_DIR, "code_digest.json")
            if _os.path.exists(jp):
                words = {w.lower()[:6] for w in _re.findall(r"[a-zA-Z_]{6,}", " ".join(probes))}
                words -= {"contex", "graph", "global", "everyt", "insigh", "knowle",
                          "questi", "answer", "system", "docume"}
                for s in _json.load(open(jp, encoding="utf-8")):
                    name = s["symbol"].lstrip("_").lower()
                    if s["symbol"] in ("main",) or len(name) < 6:
                        continue
                    if any(name.startswith(w) for w in words):
                        key = (s["symbol"], s["file"])
                        if key not in best and s["file"].startswith("truthguard/"):
                            best[key] = {**s, "similarity": 0.34}   # rank above fuzzy hits
            code_hits = sorted(best.values(),
                               key=lambda h: -h["similarity"])[:3]
        except Exception:
            pass
    if code_hits:
        parts.append("\n## Code")
        for p in code_hits[:3]:
            parts.append(f"[{p['file']}:{p['lineno']}] {p['kind']} {p['symbol']}")
            parts.append("```python\n" + p["text"][:800] + "\n```")
    if r.get("entities"):
        parts.append("\n## Known facts (knowledge graph)")
        for e in r["entities"]:
            parts.append(f"- {e['label'][:150]}")
    if len(parts) == 1:
        return "No stored context is relevant to this question."
    return _fit_budget(parts, budget_tokens)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "what did we decide about the travel limit?"
    print(format_recall(recall(q)))
