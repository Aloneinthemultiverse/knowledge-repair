"""FR-2.3..2.7 — retrieval: multi-query + RRF fusion + rerank + provenance weighting.

retrieve(store, question, llm=None) -> list of chunk dicts (top TOP_K_FINAL),
each annotated with retrieval_score. Zero-LLM if llm is None (interpretations off).
"""
import re

from . import config

_INJECTION_RE = re.compile(
    r"(ignore (all |any )?(previous|prior|above) instructions|system prompt|"
    r"you are now|disregard .*instructions|do not follow)", re.I)

_DOC_SCOPE_RE = re.compile(
    r"\b(20\d{2})\b.*\b(policy|edition|memo|guide)|\b(policy|edition|memo|guide)\b.*\b(20\d{2})\b", re.I)


def rrf_fuse(rankings: list, k: int = None, limit: int = 60) -> list:
    k = k or config.RRF_K
    scores = {}
    for ranking in rankings:
        for rank, (cid, _s) in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]


_ENTITY_RE = re.compile(r'"([^"]+)"|\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)*)\b|\b(\d[\d,.$%/-]*\d|\d)\b')
_STOP = {"The", "What", "When", "Where", "Which", "Who", "How", "Why", "Did",
         "Does", "Is", "Are", "Was", "Were", "I", "My", "You", "A", "An"}


def _question_entities(question: str) -> list:
    """Entities from the question: quoted phrases, proper nouns, numbers/dates.
    Mem0-2026 style entity signal — matched exactly against chunk text."""
    ents = set()
    for m in _ENTITY_RE.finditer(question):
        val = next((g for g in m.groups() if g), None)
        if val and val not in _STOP and len(val) > 1:
            ents.add(val.strip())
    return list(ents)


def _entity_ranking(question: str, store) -> list:
    """Third retrieval signal: rank chunks by count of exact question-entity
    matches in their text. Zero-LLM, fuses into RRF alongside vector + BM25."""
    ents = _question_entities(question)
    if not ents:
        return []
    scored = []
    for cid in store._ids:
        text = store.by_id[cid]["text"]
        hits = sum(1 for e in ents if e in text)
        if hits:
            scored.append((cid, float(hits)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:25]


def _interpretations(question: str, llm) -> list:
    """Superposed multi-query: up to N_INTERPRETATIONS readings of the question."""
    if llm is None:
        return [question]
    try:
        raw = llm.complete(
            f"Give {config.N_INTERPRETATIONS} distinct search-query interpretations "
            f"of this question, one per line, no numbering, no commentary:\n{question}",
            max_tokens=150)
        lines = [l.strip("-• ").strip() for l in raw.splitlines() if l.strip()]
        outs = [question] + [l for l in lines if 5 < len(l) < 200]
        return outs[: config.N_INTERPRETATIONS + 1]
    except Exception:
        return [question]


def _rerank(question: str, candidates: list, store) -> list:
    """Cross-encoder rerank top-fused -> ordered list of (cid, score). Falls back
    to fused order if the model is unavailable."""
    try:
        from sentence_transformers import CrossEncoder
        global _CE
        if "_CE" not in globals():
            _CE = CrossEncoder(config.RERANK_MODEL)
        pairs = [(question, store.by_id[cid]["text"][:1000]) for cid, _ in candidates]
        scores = _CE.predict(pairs)
        order = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        return [(cid, float(s)) for (cid, _f), s in order]
    except Exception:
        return candidates


def _provenance_weight(chunk: dict) -> float:
    w = 1.0
    if chunk.get("extraction") == "ocr":
        conf = chunk.get("ocr_conf") or 0.5
        w *= 0.7 + 0.3 * conf                     # low-OCR chunks down-weighted
    if _INJECTION_RE.search(chunk["text"]):
        w *= 0.4                                  # injection-phrase down-weight
    return w


def _doc_scope_filter(question: str, ranked: list, store) -> list:
    """If the question names a year+doctype, prefer chunks from matching files."""
    m = _DOC_SCOPE_RE.search(question)
    if not m:
        return ranked
    year = next((g for g in m.groups() if g and g.isdigit()), None)
    if not year:
        return ranked
    scoped = [(cid, s) for cid, s in ranked if year in store.by_id[cid]["source_file"]]
    return scoped if scoped else ranked


def retrieve(store, question: str, llm=None, namespace: str = None) -> list:
    """Full retrieval pass. Returns top chunks with .retrieval_score set.

    Scoped to `namespace` so one tenant's documents are not retrievable from
    another. Chunks ingested before namespaces existed carry none and stay
    visible, so existing corpora behave as before.
    """
    import os
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")

    def _visible(cid):
        ns = store.by_id[cid].get("namespace")
        return ns is None or ns == namespace
    rankings = []
    for q in _interpretations(question, llm):
        v = store.vector_search(q, k=25)
        b = store.keyword_search(q, k=25)
        if v:
            rankings.append(v)
        if b:
            rankings.append(b)
    e = _entity_ranking(question, store)     # 3rd signal (Mem0-2026 entity match)
    if e:
        rankings.append(e)
    fused = rrf_fuse(rankings, limit=config.TOP_K_FUSED)
    fused = _doc_scope_filter(question, fused, store)
    reranked = _rerank(question, fused, store)

    weighted = []
    for cid, score in reranked:
        if not _visible(cid):
            continue
        chunk = store.by_id[cid]
        weighted.append((score * _provenance_weight(chunk), cid))
    weighted.sort(reverse=True)

    out = []
    for wscore, cid in weighted[: config.TOP_K_FINAL]:
        c = dict(store.by_id[cid])
        c["retrieval_score"] = round(float(wscore), 4)
        out.append(c)
    return out
