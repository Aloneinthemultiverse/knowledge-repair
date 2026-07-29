"""The z-plane: what agents DID, as opposed to what they concluded.

x, y+ and y- all answer "what is known" — conclusions, documents, code. None
answer "what was done". For one assistant that gap is tolerable; for a fleet it
is disqualifying, because several agents act concurrently and a conclusion alone
cannot say which agent produced it, through which actions, or whether the run
succeeded.

Two stores, deliberately:

  raw tool calls  -> flat SQLite table. One session emits hundreds of calls
                     against a ~3,300-node graph; ingesting them raw would swamp
                     every traversal.
  episode rollup  -> ONE node per run on the z-plane, edged to the conclusion it
                     produced. This mirrors the split already in use: 603 raw
                     chunks in chunks.json, 86 rollup nodes in the graph.

Usage:
    ep = start_episode("answer question about EC2 limits")
    log_tool_call(ep, "retrieve", {"k": 10}, outcome="ok", duration_ms=120)
    end_episode(ep, outcome="SUCCESS")
"""
import json
import os
import re
import sqlite3
import time
import uuid

from . import config

_DB = None


def _db():
    """Append-only store for raw calls. WAL so concurrent agents can write."""
    global _DB
    if _DB is None:
        os.makedirs(config.STORAGE_DIR, exist_ok=True)
        _DB = sqlite3.connect(os.path.join(config.STORAGE_DIR, "episodes.db"),
                              check_same_thread=False)
        _DB.execute("PRAGMA journal_mode=WAL")
        _DB.executescript("""
            CREATE TABLE IF NOT EXISTS tool_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT, agent_id TEXT, ts REAL,
                tool TEXT, args_json TEXT,
                outcome TEXT,            -- ok | error | timeout
                error_text TEXT, duration_ms INTEGER, exit_code INTEGER
            );
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                agent_id TEXT, namespace TEXT, goal TEXT,
                started_at REAL, ended_at REAL,
                outcome TEXT,            -- SUCCESS | FAILURE | PARTIAL
                failure_signature TEXT, n_retries INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_calls_ep ON tool_calls(episode_id);
        """)
        _DB.commit()
    return _DB


# Normalise an error to a class so failures cluster instead of scattering:
# "FileNotFoundError: /a/b/c.py" and "FileNotFoundError: /x/y.py" are the same
# failure mode and should retrieve each other.
_SIG_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Timeout))")


def failure_signature(error_text: str) -> str:
    if not error_text:
        return ""
    m = _SIG_RE.match(error_text.strip())
    if m:
        return m.group(1)
    return re.sub(r"[0-9]+|/[^\s]+", "", error_text.strip())[:60].strip()


def start_episode(goal: str, agent_id: str = None, namespace: str = None) -> str:
    agent_id = agent_id or os.getenv("TG_AGENT_ID", "default")
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")
    eid = "ep:" + uuid.uuid4().hex[:12]
    db = _db()
    db.execute("INSERT INTO episodes (episode_id, agent_id, namespace, goal, "
               "started_at, outcome, n_retries) VALUES (?,?,?,?,?,?,0)",
               (eid, agent_id, namespace, goal[:300], time.time(), "RUNNING"))
    db.commit()
    return eid


def log_tool_call(episode_id: str, tool: str, args: dict = None,
                  outcome: str = "ok", duration_ms: int = 0,
                  error_text: str = "", exit_code: int = 0) -> None:
    """Never raise — observability must not be able to break the thing it observes."""
    try:
        db = _db()
        db.execute("INSERT INTO tool_calls (episode_id, agent_id, ts, tool, "
                   "args_json, outcome, error_text, duration_ms, exit_code) "
                   "VALUES (?,?,?,?,?,?,?,?,?)",
                   (episode_id, os.getenv("TG_AGENT_ID", "default"), time.time(),
                    tool, json.dumps(args or {})[:2000], outcome,
                    (error_text or "")[:500], int(duration_ms), int(exit_code)))
        db.commit()
    except Exception:
        pass


def end_episode(episode_id: str, outcome: str = "SUCCESS", cg=None) -> dict:
    """Close the run, roll it up, and add ONE node to the z-plane."""
    db = _db()
    calls = db.execute(
        "SELECT tool, outcome, error_text, duration_ms FROM tool_calls "
        "WHERE episode_id=? ORDER BY id", (episode_id,)).fetchall()
    row = db.execute("SELECT agent_id, namespace, goal, started_at FROM episodes "
                     "WHERE episode_id=?", (episode_id,)).fetchone()
    if not row:
        return {"error": "unknown episode"}
    agent_id, namespace, goal, started = row

    seq = [c[0] for c in calls]
    errors = [c[2] for c in calls if c[1] != "ok" and c[2]]
    sig = failure_signature(errors[-1]) if errors else ""
    retries = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1])
    dur = time.time() - (started or time.time())
    if outcome == "SUCCESS" and errors:
        outcome = "PARTIAL"

    db.execute("UPDATE episodes SET ended_at=?, outcome=?, failure_signature=?, "
               "n_retries=? WHERE episode_id=?",
               (time.time(), outcome, sig, retries, episode_id))
    db.commit()

    if cg is None:
        from .context_graph import ContextGraph
        cg = ContextGraph()
    cg.g.add_node(episode_id, plane="action", agent_id=agent_id,
                  namespace=namespace, goal=goal,
                  tool_sequence=seq[:40], outcome=outcome,
                  failure_signature=sig, n_retries=retries,
                  duration=round(dur, 2))
    cg.save()
    return {"episode_id": episode_id, "outcome": outcome, "n_calls": len(calls),
            "tool_sequence": seq, "failure_signature": sig, "n_retries": retries,
            "duration": round(dur, 2)}


def similar_failures(error_text: str = "", tool_sequence: list = None,
                     k: int = 5) -> list:
    """Day-one diagnosis, no model involved: which past runs failed like this?

    Matches on failure signature first (the strong signal), then on tool-sequence
    overlap, so 'what was running when this broke' is answerable immediately.
    """
    sig = failure_signature(error_text)
    seq = set(tool_sequence or [])
    rows = _db().execute(
        "SELECT episode_id, agent_id, goal, outcome, failure_signature, n_retries "
        "FROM episodes WHERE outcome IN ('FAILURE','PARTIAL')").fetchall()
    scored = []
    for eid, agent, goal, outcome, fsig, retries in rows:
        score = 0.0
        if sig and fsig == sig:
            score += 1.0
        elif sig and fsig and (sig in fsig or fsig in sig):
            score += 0.5
        if seq:
            calls = _db().execute(
                "SELECT DISTINCT tool FROM tool_calls WHERE episode_id=?",
                (eid,)).fetchall()
            past = {c[0] for c in calls}
            if past:
                score += 0.5 * len(seq & past) / len(seq | past)
        if score > 0:
            scored.append({"episode_id": eid, "agent_id": agent, "goal": goal,
                           "outcome": outcome, "failure_signature": fsig,
                           "n_retries": retries, "score": round(score, 3)})
    scored.sort(key=lambda x: -x["score"])
    return scored[:k]


def stats() -> dict:
    db = _db()
    n_ep = db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    n_call = db.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    by_outcome = dict(db.execute(
        "SELECT outcome, COUNT(*) FROM episodes GROUP BY outcome").fetchall())
    by_agent = dict(db.execute(
        "SELECT agent_id, COUNT(*) FROM episodes GROUP BY agent_id").fetchall())
    top_fail = db.execute(
        "SELECT failure_signature, COUNT(*) c FROM episodes "
        "WHERE failure_signature != '' GROUP BY failure_signature "
        "ORDER BY c DESC LIMIT 5").fetchall()
    return {"episodes": n_ep, "tool_calls": n_call, "by_outcome": by_outcome,
            "by_agent": by_agent, "top_failures": dict(top_fail)}
