"""Row-level, crash-safe storage for the context graph.

The graph was a pickle of the whole DiGraph. That has two failure modes under a
fleet, and only the first is obvious:

  corruption     two processes write the file at once; one truncates mid-write
  lost updates   A loads 3,300 nodes and saves 3,301; B does the same
                 concurrently; whichever saves last erases the other's node —
                 no error, no corruption, the claim simply vanishes

The second is disqualifying for a system whose thesis is that nothing is
silently overwritten. A file lock does not fix it; it only makes the overwrite
orderly. The fix is to stop writing the graph as a file and write it as rows, so
two agents touching different nodes both survive.

SQLite in WAL mode gives serialized writers and non-blocking readers, which is
the write scheduler that would otherwise have to be built and supervised.

One subtlety this exists to solve: the admission gate does read-then-write —
check for a conflicting claim, then insert. Two agents submitting contradicting
claims simultaneously can both read "no conflict" and both write, so the
contradiction is never detected. `writer()` opens BEGIN IMMEDIATE, which takes
the write lock for the whole decision, so the second agent blocks, then reads
the updated state and correctly sees the conflict.
"""
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager

import networkx as nx

_LOCAL = threading.local()

# A handful of node fields are duplicated out of the JSON blob into real columns.
# The admission gate asks two questions on every single write — "what is already
# claimed about this subject+relation?" and "how many claims has this agent made
# this hour?" — and both were answered by scanning every node in the graph. That
# is O(graph) per claim, on the hot path of every answer once gating is
# automatic. Indexed columns turn both into index lookups.
#
# The JSON blob stays authoritative; these columns are a derived index, rewritten
# from it on every upsert, so they cannot drift.
_INDEXED = ("plane", "namespace", "subject", "relation", "agent_id",
            "claimant", "write_verdict", "retracted", "asserted_at")

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at REAL,
    plane TEXT, namespace TEXT, subject TEXT, relation TEXT,
    agent_id TEXT, claimant TEXT, write_verdict TEXT,
    retracted INTEGER, asserted_at REAL
);
CREATE INDEX IF NOT EXISTS idx_nodes_fact
    ON nodes(plane, namespace, subject, relation);
CREATE INDEX IF NOT EXISTS idx_nodes_agent
    ON nodes(plane, namespace, agent_id, asserted_at);
CREATE TABLE IF NOT EXISTS edges (
    src TEXT, dst TEXT, relation TEXT, data TEXT NOT NULL, updated_at REAL,
    PRIMARY KEY (src, dst, relation)
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
"""


def _path(storage_dir: str) -> str:
    return os.path.join(storage_dir, "graph.db")


def connect(storage_dir: str) -> sqlite3.Connection:
    """One connection per thread. WAL so readers never block the writer.

    Setup here is deliberately conditional. Every statement below except the
    busy_timeout needs the write lock, so running them unconditionally means a
    process that merely *opens* the database while another agent is mid-write
    dies with "database is locked" before it has done anything — which is the
    opposite of what WAL is here to provide. Each step is therefore skipped when
    it is already done, and retried when it genuinely has to run.
    """
    key = f"conn::{storage_dir}"
    conn = getattr(_LOCAL, key, None)
    if conn is None:
        os.makedirs(storage_dir, exist_ok=True)
        conn = sqlite3.connect(_path(storage_dir), timeout=30,
                               isolation_level=None)   # explicit transactions
        # must come first: it is what makes every later contended statement wait
        # instead of failing, and it never takes a lock itself
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        # journal_mode is persistent in the file, so it only needs setting once —
        # and re-setting it demands an exclusive lock no other writer will yield
        if (conn.execute("PRAGMA journal_mode").fetchone()[0] or "").lower() != "wal":
            _retry(conn, lambda c: c.execute("PRAGMA journal_mode=WAL"))
        _ensure_schema(conn)
        setattr(_LOCAL, key, conn)
    return conn


def _retry(conn, fn, attempts: int = 6):
    """Retry a lock-taking statement. busy_timeout covers most contention, but
    schema changes and journal_mode can return SQLITE_BUSY without consulting
    the busy handler at all."""
    for i in range(attempts):
        try:
            return fn(conn)
        except sqlite3.OperationalError as e:
            if "lock" not in str(e).lower() or i == attempts - 1:
                raise
            time.sleep(0.05 * (2 ** i))


def _ensure_schema(conn) -> None:
    """Create tables and the derived index columns, only when actually absent.

    SQLite has no ADD COLUMN IF NOT EXISTS, and a graph predating those columns
    should keep working rather than requiring a manual migration; the values
    backfill themselves on the next write of each node.
    """
    have_tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"nodes", "edges", "meta"} <= have_tables:
        _retry(conn, lambda c: c.executescript(SCHEMA))

    missing = [c for c in _INDEXED
               if c not in {r[1] for r in conn.execute("PRAGMA table_info(nodes)")}]
    for col in missing:
        typ = "REAL" if col == "asserted_at" else (
            "INTEGER" if col == "retracted" else "TEXT")
        _retry(conn, lambda c, col=col, typ=typ:
               c.execute(f"ALTER TABLE nodes ADD COLUMN {col} {typ}"))
    if missing:
        _retry(conn, lambda c: c.executescript(SCHEMA))   # (re)create the indexes


@contextmanager
def writer(storage_dir: str):
    """Exclusive write transaction covering an entire read-then-write decision.

    BEGIN IMMEDIATE acquires the write lock up front rather than on first write,
    so a gate check and the insert that depends on it cannot interleave with
    another agent's.
    """
    conn = connect(storage_dir)
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def version(storage_dir: str) -> int:
    row = connect(storage_dir).execute(
        "SELECT v FROM meta WHERE k='version'").fetchone()
    return int(row[0]) if row else 0


def _bump(conn) -> int:
    row = conn.execute("SELECT v FROM meta WHERE k='version'").fetchone()
    v = (int(row[0]) if row else 0) + 1
    conn.execute("INSERT INTO meta (k,v) VALUES ('version',?) "
                 "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (str(v),))
    return v


def write_graph(storage_dir: str, g: nx.DiGraph, last_spine: str = None,
                conn: sqlite3.Connection = None) -> int:
    """Upsert every node and edge as rows.

    Row-level upserts are what make concurrent agents safe: two agents adding
    different nodes both persist, because neither rewrites the other's row.
    """
    cols = ",".join(_INDEXED)
    sets = ",".join(f"{c}=excluded.{c}" for c in _INDEXED)
    ins = (f"INSERT INTO nodes (id,data,updated_at,{cols}) "
           f"VALUES (?,?,?,{','.join('?' * len(_INDEXED))}) "
           f"ON CONFLICT(id) DO UPDATE SET data=excluded.data, "
           f"updated_at=excluded.updated_at,{sets}")

    def _row(n, d, now):
        return (str(n), json.dumps(d, default=str), now,
                *(int(bool(d.get(c))) if c == "retracted" else d.get(c)
                  for c in _INDEXED))

    def _do(c):
        now = time.time()
        c.executemany(ins, [_row(n, d, now) for n, d in g.nodes(data=True)])
        c.executemany(
            "INSERT INTO edges (src,dst,relation,data,updated_at) VALUES (?,?,?,?,?) "
            "ON CONFLICT(src,dst,relation) DO UPDATE SET data=excluded.data, "
            "updated_at=excluded.updated_at",
            [(str(u), str(v_), str(d.get("relation") or ""),
              json.dumps(d, default=str), now) for u, v_, d in g.edges(data=True)])
        c.execute("INSERT INTO meta (k,v) VALUES ('last_spine',?) "
                  "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (last_spine or "",))
        c.execute("INSERT INTO meta (k,v) VALUES ('n_turns',?) "
                  "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                  (str(g.graph.get("n_turns", 0)),))
        return _bump(c)

    if conn is not None:                      # already inside a transaction
        return _do(conn)
    with writer(storage_dir) as c:
        return _do(c)


def read_graph(storage_dir: str):
    """Rebuild a DiGraph from rows. Returns (graph, last_spine, version)."""
    conn = connect(storage_dir)
    g = nx.DiGraph()
    for nid, data in conn.execute("SELECT id,data FROM nodes"):
        g.add_node(nid, **json.loads(data))
    for src, dst, rel, data in conn.execute("SELECT src,dst,relation,data FROM edges"):
        g.add_edge(src, dst, **json.loads(data))
    meta = dict(conn.execute("SELECT k,v FROM meta").fetchall())
    g.graph["n_turns"] = int(meta.get("n_turns", 0) or 0)
    return g, (meta.get("last_spine") or None), int(meta.get("version", 0) or 0)


def facts_about(storage_dir: str, namespace: str, subject: str,
                relation: str) -> list:
    """Asserted claims on one subject+relation. Returns [(node_id, data)].

    This is the gate's contradiction check. Scoped by the covering index, so it
    costs the same on a 3,000-node graph and a 300,000-node one — which matters
    because it now runs on every conclusion the fleet reaches, not only on the
    ones an agent chose to submit.
    """
    rows = connect(storage_dir).execute(
        "SELECT id,data FROM nodes WHERE plane='claim' AND namespace=? "
        "AND subject=? AND relation=? "
        "AND (write_verdict IS NULL OR write_verdict != 'QUARANTINED') "
        "AND (retracted IS NULL OR retracted = 0)",
        (namespace, subject, relation)).fetchall()
    return [(r[0], json.loads(r[1])) for r in rows]


def claims_since(storage_dir: str, namespace: str, agent_id: str,
                 cutoff: float) -> int:
    """How many claims this agent has written since `cutoff` — the quota count."""
    return connect(storage_dir).execute(
        "SELECT COUNT(*) FROM nodes WHERE plane='claim' AND namespace=? "
        "AND agent_id=? AND asserted_at >= ?",
        (namespace, agent_id, cutoff)).fetchone()[0]


def exists(storage_dir: str) -> bool:
    p = _path(storage_dir)
    return os.path.exists(p) and os.path.getsize(p) > 0


def migrate_from_pickle(storage_dir: str, g: nx.DiGraph, last_spine: str = None) -> dict:
    """One-time import of an existing pickled graph."""
    v = write_graph(storage_dir, g, last_spine)
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "version": v}
