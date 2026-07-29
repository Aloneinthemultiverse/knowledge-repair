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

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at REAL
);
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
    """One connection per thread. WAL so readers never block the writer."""
    key = f"conn::{storage_dir}"
    conn = getattr(_LOCAL, key, None)
    if conn is None:
        os.makedirs(storage_dir, exist_ok=True)
        conn = sqlite3.connect(_path(storage_dir), timeout=30,
                               isolation_level=None)   # explicit transactions
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)
        setattr(_LOCAL, key, conn)
    return conn


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
    def _do(c):
        now = time.time()
        c.executemany(
            "INSERT INTO nodes (id,data,updated_at) VALUES (?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
            [(str(n), json.dumps(d, default=str), now) for n, d in g.nodes(data=True)])
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


def exists(storage_dir: str) -> bool:
    p = _path(storage_dir)
    return os.path.exists(p) and os.path.getsize(p) > 0


def migrate_from_pickle(storage_dir: str, g: nx.DiGraph, last_spine: str = None) -> dict:
    """One-time import of an existing pickled graph."""
    v = write_graph(storage_dir, g, last_spine)
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "version": v}
