"""TruthGuard MCP server — chat with the memory and WATCH THE GRAPH GROW.

Tools:
  ask(question, followup?, baseline?)  -> self-corrected answer (records a turn)
  ingest_document(path)                -> add a file to the corpus + reindex
  link_code_repo(path)                 -> gitnexus-index a repo as the y- plane
  rebuild_communities()                -> re-run DG community recipe on all planes
  graph_stats()                        -> nodes/edges/planes/turns
  live_view_url()                      -> URL of the auto-refreshing 3D view

Every mutating call re-exports graph3d_data.json; a tiny HTTP server serves
graph3d_live.html which polls it — the 3D view updates in place while you chat.

Connect (Claude Code):
  claude mcp add truthguard -- python -m truthguard.mcp_server
  (cwd must be the dg-core folder; or use absolute paths in .claude.json)
"""
import os
import sys
import json
import shutil
import asyncio
import threading
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.getenv("TG_LIVE_PORT", "7787"))

app = Server("truthguard")
_state = {"store": None, "llm": None}


def _store():
    """Chunk store, reloaded when another process has rewritten the index.

    Studio and this MCP server are separate processes sharing one storage dir,
    so an ingest in one is invisible to the other's in-memory copy. Comparing
    chunks.json's mtime makes the shared graph actually shared: ingest in the
    web UI, ask in an MCP client, no restart.
    """
    import os
    from . import config
    p = os.path.join(config.STORAGE_DIR, "chunks.json")
    mt = os.path.getmtime(p) if os.path.exists(p) else 0.0
    if _state["store"] is None or mt > _state.get("mtime", 0.0):
        from .chunk_store import ChunkStore
        _state["store"] = ChunkStore()
        _state["mtime"] = mt
    return _state["store"]


def _llm():
    if _state["llm"] is None:
        from .llm import LLM
        _state["llm"] = LLM()
    return _state["llm"]


def _export_live_data():
    """Dump the FULL 3-plane graph (docs+code+chat) for the live 3D view."""
    try:
        from .full3d import export_full
        export_full()
    except Exception:
        pass


def _start_live_server():
    import http.server, functools
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=ROOT)
    handler.log_message = lambda *a, **k: None

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=ROOT, **k)
        def log_message(self, *a):
            pass

    def run():
        try:
            http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Quiet).serve_forever()
        except OSError:
            pass  # already running from a previous session
    threading.Thread(target=run, daemon=True).start()


TOOLS = [
    types.Tool(name="ask",
        description="Ask the TruthGuard self-correcting RAG a question. Detects "
                    "contradictions (dual-answers), ambiguity (clarifies), and missing "
                    "info (refuses with gap analysis). Records the turn into the 3-plane "
                    "context graph — the live 3D view grows with every call.",
        inputSchema={"type": "object", "properties": {
            "question": {"type": "string"},
            "followup": {"type": "string", "description": "answer to a prior clarify"},
            "baseline": {"type": "boolean", "description": "bypass the correction layer (ablation)"}},
            "required": ["question"]}),
    types.Tool(name="ingest_document",
        description="Add a document (PDF/DOCX/MD/TXT — scans OK, OCR runs) to the corpus "
                    "and rebuild the index. The knowledge plane grows.",
        inputSchema={"type": "object", "properties": {
            "path": {"type": "string", "description": "absolute path to the file"}},
            "required": ["path"]}),
    types.Tool(name="link_code_repo",
        description="Index a git repository with GitNexus as the y- code plane. Code "
                    "questions then answer structurally (callers, symbols) with zero LLM.",
        inputSchema={"type": "object", "properties": {
            "path": {"type": "string", "description": "absolute path to the repo"}},
            "required": ["path"]}),
    types.Tool(name="rebuild_communities",
        description="Re-run DecisionGraph's community recipe on all three planes "
                    "(topic communities on turns, semantic communities on knowledge).",
        inputSchema={"type": "object", "properties": {}}),
    types.Tool(name="graph_stats",
        description="Current 3-plane graph statistics: nodes/edges per plane, turns, communities.",
        inputSchema={"type": "object", "properties": {}}),
    types.Tool(name="recall",
        description="Search PAST conversation turns (DG DecisionMemory.query recipe): "
                    "embeds the question, finds similar old turns + their topic "
                    "communities, returns each with the documents/code it grounded on. "
                    "Use before answering anything that might have been discussed before.",
        inputSchema={"type": "object", "properties": {
            "question": {"type": "string"}}, "required": ["question"]}),
    types.Tool(name="ingest_project",
        description="Absorb a WHOLE project in one call: the entire codebase "
                    "(GitNexus index + every function body, involved in chat or "
                    "not), every document in the repo (.md/.pdf/.docx/.txt), and "
                    "optionally a chat transcript — all cross-linked with "
                    "reference points and shown in the 3D view. For y+ entities "
                    "afterwards, call rebuild_communities.",
        inputSchema={"type": "object", "properties": {
            "repo_path": {"type": "string", "description": "absolute path to the project repo"},
            "chat_path": {"type": "string", "description": "optional chat transcript (.jsonl or user:/assistant: text)"}},
            "required": ["repo_path"]}),
    types.Tool(name="ingest_chat",
        description="Import a whole chat transcript into the x plane: Claude session "
                    ".jsonl OR plain text with 'user:'/'assistant:' lines. Each turn "
                    "becomes a spine node, auto-cross-linked to the entities, document "
                    "chunks and code it talks about. The 3D view grows immediately.",
        inputSchema={"type": "object", "properties": {
            "path": {"type": "string", "description": "absolute path to the transcript"}},
            "required": ["path"]}),
    types.Tool(name="get_context",
        description="CONTEXT ROUTER — call this BEFORE answering any question. "
                    "Returns one ready-to-use context block with the best of every "
                    "plane: document evidence, actual code bodies, knowledge-graph "
                    "facts, compiled topic truths, and relevant past conversation "
                    "with its reference points. Prepend it to your reasoning; it is "
                    "the shared memory across all chats, models, and tools.",
        inputSchema={"type": "object", "properties": {
            "question": {"type": "string"}}, "required": ["question"]}),
    types.Tool(name="query_code",
        description="Traverse the y- code graph structurally (zero LLM): given a "
                    "symbol name, returns its definition (actual source), callers, "
                    "and callees from the GitNexus index. Optionally pass a raw "
                    "cypher query instead.",
        inputSchema={"type": "object", "properties": {
            "symbol": {"type": "string", "description": "function/class name"},
            "cypher": {"type": "string", "description": "raw GitNexus cypher (advanced)"}}}),
    types.Tool(name="graph_query",
        description="Structural traversal of THE PROJECT'S OWN 3-plane graph (the "
                    "one in the 3D view) — never any external index: "
                    "context (360° view of any node — code symbol, doc entity, or "
                    "chat turn — with all edges grouped by relation + source body), "
                    "impact (cross-plane blast radius: which code, doc entities AND "
                    "chat turns are wired to this node), "
                    "find (locate nodes by name on any plane), "
                    "edit_plan (pre-edit checklist: impact radius + callers whose "
                    "contract must hold + callees + docs/chat decisions to review "
                    "so original functionality is preserved), "
                    "path (shortest path between two concepts across planes, each hop "
                    "tagged EXTRACTED/INFERRED — pass the second concept as target), "
                    "report (graph highlights: god nodes, surprising links, questions).",
        inputSchema={"type": "object", "properties": {
            "command": {"type": "string", "enum": ["context", "impact", "find",
                        "edit_plan", "path", "report"]},
            "name": {"type": "string", "description": "symbol / entity / node name (unused for report)"},
            "target": {"type": "string", "description": "second concept (path only)"}},
            "required": ["command"]}),
    types.Tool(name="live_view_url",
        description="URL of the live auto-refreshing 3D graph view (open in a browser and keep it open while chatting).",
        inputSchema={"type": "object", "properties": {}}),
]


@app.list_tools()
async def list_tools():
    return TOOLS


def _fmt_response(r: dict) -> str:
    out = [f"[{r['kind'].upper()}]"
           + (f" confidence={r['confidence']} ({r['band']})" if r.get("confidence") is not None else "")]
    out.append(r["text"])
    if r.get("citations"):
        out.append("Sources: " + " | ".join(r["citations"]))
    for f in r.get("figures") or []:
        out.append(f"[image] {f['figure']} -> {f['image_path']}")
    out.append("trace: " + " -> ".join(s["step"] for s in r["trace"]))
    out.append(f"(graph grew — refresh feeling not needed, live view updates itself)")
    return "\n".join(out)


@app.call_tool()
async def call_tool(name: str, args: dict):
    try:
        if name == "ask":
            from .controller import ask as _ask
            r = _ask(_store(), _llm(), args["question"],
                     baseline=bool(args.get("baseline")),
                     followup=args.get("followup"))
            _export_live_data()
            return [types.TextContent(type="text", text=_fmt_response(r))]

        if name == "ingest_document":
            src = args["path"]
            if not os.path.isfile(src):
                return [types.TextContent(type="text", text=f"file not found: {src}")]
            dst = os.path.join(config.CORPUS_DIR, os.path.basename(src))
            os.makedirs(config.CORPUS_DIR, exist_ok=True)
            shutil.copy2(src, dst)
            from .pipeline import ingest_corpus
            from .chunk_store import build_index
            chunks, report = ingest_corpus()
            engine = build_index()
            _state["store"] = None      # reload with new chunks
            # re-wire reference points: existing chat turns may talk about this doc
            try:
                from sentence_transformers import SentenceTransformer
                from .context_graph import ContextGraph
                from .planes import retro_link_spine
                retro_link_spine(ContextGraph(), SentenceTransformer(config.EMBED_MODEL))
            except Exception:
                pass
            _export_live_data()
            return [types.TextContent(type="text", text=
                f"ingested {os.path.basename(src)} — corpus now {report['total_chunks']} chunks "
                f"(pages: {report['pages']}), index: {engine}. Ask about it!")]

        if name == "link_code_repo":
            import subprocess
            path = args["path"]
            res = subprocess.run(["gitnexus", "analyze", path],
                                 capture_output=True, text=True, timeout=600, shell=True)
            from . import code_link
            code_link.CODE_REPO = os.path.basename(os.path.normpath(path))
            code_link._cache.clear()
            from . import code_digest
            dig = code_digest.build([ROOT, path], incremental=True)   # refresh code BODIES for recall
            _export_live_data()
            tail = (res.stdout or res.stderr).strip().splitlines()[-2:]
            return [types.TextContent(type="text", text=
                f"repo indexed as y- plane: {code_link.CODE_REPO}\n" + "\n".join(tail)
                + f"\ncode digest: {dig['symbols']} symbols from {dig['files']} files (bodies retrievable)"
                + "\nAsk 'who calls X?' for zero-LLM structural answers.")]

        if name == "rebuild_communities":
            import anthropic
            from sentence_transformers import SentenceTransformer
            from .context_graph import ContextGraph
            from .planes import build_x, build_y_minus
            cg = ContextGraph()
            cg.g.remove_nodes_from([n for n, d in cg.g.nodes(data=True)
                                    if d.get("plane") == "x_community"])
            client = anthropic.Anthropic(base_url=config.LLM_BASE_URL,
                                         api_key=config.LLM_API_KEY)
            embed = SentenceTransformer(config.EMBED_MODEL)
            rx = build_x(cg, client, embed)
            ry = build_y_minus(cg)
            _export_live_data()
            return [types.TextContent(type="text", text=
                f"communities rebuilt — x: {rx}, y-: {ry}. Live view updated.")]

        if name == "graph_stats":
            from .context_graph import ContextGraph
            from collections import Counter
            g = ContextGraph().g
            planes = Counter(d.get("plane") for _, d in g.nodes(data=True))
            rels = Counter(e.get("relation") for _, _, e in g.edges(data=True))
            return [types.TextContent(type="text", text=json.dumps(
                {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
                 "turns": g.graph.get("n_turns", 0),
                 "planes": dict(planes), "edge_types": dict(rels)}, indent=1))]

        if name == "ingest_project":
            from .ingest_all import ingest_project
            r = ingest_project(args["repo_path"], args.get("chat_path"))
            _state["store"] = None
            return [types.TextContent(type="text", text=json.dumps(r, indent=1))]

        if name == "ingest_chat":
            from .import_chat import import_chat
            r = import_chat(args["path"])
            return [types.TextContent(type="text", text=json.dumps(r, indent=1))]

        if name == "get_context":
            from .recall import get_context
            return [types.TextContent(type="text",
                    text=get_context(args["question"]))]

        if name == "recall":
            from .recall import recall, format_recall
            return [types.TextContent(type="text",
                    text=format_recall(recall(args["question"])))]

        if name == "query_code":
            from . import code_link
            if args.get("cypher"):
                md = code_link._cypher(args["cypher"])
                return [types.TextContent(type="text", text=md or "no results")]
            info = code_link.symbol_info(args.get("symbol", ""))
            return [types.TextContent(type="text", text=json.dumps(info, indent=1))]

        if name == "graph_query":
            from . import graph_query
            cmd = args["command"]
            if cmd == "report":
                r = graph_query.report()
            elif cmd == "path":
                r = graph_query.path(args["name"], args.get("target", ""))
            else:
                fn = {"context": graph_query.context, "impact": graph_query.impact,
                      "find": graph_query.find,
                      "edit_plan": graph_query.edit_plan}[cmd]
                r = fn(args["name"])
            return [types.TextContent(type="text",
                    text=graph_query.fmt(r)[:8000])]

        if name == "live_view_url":
            _export_live_data()
            return [types.TextContent(type="text", text=
                f"http://127.0.0.1:{PORT}/graph3d_live.html — open it and keep it open; "
                f"it polls every 4s and grows as you chat (camera position is preserved).")]

        return [types.TextContent(type="text", text=f"unknown tool: {name}")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"error: {type(e).__name__}: {e}")]


def _start_watcher(interval: int = 20):
    """graphify-style watch mode: poll source mtimes; on change re-digest the
    code bodies and refresh the live 3D view — the graph grows as you code."""
    import time

    def newest(root):
        latest = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in
                           (".git", "__pycache__", "node_modules", ".claude",
                            "storage", ".gitnexus", ".venv")]
            for fn in filenames:
                if fn.endswith((".py", ".js", ".jsx", ".ts", ".tsx", ".md")):
                    try:
                        latest = max(latest, os.path.getmtime(
                            os.path.join(dirpath, fn)))
                    except OSError:
                        pass
        return latest

    def run():
        last = newest(ROOT)
        while True:
            time.sleep(interval)
            try:
                cur = newest(ROOT)
                if cur > last:
                    last = cur
                    from . import code_digest
                    # incremental: only the file(s) that changed get re-parsed and
                    # re-embedded, so the watcher reacts to an edit in ~0.1s
                    # instead of re-embedding the whole tree every save.
                    code_digest.build(incremental=True)
                    _export_live_data()
            except Exception:
                pass
    threading.Thread(target=run, daemon=True).start()


async def main():
    _start_live_server()
    _start_watcher()
    _export_live_data()
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
