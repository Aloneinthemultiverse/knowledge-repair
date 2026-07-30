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

    # ── fleet: governed writes into shared memory ────────────────────────────
    types.Tool(name="submit_claim",
        description="Submit a conclusion to SHARED memory through the admission gate. "
                    "Use this instead of asserting a fact directly when other agents "
                    "read the same graph. Returns ACCEPTED, QUARANTINED (below the "
                    "confidence floor — stored but not served), or CONFLICTED (an "
                    "existing claim asserts a different value over an overlapping "
                    "validity window; BOTH are kept and a conflict is raised for a "
                    "human). Nothing is ever silently overwritten.",
        inputSchema={"type": "object", "properties": {
            "subject": {"type": "string", "description": "what the claim is about"},
            "relation": {"type": "string", "description": "the property being asserted"},
            "object": {"type": "string", "description": "the asserted value"},
            "confidence": {"type": "number", "description": "0-1; be honest, low confidence is quarantined not rejected"},
            "valid_from": {"type": "string", "description": "ISO date this becomes true; omit if always"},
            "valid_until": {"type": "string", "description": "ISO date it stops being true; omit if current"},
            "severity": {"type": "string", "description": "low | normal | high | critical"},
            "sources": {"type": "array", "items": {"type": "string"},
                        "description": "chunk or node ids this was grounded on"},
            "derived_from": {"type": "array", "items": {"type": "string"},
                        "description": "ids of OTHER CLAIMS this was reasoned from, so a "
                                       "later retraction can flag this one"},
            # Without this the gate falls back to the agent id, and a
            # disagreement between two document revisions is recorded as
            # "agent-00 vs agent-05" — blaming the couriers for what the sources
            # said. admission.py has always supported the distinction; the tool
            # agents actually call simply never exposed it, so nobody could use
            # it. Found by reading the claimants of a real 20-agent run.
            "claimant": {"type": "string",
                        "description": "WHO ASSERTS this — the source document, when you "
                                       "are relaying what you read. Omit only if the "
                                       "conclusion is your own reasoning."},
            "namespace": {"type": "string"}},
            "required": ["subject", "relation", "object", "confidence"]}),

    types.Tool(name="open_conflicts",
        description="Unresolved disagreements between agents in shared memory, most "
                    "severe first. Check this before acting on a fact that another "
                    "agent may have contradicted.",
        inputSchema={"type": "object", "properties": {
            "namespace": {"type": "string"},
            "min_severity": {"type": "string", "description": "low | normal | high | critical"}}}),

    types.Tool(name="adjudicate",
        description="Resolve a conflict by naming the winning claim. The loser is "
                    "retracted (not deleted — the audit trail survives) and every "
                    "conclusion derived from it is flagged for review. The conflict "
                    "stops appearing in the queue.",
        inputSchema={"type": "object", "properties": {
            "conflict_id": {"type": "string"},
            "winning_node": {"type": "string"},
            "resolved_by": {"type": "string"}},
            "required": ["conflict_id"]}),

    types.Tool(name="retract_claim",
        description="Mark a stored claim as wrong and flag everything reasoned from it. "
                    "Use when a source turns out to be unreliable outside of a conflict.",
        inputSchema={"type": "object", "properties": {
            "node": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["node"]}),

    types.Tool(name="fleet_status",
        description="Health of the shared memory: agents active, claims by verdict, "
                    "open conflicts, conclusions awaiting review, episode outcomes and "
                    "the most common failure signatures.",
        inputSchema={"type": "object", "properties": {"namespace": {"type": "string"}}}),

    types.Tool(name="register_agent",
        description="Operator tool. Register an agent and return its secret token ONCE, "
                    "so that its claimed identity can be verified at the write gate "
                    "instead of trusted. Re-registering rotates the secret, which is "
                    "how a leaked token is revoked. action=revoke disables an identity; "
                    "action=roster lists who may act, without secrets.",
        inputSchema={"type": "object", "properties": {
            "agent_id": {"type": "string"},
            "namespace": {"type": "string"},
            "action": {"type": "string",
                       "description": "register (default) | revoke | roster"},
            "require_auth": {"type": "boolean",
                             "description": "make this namespace reject unregistered agents"}}}),

    types.Tool(name="prune_history",
        description="Operator tool. Delete raw tool-call rows older than the retention "
                    "window, keeping episode rollups and the call sequences of failed "
                    "runs (those are what failure diagnosis reads).",
        inputSchema={"type": "object", "properties": {
            "days": {"type": "integer"},
            "keep_failures": {"type": "boolean"}}}),

    types.Tool(name="fleet_activity",
        description="What the fleet actually DID: tool calls in order, per agent, with "
                    "timing and outcome. Use to see how agents are working — who called "
                    "what, in what sequence, what is slow, what keeps failing, and where "
                    "work passed from one agent to another.",
        inputSchema={"type": "object", "properties": {
            "agent_id": {"type": "string", "description": "omit for the whole fleet"},
            "limit": {"type": "integer"},
            "episode_id": {"type": "string", "description": "for view=detail"},
            "tool": {"type": "string", "description": "for view=detail"},
            "view": {"type": "string",
                     "description": "timeline (default) | usage | handoffs | detail. "
                                    "detail shows what each call RETURNED — the "
                                    "context an agent actually saw before concluding."}}}),

    types.Tool(name="diagnose_failure",
        description="Given an error, find past agent runs that failed the same way and "
                    "what was running at the time. Pure retrieval over the episode "
                    "plane — no model call.",
        inputSchema={"type": "object", "properties": {
            "error_text": {"type": "string"},
            "tool_sequence": {"type": "array", "items": {"type": "string"}}},
            "required": ["error_text"]}),
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
    """Every tool call is recorded on the z-plane before it is dispatched.

    A fleet is only debuggable if you can see what each agent actually did, in
    order, and how long it took — not merely what it concluded. Recording at this
    boundary means no individual tool has to remember to log, and a tool added
    later is captured automatically.

    An agent's episode is opened lazily on its first call and stays open for the
    life of the process, so one agent session is one episode.
    """
    import time as _t
    from . import episodes as _ep

    _agent = os.getenv("TG_AGENT_ID", "default")

    # Identity gates the whole surface, not only writes. Verifying at the gate
    # stopped an agent from asserting under a borrowed name, but it could still
    # READ everything under one — and in a multi-tenant fleet the read side is the
    # confidentiality boundary. Checked once per process and cached: it is the same
    # answer every call, and a failure here must be loud rather than silent.
    if _state.get("identity") is None:
        from . import identity as _id
        from .context_graph import ContextGraph as _CG
        _ns = os.getenv("TG_NAMESPACE", "default")
        try:
            _state["identity"] = _id.check(_CG(), _agent, _ns)
        except Exception as e:
            # never fail closed on an infrastructure error — an unreadable graph is
            # not an authentication failure
            _state["identity"] = {"ok": True, "reason": f"identity check skipped: {e}"}
    # register_agent is exempt or the fleet locks itself out: turning require_auth
    # on for a namespace would block the very tool needed to register anyone in it.
    # Reaching this tool already requires the server's own token, which is the
    # operator boundary.
    if not _state["identity"]["ok"] and name != "register_agent":
        raise PermissionError(
            f"{_state['identity']['reason']}. Register with the register_agent tool "
            f"and set TG_AGENT_ID and TG_AGENT_TOKEN for this agent.")

    if not _state.get("episode"):
        try:
            _state["episode"] = _ep.start_episode(
                f"{_agent} session", agent_id=_agent)
        except Exception:
            _state["episode"] = None

    _t0 = _t.perf_counter()
    _ok, _err, _res = "ok", "", None
    try:
        _res = await _dispatch_tool(name, args)
        return _res
    except Exception as e:
        _ok, _err = "error", f"{type(e).__name__}: {e}"
        raise
    finally:
        if _state.get("episode"):
            # args and result are recorded but truncated; the result is what makes
            # "what did this agent see before it concluded that" answerable.
            # Never let observability raise.
            _ep.log_tool_call(_state["episode"], name, args, outcome=_ok,
                              duration_ms=int((_t.perf_counter() - _t0) * 1000),
                              error_text=_err, result=_res)


def _scope(args: dict) -> str:
    """The namespace this process may act in — from its identity, never its input.

    Found by running real agents: CrewAI happily sent
    `open_conflicts {'namespace': 'auth model'}`, putting its subject into the
    scope parameter. The server honoured it. Since namespace is the
    confidentiality boundary between tenants, an agent naming any namespace could
    read another tenant's conflicts — identity was verified and then allowed to be
    overridden by an argument, which makes the verification pointless.

    The environment is set by whoever launched the process, alongside the token
    that authenticates it; the agent cannot alter it. A mismatched request is
    dropped rather than rejected loudly, because a confused model asking for the
    wrong scope is not an attack and should not fail its task — it simply does
    not get to choose.

    Operator tools that genuinely act across namespaces take the value explicitly
    and do not route through here.
    """
    return os.getenv("TG_NAMESPACE", "default")


async def _dispatch_tool(name: str, args: dict):
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

        # ── fleet: governed writes ──────────────────────────────────────────
        if name == "submit_claim":
            from . import admission
            from .context_graph import ContextGraph
            cg = ContextGraph()
            r = admission.admit(cg, args, namespace=_scope(args),
                                episode_id=_state.get("episode"))
            # record the reasoning chain so a later retraction can walk it
            if r.get("node") and args.get("derived_from"):
                admission.link_derivation(cg, r["node"], args["derived_from"])
            if r["verdict"] == "CONFLICTED":
                r["note"] = ("Both claims are stored and neither is authoritative. "
                             "Do not proceed as though your value were accepted; "
                             "surface the disagreement.")
            return [types.TextContent(type="text", text=json.dumps(r, indent=2))]

        if name == "open_conflicts":
            from . import admission
            from .context_graph import ContextGraph
            return [types.TextContent(type="text", text=json.dumps(
                admission.open_conflicts(ContextGraph(), _scope(args),
                                         args.get("min_severity", "low")),
                indent=2, default=str))]

        if name == "adjudicate":
            from . import admission
            from .context_graph import ContextGraph
            return [types.TextContent(type="text", text=json.dumps(
                admission.adjudicate(ContextGraph(), args["conflict_id"],
                                     args.get("winning_node"),
                                     args.get("resolved_by", "agent")),
                indent=2, default=str))]

        if name == "retract_claim":
            from . import admission
            from .context_graph import ContextGraph
            return [types.TextContent(type="text", text=json.dumps(
                admission.retract(ContextGraph(), args["node"], args.get("reason", "")),
                indent=2, default=str))]

        if name == "fleet_status":
            from . import admission, episodes
            from .context_graph import ContextGraph
            from collections import Counter
            ns = _scope(args)
            g = ContextGraph().g
            claims = [d for _, d in g.nodes(data=True)
                      if d.get("plane") == "claim" and d.get("namespace") == ns]
            out = {
                "namespace": ns,
                "claims": len(claims),
                "by_verdict": dict(Counter(c.get("write_verdict") for c in claims)),
                "submitted_by_agent": dict(Counter(c.get("agent_id") for c in claims)),
                # who ASSERTS each fact — usually a source document, not an agent
                "asserted_by": dict(Counter(
                    c.get("claimant") or c.get("agent_id") for c in claims).most_common(10)),
                "unverified_identity": sum(
                    1 for c in claims if not c.get("identity_verified")),
                "retracted": sum(1 for c in claims if c.get("retracted")),
                # Whether reasoning is actually legible on the graph. derived_from
                # is written only when a conclusion grounds on another conclusion,
                # and the attribution comes from the triple extractor, so the rate
                # is empirical rather than guaranteed. If premises stays near zero
                # the reasoning graph is sparse and trust propagation has nothing
                # to walk, whatever the code is capable of.
                "reasoning_edges": {
                    "premises_derived_from": sum(
                        1 for _, _, e in g.edges(data=True)
                        if e.get("relation") == "derived_from"),
                    "evidence_grounds": sum(
                        1 for _, _, e in g.edges(data=True)
                        if e.get("relation") == "grounds"),
                    "produced": sum(1 for _, _, e in g.edges(data=True)
                                    if e.get("relation") == "produced"),
                    "reasserted": sum(1 for _, _, e in g.edges(data=True)
                                      if e.get("relation") == "reasserted"),
                },
                "adjudications": sum(
                    1 for _, d in g.nodes(data=True)
                    if d.get("plane") == "conflict" and d.get("status") == "RESOLVED"),
                "open_conflicts": len(admission.open_conflicts(ContextGraph(), ns)),
                "needs_review": len(admission.review_queue(ContextGraph(), ns)),
                "episodes": episodes.stats(),
            }
            return [types.TextContent(type="text", text=json.dumps(out, indent=2, default=str))]

        if name == "register_agent":
            from . import identity
            from .context_graph import ContextGraph
            cg = ContextGraph()
            action = args.get("action", "register")
            ns = _scope(args)
            if action == "roster":
                out = {"roster": identity.roster(cg),
                       "requires_auth": identity.requires_auth(cg, ns),
                       "namespace": ns}
            elif action == "revoke":
                out = identity.revoke_agent(cg, args["agent_id"])
            else:
                out = identity.register_agent(cg, args["agent_id"], ns, save=False)
                if args.get("require_auth"):
                    # switching the namespace on is a separate, deliberate step:
                    # register every agent first, or the fleet locks itself out
                    # merge, never replace: the namespace node may already carry
                    # an admit_floor that must survive
                    cg.g.add_node(f"ns:{ns}", plane="namespace")
                    cg.g.nodes[f"ns:{ns}"]["require_auth"] = True
                    out["require_auth"] = True
                cg.save()
            return [types.TextContent(type="text", text=json.dumps(out, indent=2, default=str))]

        if name == "prune_history":
            from . import episodes
            out = episodes.prune(args.get("days"),
                                 bool(args.get("keep_failures", True)))
            return [types.TextContent(type="text", text=json.dumps(out, indent=2))]

        if name == "fleet_activity":
            from . import episodes
            view = args.get("view", "timeline")
            if view == "usage":
                out = {"tool_usage_by_agent": episodes.tool_usage(args.get("agent_id"))}
            elif view == "handoffs":
                out = {"handoffs": episodes.handoffs()}
            elif view == "detail":
                out = {"calls": episodes.call_detail(args.get("episode_id"),
                                                     args.get("tool"),
                                                     int(args.get("limit", 20)))}
            else:
                out = {"timeline": episodes.activity(args.get("agent_id"),
                                                    int(args.get("limit", 50)))}
            return [types.TextContent(type="text", text=json.dumps(out, indent=2, default=str))]

        if name == "diagnose_failure":
            from . import episodes
            hits = episodes.similar_failures(args["error_text"],
                                             args.get("tool_sequence"))
            return [types.TextContent(type="text", text=json.dumps(
                {"matches": hits,
                 "note": "no model involved — retrieval over the episode plane"},
                indent=2, default=str))]

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
