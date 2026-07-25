"""TruthGuard Studio API — FastAPI wrapper over the same functions the MCP
tools call. Zero new logic; the UI and MCP can't drift apart.

Run:  uvicorn truthguard.api:app --port 7788
Docs: http://127.0.0.1:7788/docs
"""
import os
import json
import shutil
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TG_STORAGE_DIR", os.path.join(_ROOT, "storage", "truthguard"))
os.environ.setdefault("TG_CORPUS_DIR", os.path.join(_ROOT, "corpus"))

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── public exposure controls ────────────────────────────────────────────────
# Unset TG_API_TOKEN (the default) means purely local use: no auth, writes on,
# behaves exactly as before. Setting it switches the server into "exposed" mode
# for tunnelling: every request needs the token, and writes are off unless
# explicitly re-enabled. Fail closed, because the graph holds private code and
# chat transcripts.
API_TOKEN = os.getenv("TG_API_TOKEN", "").strip()
EXPOSED = bool(API_TOKEN)
# TG_READONLY is for a public deployment with no token at all: anyone may read
# and ask, nobody may write or touch credentials. Without it, "no token" means
# local mode, where writes are on — which would be wrong on a public host.
READONLY = os.getenv("TG_READONLY", "").strip() == "1"
ALLOW_WRITE = os.getenv("TG_ALLOW_WRITE", "").strip() == "1" and not READONLY

# Routes that mutate state or touch credentials. Blocked when exposed unless
# TG_ALLOW_WRITE=1 is set deliberately.
_WRITE_PREFIXES = ("/ingest", "/config")

app = FastAPI(title="TruthGuard Studio API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=(["*"] if not EXPOSED else
                   [o for o in os.getenv("TG_ALLOWED_ORIGINS", "").split(",") if o]
                   or ["https://truthguard-pink.vercel.app"]),
    allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def _guard(request: Request, call_next):
    # Read-only mode applies with or without a token, so a public deployment
    # cannot be written to even though visitors need no credentials.
    if READONLY and request.method != "GET" \
            and request.url.path.startswith(_WRITE_PREFIXES):
        return JSONResponse(
            {"error": "this deployment is read-only",
             "hint": "run your own instance to ingest or change providers"}, status_code=403)
    if not EXPOSED:
        return await call_next(request)
    # CORS preflight carries no credentials by design — let the CORS middleware
    # answer it, or the browser never gets far enough to send the real request.
    if request.method == "OPTIONS":
        return await call_next(request)
    path = request.url.path
    # The token may arrive as a header or a query param — iframes and <script>
    # loads for the graph view cannot set headers.
    from_query = request.query_params.get("token", "")
    supplied = (request.headers.get("x-tg-token") or from_query
                or request.cookies.get("tg_token", ""))
    if supplied != API_TOKEN:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if request.method != "GET" and path.startswith(_WRITE_PREFIXES) and not ALLOW_WRITE:
        return JSONResponse(
            {"error": "this endpoint is disabled while the server is publicly exposed",
             "hint": "set TG_ALLOW_WRITE=1 to re-enable"}, status_code=403)
    resp = await call_next(request)
    # A graph view loaded with ?token=... fetches its data with a plain relative
    # request that carries no header, so hand it a cookie to travel on.
    if from_query == API_TOKEN:
        resp.set_cookie("tg_token", API_TOKEN, httponly=True,
                        secure=True, samesite="none", max_age=86400)
    return resp

_state = {"store": None, "llm": None}


def _store():
    """Chunk store, reloaded when another process has rewritten the index.

    The MCP server and this API are separate processes sharing one storage dir,
    so an ingest in an MCP client is invisible to this process's in-memory copy.
    Comparing chunks.json's mtime makes the shared graph actually shared in both
    directions: ingest anywhere, ask anywhere, no restart.
    """
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


@app.on_event("startup")
def _warm():
    """Load the embedder and cross-encoder before the first request.

    Both are lazy, so without this the first question pays ~54s of model
    loading — exactly the question a judge or a first-time visitor asks.
    Doing it at startup moves that cost to boot, where nobody is waiting.
    """
    import threading

    def warm():
        try:
            _store().max_similarity("warmup")          # loads the embedder
            from .retrieve import _rerank
            ids = list(_store().by_id)[:1]
            if ids:
                _rerank("warmup", [(ids[0], 1.0)], _store())   # loads cross-encoder
            print("[truthguard] models warm", flush=True)
        except Exception as e:                          # never block startup
            print(f"[truthguard] warmup skipped: {e}", flush=True)

    threading.Thread(target=warm, daemon=True).start()


def _export():
    try:
        from .full3d import export_full
        export_full()
    except Exception:
        pass


class AskBody(BaseModel):
    question: str
    followup: str | None = None
    baseline: bool = False
    # Interactive callers default to the low-latency path; the eval harness
    # calls the controller directly and keeps the full retrieval.
    fast: bool = True


@app.post("/ask")
def ask(body: AskBody):
    from .controller import ask as _ask
    try:
        r = _ask(_store(), _llm(), body.question,
                 baseline=body.baseline, followup=body.followup, fast=body.fast)
    except Exception as e:
        msg = ("LLM provider is rate-limited right now (a benchmark may be "
               "running on the same key). Try again in a minute."
               if "429" in str(e) else f"{type(e).__name__}: {e}")
        return {"kind": "error", "text": msg, "confidence": None,
                "band": None, "citations": [], "trace": [{"step": "llm_unavailable"}]}
    _export()
    return r


# ── async ask ───────────────────────────────────────────────────────────────
# A full /ask takes minutes, and holding one HTTP connection open that long does
# not survive a consumer link behind a tunnel — the tunnel's control session
# drops and the request dies with it. So the work runs in a thread and the
# client polls: every request is short, and a dropped connection costs one poll
# instead of the whole answer.
_jobs: dict = {}


@app.post("/ask_async")
def ask_async(body: AskBody):
    import threading, uuid
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "result": None}

    def run():
        try:
            _jobs[job_id] = {"status": "done", "result": ask(body)}
        except Exception as e:
            _jobs[job_id] = {"status": "error",
                             "result": {"kind": "error", "text": f"{type(e).__name__}: {e}",
                                        "trace": [{"step": "failed"}], "citations": []}}

    threading.Thread(target=run, daemon=True).start()
    # Keep the table from growing without bound over a long demo session.
    if len(_jobs) > 50:
        for k in list(_jobs)[:-50]:
            _jobs.pop(k, None)
    return {"job_id": job_id, "status": "running"}


@app.get("/ask_job/{job_id}")
def ask_job(job_id: str):
    return _jobs.get(job_id, {"status": "unknown", "result": None})


class QueryBody(BaseModel):
    question: str


@app.post("/recall")
def recall(body: QueryBody):
    from .recall import recall as _recall
    return _recall(body.question)


@app.post("/get_context")
def get_context(body: QueryBody):
    from .recall import get_context as _gc
    return {"context": _gc(body.question)}


class GraphQueryBody(BaseModel):
    command: str          # context | impact | find | edit_plan | path | report
    name: str = ""
    target: str = ""


@app.post("/graph_query")
def graph_query(body: GraphQueryBody):
    from . import graph_query as gq
    if body.command == "report":
        return gq.report()
    if body.command == "path":
        return gq.path(body.name, body.target)
    fn = {"context": gq.context, "impact": gq.impact,
          "find": gq.find, "edit_plan": gq.edit_plan}.get(body.command)
    return fn(body.name) if fn else {"error": f"unknown command {body.command}"}


@app.post("/ingest/document")
async def ingest_document(file: UploadFile = File(...)):
    os.makedirs(config.CORPUS_DIR, exist_ok=True)
    dst = os.path.join(config.CORPUS_DIR, os.path.basename(file.filename))
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)
    from .pipeline import ingest_corpus
    from .chunk_store import build_index
    chunks, report = ingest_corpus()
    engine = build_index()
    _state["store"] = None
    _export()
    return {"file": file.filename, "total_chunks": report["total_chunks"],
            "engine": engine}


class PathBody(BaseModel):
    path: str
    chat_path: str | None = None


@app.post("/ingest/project")
def ingest_project(body: PathBody):
    from .ingest_all import ingest_project as _ip
    r = _ip(body.path, body.chat_path)
    _state["store"] = None
    return r


@app.post("/ingest/chat")
def ingest_chat(body: PathBody):
    from .import_chat import import_chat
    return import_chat(body.path)


@app.get("/graph3d")
def graph3d():
    p = os.path.join(ROOT, "graph3d_data.json")
    return FileResponse(p) if os.path.exists(p) else JSONResponse(
        {"nodes": [], "links": []})


# The graph views used to be served by a separate SimpleHTTPServer on :7787,
# which exposed the whole project directory. Serving them here instead means
# one tunnel, and only these files are reachable.
_GRAPH_VIEWS = {
    "FULL_3plane_clean.html", "full_3plane_3d.html", "full_everything_3d.html",
    "graph3d.html", "graph3d_live.html", "unified_dg_3d.html",
    "context_graph.html", "context_graph_dg.html", "chat_graph_3d.html",
    "layer_graphs.html", "layers_view.html",
    # the views fetch this relatively, so it has to resolve under /graph/ too
    "graph3d_data.json",
}


@app.get("/graph/{name}")
def graph_view(name: str):
    if name not in _GRAPH_VIEWS:
        return JSONResponse({"error": "unknown view", "available": sorted(_GRAPH_VIEWS)},
                            status_code=404)
    p = os.path.join(ROOT, name)
    if not os.path.exists(p):
        return JSONResponse({"error": f"{name} has not been generated yet"}, status_code=404)
    return FileResponse(p)


@app.get("/stats")
def stats():
    from .context_graph import ContextGraph
    from collections import Counter
    g = ContextGraph().g
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
            "turns": g.graph.get("n_turns", 0),
            "planes": dict(Counter(d.get("plane") for _, d in g.nodes(data=True)))}


@app.get("/benchmarks")
def benchmarks():
    out = {}
    ev = os.path.join(ROOT, "eval")
    for name in ("locomo_results", "longmemeval_results", "h2h_results"):
        p = os.path.join(ev, name + ".json")
        if os.path.exists(p):
            out[name] = json.load(open(p, encoding="utf-8"))
    slices = []
    for f in sorted(os.listdir(ev)):
        if f.startswith("longmemeval_slice_"):
            slices.append(json.load(open(os.path.join(ev, f), encoding="utf-8")))
    if slices:
        qn = sum(s["qa_n"] for s in slices)
        out["longmemeval_slices"] = {
            "slices_done": len(slices),
            "recall@10": sum(s["recall@10"] * s["recall_n"] for s in slices)
                         / max(sum(s["recall_n"] for s in slices), 1),
            "qa": (sum(s["qa_sum"] for s in slices) / qn) if qn else None,
            "qa_n": qn}
    return out


_ENV_PATH = os.path.join(ROOT, ".env")
_MASK = lambda v: (v[:6] + "…" + v[-4:]) if v and len(v) > 12 else ("set" if v else "")


class ConfigBody(BaseModel):
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_provider: str | None = None
    mistral_ocr_api_key: str | None = None


@app.get("/config")
def get_config():
    """Masked view of the current provider settings — never returns full keys."""
    from . import config as cfg
    return {
        "llm_provider": cfg.LLM_PROVIDER,
        "llm_base_url": cfg.LLM_BASE_URL,
        "llm_model": cfg.LLM_MODEL,
        "llm_api_key": _MASK(cfg.LLM_API_KEY),
        "mistral_ocr_api_key": _MASK(cfg.MISTRAL_OCR_API_KEY),
        "llm_ready": bool(cfg.LLM_API_KEY),
        "ocr_ready": bool(cfg.MISTRAL_OCR_API_KEY),
    }


@app.get("/models")
def list_models(base_url: str = "", api_key: str = ""):
    """Ask the configured provider what models it serves, so the settings panel
    can offer a list instead of making the user type an exact id from memory.
    Falls back to an empty list if the provider does not support /v1/models."""
    import httpx
    from . import config as cfg
    base = (base_url or cfg.LLM_BASE_URL).rstrip("/")
    key = api_key or cfg.LLM_API_KEY
    try:
        r = httpx.get(f"{base}/models", timeout=10,
                      headers={"Authorization": f"Bearer {key}"} if key else {})
        data = r.json().get("data", [])
        names = sorted(m.get("id", "") for m in data if m.get("id"))
        return {"models": names, "base_url": base}
    except Exception as e:
        return {"models": [], "base_url": base, "error": f"{type(e).__name__}: {e}"}


@app.post("/config")
def set_config(body: ConfigBody):
    """Write provider settings to .env and hot-reload them. Local use only —
    the server binds to 127.0.0.1 and keys are never echoed back."""
    import re
    from . import config as cfg
    updates = {
        "LLM_API_KEY": body.llm_api_key, "LLM_BASE_URL": body.llm_base_url,
        "LLM_MODEL": body.llm_model, "LLM_PROVIDER": body.llm_provider,
        "MISTRAL_OCR_API_KEY": body.mistral_ocr_api_key,
    }
    updates = {k: v for k, v in updates.items() if v}
    if not updates:
        return {"ok": False, "error": "nothing to update"}
    env = open(_ENV_PATH, encoding="utf-8").read() if os.path.exists(_ENV_PATH) else ""
    for k, v in updates.items():
        env = (re.sub(rf"^{k}=.*$", f"{k}={v}", env, flags=re.M)
               if re.search(rf"^{k}=", env, re.M) else env.rstrip("\n") + f"\n{k}={v}\n")
        setattr(cfg, k, v)              # hot-reload for this process
        os.environ[k] = v
    with open(_ENV_PATH, "w", encoding="utf-8") as f:
        f.write(env)
    _state["llm"] = None                # force a fresh client on next call
    return {"ok": True, "updated": sorted(updates)}


@app.get("/")
def studio():
    return FileResponse(os.path.join(ROOT, "studio.html"))


@app.get("/architecture.html")
def architecture():
    return FileResponse(os.path.join(ROOT, "architecture.html"))


@app.get("/about.html")
def about():
    return FileResponse(os.path.join(ROOT, "about.html"))
