"""TruthGuard over HTTP: one shared server instead of one process per agent.

The stdio server works and is what every agent has used so far, but it has a
shape problem that only appears at scale. Thirty agents meant thirty TruthGuard
processes against one SQLite file, each spawning an interpreter, loading the
graph, and holding a connection. It survived — 150/150 claims, no losses — but
that is a demo topology, not a deployment.

It also excludes anything that is not a same-machine Python child process:
remote agents, hosted runners, Node and TypeScript frameworks whose MCP client
speaks only HTTP, and agentic-fleet, whose MCP tool is HTTP-only by construction.

WHERE IDENTITY MOVES, AND WHY THAT IS THE HARD PART:

Over stdio the agent's token sat in a child process's environment. The model
never saw it and could not alter it — the credential was outside the agent's
reach by construction. Over HTTP it becomes a bearer token the client sends, and
the environment stops protecting it.

So this server derives identity from the Authorization header and NOTHING else.
An `agent_id` in a tool argument is ignored exactly as `namespace` already is,
because a caller that can name itself can name anyone. The token is matched
against registered agents via identity.verify, and an unmatched token is refused
before any tool runs rather than degraded to a default identity — on a network
port, failing open means the graph is readable by anything that can reach it.

    python -m truthguard.http_server                 # 127.0.0.1:7799
    TG_HTTP_HOST=0.0.0.0 python -m truthguard.http_server

Client side:
    {"transport": "http", "url": "http://host:7799/mcp/",   # trailing slash
     "headers": {"Authorization": "Bearer <the agent's token>"}}
"""
import contextlib
import os

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount

from . import mcp_server

HOST = os.getenv("TG_HTTP_HOST", "127.0.0.1")
PORT = int(os.getenv("TG_HTTP_PORT", "7799"))

# Off by default. An unauthenticated port serving a governed graph is worse than
# no governance, because it looks governed. Turning it off is an explicit act.
REQUIRE_AUTH = os.getenv("TG_HTTP_REQUIRE_AUTH", "1") != "0"


def _identify(request):
    """Resolve the caller from its bearer token. Returns (agent_id, error).

    The token is the ONLY input consulted. Over stdio the environment carried
    identity and the agent could not reach it; over HTTP the client controls
    every field it sends, so anything the client can name cannot be trusted to
    name it.
    """
    from .context_graph import ContextGraph
    from . import identity

    auth = request.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not token:
        return None, "missing Authorization: Bearer <token>"

    cg = ContextGraph()
    for n, d in cg.g.nodes(data=True):
        if d.get("plane") != "agent" or not d.get("token_hash"):
            continue
        aid = d.get("agent_id")
        v = identity.verify(cg, aid, token)
        if v.get("verified"):
            return aid, None
    # Deliberately not "which agent failed" — a caller probing tokens should not
    # learn whether an id exists.
    return None, "token does not match any registered agent"


class IdentityMiddleware:
    """Authenticate once per request and pin the identity for the tools.

    The tool layer reads TG_AGENT_ID from the environment, which is process-wide
    and therefore wrong for a shared server handling several agents. Setting it
    per request works only because each request is handled to completion before
    the next on a single worker; a multi-worker deployment needs the identity
    threaded through the call instead, and that is a real change, not a config.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/mcp"):
            return await self.app(scope, receive, send)

        if REQUIRE_AUTH:
            from starlette.requests import Request
            agent_id, err = _identify(Request(scope, receive))
            if err:
                r = JSONResponse({"error": err}, status_code=401)
                return await r(scope, receive, send)
            os.environ["TG_AGENT_ID"] = agent_id
            # Cleared so a tool cannot fall back to a token the caller never sent.
            os.environ.pop("TG_AGENT_TOKEN", None)
        return await self.app(scope, receive, send)


async def health(_request):
    from .context_graph import ContextGraph
    try:
        n = ContextGraph().g.number_of_nodes()
        return JSONResponse({"status": "ok", "nodes": n,
                             "namespace": os.getenv("TG_NAMESPACE", "default"),
                             "auth_required": REQUIRE_AUTH,
                             "tools": len(mcp_server.TOOLS)})
    except Exception as e:
        return JSONResponse({"status": "degraded", "error": str(e)[:200]}, 503)


def build_app() -> Starlette:
    manager = StreamableHTTPSessionManager(app=mcp_server.app, json_response=False,
                                           stateless=True)

    async def handle(scope, receive, send):
        await manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with manager.run():
            yield

    # The endpoint is /mcp/ WITH the trailing slash. Starlette's Mount only
    # matches the slashed form and 307s the bare path; a client that does not
    # re-send its body on redirect then sees an empty reply and reads it as a
    # broken server. redirect_slashes is left off so that failure is a clean 404
    # rather than a silent empty 307.
    app = Starlette(
        routes=[Route("/health", health), Mount("/mcp", app=handle)],
        lifespan=lifespan,
    )
    app.router.redirect_slashes = False
    return app


def serve(host: str = HOST, port: int = PORT):
    import uvicorn
    app = IdentityMiddleware(build_app())
    print(f"TruthGuard MCP over HTTP -> http://{host}:{port}/mcp/")
    print(f"  auth required : {REQUIRE_AUTH}")
    print(f"  namespace     : {os.getenv('TG_NAMESPACE', 'default')}")
    print(f"  tools         : {len(mcp_server.TOOLS)}")
    if not REQUIRE_AUTH:
        print("  WARNING: auth disabled — anything reaching this port can write "
              "to the graph")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    serve()
