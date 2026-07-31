"""A page a human can actually use to govern the fleet.

Everything the gate does ends in "a human decides", and until now that human had
no way to look. Fifteen conflicts existed after the 30-agent run and the only way
to see them was a Python script. A governance system nobody can read is a
governance system nobody uses.

Deliberately one file, no build step, no framework: it has to still run in six
months without an npm install. Served by the stdlib.

What it shows, in the order a reviewer needs it:

  conflicts     what is disputed, both sides, WHO asserts each — the queue
  blast radius  what gets flagged if you retract, shown BEFORE you decide
  claims        what the fleet currently believes
  agents        reliability, worst first

The blast radius preview is the part that matters. Adjudicating without it means
retracting a premise and only then discovering that six conclusions rested on it.

    python -m truthguard.viewer            # http://127.0.0.1:7788
"""
import html
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

PORT = int(os.getenv("TG_VIEWER_PORT", "7788"))


def _graph(ns):
    from .context_graph import ContextGraph
    return ContextGraph()


def _state(ns: str) -> dict:
    from . import admission, trust, labels
    cg = _graph(ns)
    claims = [(n, d) for n, d in cg.g.nodes(data=True)
              if d.get("plane") == "claim" and d.get("namespace") == ns]
    conflicts = admission.open_conflicts(cg, ns)

    # Blast radius is computed per conflict side, before anything is retracted —
    # a reviewer needs to know the cost of a ruling while deciding it.
    for c in conflicts:
        for side in c["claims"]:
            side["blast"] = _blast(cg, side["node"])

    return {
        "namespace": ns,
        "conflicts": conflicts,
        "claims": [{"id": n, "subject": d.get("subject"), "relation": d.get("relation"),
                    "object": d.get("object"), "claimant": d.get("claimant") or d.get("agent_id"),
                    "agent": d.get("agent_id"), "verdict": d.get("write_verdict"),
                    "retracted": bool(d.get("retracted")),
                    "needs_review": bool(d.get("needs_review")),
                    "review_reason": d.get("review_reason"),
                    "sensitivity": labels.effective_sensitivity(cg, n),
                    "confidence": d.get("confidence")}
                   for n, d in claims],
        "review": admission.review_queue(cg, ns),
        "agents": trust.fleet_scores(cg, ns),
        "resolved": sum(1 for _, d in cg.g.nodes(data=True)
                        if d.get("plane") == "conflict" and d.get("status") == "RESOLVED"),
    }


def _blast(cg, node: str) -> list:
    """Which conclusions would be flagged if this claim were retracted.

    Walks the same edges propagation walks, without mutating anything — a dry run
    of the decision the reviewer is about to make.
    """
    out, seen, queue = [], {node}, [(node, 0)]
    while queue:
        cur, depth = queue.pop(0)
        if depth >= 6:
            continue
        for dep in cg.g.predecessors(cur):
            if dep in seen:
                continue
            rel = cg.g.edges[dep, cur].get("relation")
            if rel not in ("derived_from", "informed_by"):
                continue
            seen.add(dep)
            d = cg.g.nodes[dep]
            out.append({"id": dep, "subject": d.get("subject"),
                        "object": str(d.get("object"))[:60],
                        "strength": "strong" if rel == "derived_from" else "weak"})
            queue.append((dep, depth + 1))
    return out


PAGE = """<!doctype html><meta charset=utf-8><title>TruthGuard — fleet governance</title>
<style>
*{box-sizing:border-box}body{font:14px/1.5 ui-sans-serif,system-ui,sans-serif;margin:0;
background:#0d1117;color:#c9d1d9}header{padding:14px 20px;border-bottom:1px solid #30363d;
display:flex;gap:18px;align-items:center}h1{font-size:15px;margin:0;font-weight:600}
.ns{color:#8b949e}.tabs{display:flex;gap:4px;margin-left:auto}
.tabs button{background:#161b22;color:#8b949e;border:1px solid #30363d;padding:5px 12px;
border-radius:6px;cursor:pointer;font:inherit}.tabs button.on{background:#1f6feb;color:#fff}
main{padding:18px 20px;max-width:1100px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin:0 0 12px}
.sub{font-weight:600;color:#e6edf3}.muted{color:#8b949e;font-size:12px}
.side{border-left:3px solid #30363d;padding:8px 12px;margin:8px 0;background:#0d1117;border-radius:0 6px 6px 0}
.side.win{border-color:#2ea043}.val{font-family:ui-monospace,monospace;color:#79c0ff}
button.act{background:#238636;color:#fff;border:0;padding:5px 12px;border-radius:6px;
cursor:pointer;font:inherit;margin-right:6px}button.act.alt{background:#30363d}
table{width:100%;border-collapse:collapse}td,th{padding:6px 8px;border-bottom:1px solid #21262d;
text-align:left;vertical-align:top}th{color:#8b949e;font-weight:500;font-size:12px}
.tag{font-size:11px;padding:1px 7px;border-radius:10px;border:1px solid #30363d}
.CONFLICTED{background:#3d1d1d;color:#ff7b72}.ACCEPTED{background:#132e1a;color:#3fb950}
.QUARANTINED{background:#3d2f10;color:#d29922}.retracted{opacity:.45;text-decoration:line-through}
.blast{font-size:12px;color:#d29922;margin-top:5px}.empty{color:#8b949e;padding:22px;text-align:center}
.b-unreliable{color:#ff7b72}.b-questionable{color:#d29922}.b-unproven{color:#8b949e}.b-reliable{color:#3fb950}
</style>
<header><h1>TruthGuard</h1><span class=ns id=ns></span>
<span class=tabs>
<button class=on onclick="show('conflicts',this)">Conflicts</button>
<button onclick="show('claims',this)">Claims</button>
<button onclick="show('review',this)">Review</button>
<button onclick="show('agents',this)">Agents</button>
</span></header><main id=app>loading…</main>
<script>
let S={},TAB='conflicts';
const esc=s=>String(s??'').replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
async function load(){S=await (await fetch('/api/state'+location.search)).json();
 document.getElementById('ns').textContent=S.namespace+' · '+S.conflicts.length+' open · '+S.resolved+' resolved';render()}
function show(t,b){TAB=t;document.querySelectorAll('.tabs button').forEach(x=>x.className='');b.className='on';render()}
async function rule(cid,win){await fetch('/api/adjudicate',{method:'POST',
 headers:{'content-type':'application/json'},body:JSON.stringify({conflict:cid,winner:win,namespace:S.namespace})});load()}
function render(){const a=document.getElementById('app');
 if(TAB=='conflicts')a.innerHTML=S.conflicts.length?S.conflicts.map(c=>`<div class=card>
   <div class=sub>${esc(c.subject)} <span class=muted>${esc(c.relation||'')} · ${esc(c.severity||'')}</span></div>
   ${c.claims.map(s=>`<div class=side><b>${esc(s.claimant)}</b> says <span class=val>${esc(s.value)}</span>
     <span class=muted>(conf ${s.confidence??'?'} · via ${esc(s.submitted_by||'?')})</span>
     ${s.blast&&s.blast.length?`<div class=blast>⚠ retracting this flags ${s.blast.length}: `+
       s.blast.map(b=>esc(b.subject)+' ('+b.strength+')').join(', ')+`</div>`:
       `<div class=blast style=color:#8b949e>nothing downstream</div>`}
     <div style=margin-top:8px><button class=act onclick="rule('${c.conflict}','${s.node}')">This one holds</button></div>
   </div>`).join('')}</div>`).join(''):'<div class=empty>No open conflicts.</div>';
 if(TAB=='claims')a.innerHTML=`<div class=card><table><tr><th>subject<th>value<th>asserted by<th>verdict<th>label</tr>
   ${S.claims.map(c=>`<tr class="${c.retracted?'retracted':''}"><td>${esc(c.subject)} <span class=muted>${esc(c.relation)}</span>
   <td class=val>${esc(c.object)}<td>${esc(c.claimant)}<td><span class="tag ${c.verdict}">${esc(c.verdict)}</span>
   <td class=muted>${esc(c.sensitivity)}</tr>`).join('')}</table></div>`;
 if(TAB=='review')a.innerHTML=S.review.length?`<div class=card><table><tr><th>conclusion<th>why</tr>
   ${S.review.map(r=>`<tr><td>${esc(r.subject)}<td class=muted>${esc(r.reason)}</tr>`).join('')}</table></div>`
   :'<div class=empty>Nothing awaiting review.</div>';
 if(TAB=='agents')a.innerHTML=`<div class=card><table><tr><th>agent<th>score<th>band<th>claims<th>signals</tr>
   ${S.agents.map(g=>`<tr><td>${esc(g.agent_id)}<td>${g.score}<td class="b-${g.band}">${g.band}
   <td>${g.claims}<td class=muted>${esc(JSON.stringify(g.signals))}</tr>`).join('')}</table></div>`}
load();setInterval(load,5000);
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        ns = (parse_qs(u.query).get("namespace") or
              [os.getenv("TG_NAMESPACE", "default")])[0]
        if u.path == "/api/state":
            try:
                return self._send(json.dumps(_state(ns), default=str))
            except Exception as e:
                return self._send(json.dumps({"error": str(e), "namespace": ns,
                                              "conflicts": [], "claims": [],
                                              "review": [], "agents": [],
                                              "resolved": 0}))
        return self._send(PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or "{}")
        if urlparse(self.path).path == "/api/adjudicate":
            from . import admission
            cg = _graph(body.get("namespace"))
            r = admission.adjudicate(cg, body["conflict"],
                                     winning_node=body.get("winner"),
                                     resolved_by=os.getenv("TG_REVIEWER", "reviewer"))
            return self._send(json.dumps(r, default=str))
        self._send(json.dumps({"error": "unknown"}))


def serve(port: int = PORT):
    print(f"TruthGuard viewer  ->  http://127.0.0.1:{port}"
          f"?namespace={os.getenv('TG_NAMESPACE','default')}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    serve()
