"""TruthGuard as MetaGPT's memory: govern the handoffs, change nothing else.

MetaGPT passes context down a waterfall — PM to architect to engineer to QA —
as messages, and nothing validates the handoff. When the architect specifies
`GET /api/users` and the engineer implements `/users`, that contradiction travels
silently until something 404s at runtime. The agents are not weak; the channel
between them is unguarded, and every extra role makes it worse rather than better.

This subclasses `metagpt.memory.Memory` and injects via `RoleContext`'s
`default_factory`, so MetaGPT's own code is untouched. That matters for the
measurement more than for the engineering: if the agents themselves were modified,
any improvement could be attributed to the modification rather than to governance.

WHAT GETS GATED, and why not everything:

MetaGPT messages are prose and code. Extracting claims from prose needs an LLM
call per message — extra cost, and extraction error would masquerade as governance
error, so a bad result would be unattributable. But `WriteDesign` already emits
STRUCTURED artifacts through typed ActionNodes: "File list", "Data structures and
interfaces", "Project name". Those are the contracts roles must agree on, they are
already machine-readable, and they are exactly where integration failures come
from. So the gate reads artifacts, not prose: a parsing problem instead of an
inference one.

Usage — control and treatment differ by one line:

    from truthguard.metagpt_memory import governed
    role.rc.memory = governed(agent_id="architect", namespace="proj-x")
"""
import json
import os
import re

try:                                     # MetaGPT is optional at import time
    from metagpt.memory import Memory
    from metagpt.schema import Message
except Exception:                        # pragma: no cover - lets tests import
    Memory, Message = object, object


# Artifact keys worth treating as contracts. Each maps to a relation, so that two
# roles asserting different file lists collide on the same subject+relation and the
# gate can see it as a contradiction rather than as two unrelated facts.
_CONTRACT_KEYS = {
    "Project name": ("project", "named"),
    "File list": ("file list", "is"),
    "Data structures and interfaces": ("interfaces", "define"),
    "Program call flow": ("call flow", "is"),
    "Implementation approach": ("approach", "is"),
    "Refined File list": ("file list", "is"),
    "Refined Data structures and interfaces": ("interfaces", "define"),
}

# Signatures inside the mermaid classDiagram: `+method(arg) ReturnType`. These are
# the finest-grained contract MetaGPT produces and the one integration actually
# breaks on, so they are claimed individually rather than as one opaque blob.
_CLASS_RE = re.compile(r"class\s+(\w+)\s*\{([^}]*)\}", re.S)
_METHOD_RE = re.compile(r"^\s*[+\-#]?\s*(\w+)\s*\(([^)]*)\)", re.M)


def _artifact(message) -> dict:
    """Structured payload of a message, or {} if it carries only prose."""
    ic = getattr(message, "instruct_content", None)
    if ic is not None:
        try:
            return ic.model_dump()
        except Exception:
            try:
                return dict(ic)
            except Exception:
                pass
    content = getattr(message, "content", "") or ""
    if content.strip().startswith("{"):
        try:
            return json.loads(content)
        except Exception:
            return {}
    return {}


def _claims_from(artifact: dict) -> list:
    """Turn one design artifact into claims. Pure parsing — no model involved."""
    out = []
    for key, (subject, relation) in _CONTRACT_KEYS.items():
        if key not in artifact or artifact[key] in (None, "", [], {}):
            continue
        val = artifact[key]
        if isinstance(val, (list, tuple)):
            # order must not create a false conflict: ["a.py","b.py"] and
            # ["b.py","a.py"] are the same contract
            val = ", ".join(sorted(str(v) for v in val))

        if key.endswith("Data structures and interfaces"):
            # Per-method signatures ONLY. Claiming the whole diagram as well raises
            # a second conflict saying "the interfaces differ" beside the precise
            # one saying "get_users lost its limit argument" — same disagreement,
            # twice, and the vague copy is the one a reviewer cannot act on. A
            # governance queue earns its alert fatigue exactly this way.
            for cls, body in _CLASS_RE.findall(str(val)):
                for meth, args in _METHOD_RE.findall(body):
                    out.append({
                        "subject": f"{cls}.{meth}".lower(),
                        "relation": "signature",
                        "object": f"({args.strip()})"})
            continue

        out.append({"subject": subject, "relation": relation,
                    "object": str(val)[:400]})
    return out


class TruthGuardMemory(Memory):
    """MetaGPT Memory that routes design contracts through the admission gate.

    `add` is the write path: MetaGPT's own behaviour runs first and unchanged, then
    any contract in the message is offered to shared memory. A CONFLICTED verdict
    is appended to the message content, because a gate the agent cannot see changes
    nothing — the contradiction has to arrive where the next role will read it.

    `get` is the read path: MetaGPT's messages, plus what the fleet has already
    established. Without this an agent still reasons from its own local history and
    the shared graph is write-only.
    """

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def _tg(self):
        mem = getattr(self, "_tg_mem", None)
        if mem is None:
            from .fleet_adapter import FleetMemory
            mem = FleetMemory(
                agent_id=getattr(self, "_tg_agent", None) or os.getenv("TG_AGENT_ID", "metagpt"),
                namespace=getattr(self, "_tg_ns", None) or os.getenv("TG_NAMESPACE", "default"),
                goal="metagpt role")
            object.__setattr__(self, "_tg_mem", mem)
        return mem

    # ── write path ───────────────────────────────────────────────────────────
    def add(self, message):
        super().add(message)                       # MetaGPT behaviour, unchanged
        if not getattr(self, "_tg_enabled", True):
            return
        try:
            claims = _claims_from(_artifact(message))
        except Exception:
            return
        if not claims:
            return

        role = getattr(message, "role", None) or getattr(message, "sent_from", "?")
        conflicts = []
        for c in claims:
            try:
                v = self._tg().remember(
                    c["subject"], c["relation"], c["object"],
                    confidence=0.9,
                    # the ROLE that decided is the claimant: a disagreement then
                    # reads "architect vs engineer", which is the thing a reviewer
                    # can actually act on
                    claimant=str(role))
            except Exception:
                continue
            if v.get("verdict") == "CONFLICTED":
                conflicts.append(f"{c['subject']} {c['relation']}: {v.get('reason')}")

        if conflicts:
            # Surface it INTO the conversation. A verdict recorded only in the graph
            # is invisible to the waterfall and cannot change what the next role
            # builds — which is the entire point of catching it at decision time.
            warn = ("\n\n[TRUTHGUARD] This contradicts a decision already recorded:\n"
                    + "\n".join(f"  - {c}" for c in conflicts)
                    + "\nResolve it before building on either version.")
            try:
                message.content = (message.content or "") + warn
            except Exception:
                pass

    # ── read path ────────────────────────────────────────────────────────────
    def get(self, k=0):
        msgs = super().get(k)
        if not getattr(self, "_tg_enabled", True):
            return msgs
        try:
            facts = self._tg().recall("project interfaces file list approach", k=8)
            disputed = self._tg().conflicts()
        except Exception:
            return msgs
        if not facts and not disputed:
            return msgs

        lines = ["[SHARED MEMORY] Decisions already established by this team:"]
        for f in facts:
            flag = "  DISPUTED" if f.get("disputed") else ""
            lines.append(f"  - {f['subject']} {f['relation']} {f['object']}"
                         f" (per {f['claimant']}){flag}")
        for d in disputed:
            vals = " vs ".join(f"{x.get('claimant')}:{x.get('value')}" for x in d["claims"])
            lines.append(f"  ! UNRESOLVED {d['subject']}: {vals}")
        lines.append("Build on these. Do not silently contradict them.")

        try:
            msgs = list(msgs) + [Message(content="\n".join(lines), role="shared_memory")]
        except Exception:
            pass
        return msgs


def governed(agent_id: str, namespace: str = None, enabled: bool = True):
    """Build a memory for one role. `enabled=False` is the control arm — same
    object, same code path, gate off — so a comparison isolates governance rather
    than comparing two different programs."""
    m = TruthGuardMemory()
    object.__setattr__(m, "_tg_agent", agent_id)
    object.__setattr__(m, "_tg_ns", namespace or os.getenv("TG_NAMESPACE", "default"))
    object.__setattr__(m, "_tg_enabled", enabled)
    return m
