"""One governance surface, thin bindings per framework.

Copied in shape from Microsoft's Agent Governance Toolkit (MIT), which ships 15
frameworks across 4 languages without writing 60 integrations: there is a single
enforcement surface, and each "adapter" is a thin translation from a framework's
own callback shape into it.

Their surface has six points because they govern EXECUTION — input/output,
pre/post model call, pre/post tool call, plus an escalate seam. Governing BELIEF
needs fewer:

    on_conclusion(subject, relation, object, ...)  -> gate the write
    on_context(query)                              -> serve governed reads

Everything TruthGuard does hangs off those two. An adapter's whole job is to find
where a framework produces a conclusion and where it reads context, and wire them.

The other thing worth copying is that their adapters bind a framework's NATIVE
hook API rather than wrapping objects one by one. CrewAI exposes global
`before_tool_call` / `after_tool_call` hooks, so one registration governs every
agent in every crew — no per-agent MCP server, no per-agent surgery. The earlier
fleet harness spawned 30 MCP server processes to achieve less than this does in
one call.

    from truthguard.adapters import crewai_governance
    gov = crewai_governance(agent_id="researcher", namespace="proj").register()
    ...
    gov.unregister()
"""
import os


class GovernanceSurface:
    """The two hooks every adapter binds to. Framework-agnostic by design."""

    def __init__(self, agent_id: str = None, namespace: str = None,
                 token: str = None, observe_only: bool = False):
        self.agent_id = agent_id or os.getenv("TG_AGENT_ID", "default")
        self.namespace = namespace or os.getenv("TG_NAMESPACE", "default")
        self.token = token if token is not None else os.getenv("TG_AGENT_TOKEN", "")
        self.observe_only = observe_only
        self._mem = None
        self.stats = {"conclusions": 0, "conflicts": 0, "denied": 0, "reads": 0}

    @property
    def mem(self):
        if self._mem is None:
            from .fleet_adapter import FleetMemory
            self._mem = FleetMemory(self.agent_id, self.namespace, self.token,
                                    goal=f"{self.agent_id} via adapter")
        return self._mem

    def on_conclusion(self, subject, relation, object, claimant=None,
                      confidence=0.8, derived_from=None, sensitivity=None) -> dict:
        """A conclusion was reached. Returns the gate's verdict.

        Never raises: an adapter sits on a framework's hot path, and a governance
        layer that can crash the agent it governs will be switched off.
        """
        try:
            v = self.mem.remember(subject, relation, object, confidence=confidence,
                                  claimant=claimant, derived_from=derived_from or [])
            self.stats["conclusions"] += 1
            if v.get("verdict") == "CONFLICTED":
                self.stats["conflicts"] += 1
            if v.get("verdict") == "DENIED":
                self.stats["denied"] += 1
            return v
        except Exception as e:
            return {"verdict": "ERROR", "reason": f"{type(e).__name__}: {e}"}

    def on_context(self, query: str, k: int = 5) -> list:
        """What the fleet already believes about this, for injection into a prompt."""
        try:
            self.stats["reads"] += 1
            return self.mem.recall(query, k=k)
        except Exception:
            return []

    def as_prompt_block(self, query: str) -> str:
        """Governed context as text an agent can actually read. Empty when there is
        nothing established, so a first-mover is not handed a misleading header."""
        facts = self.on_context(query)
        if not facts:
            return ""
        lines = ["[SHARED MEMORY] Already established by this fleet:"]
        for f in facts:
            flag = "  (DISPUTED)" if f.get("disputed") else ""
            lines.append(f"  - {f['subject']} {f['relation']} {f['object']}"
                         f" (per {f['claimant']}){flag}")
        return "\n".join(lines)


# ── CrewAI ───────────────────────────────────────────────────────────────────

class CrewAIGovernance:
    """Binds CrewAI's global hooks. One register() governs every crew.

    Gates TOOL RESULTS rather than LLM output. A tool result is a fact the agent
    obtained — "the file says X", "the API returned Y" — which is what other
    agents will build on. LLM output is reasoning about those facts and is far
    noisier to claim, so gating it would fill shared memory with prose.
    """

    def __init__(self, surface: GovernanceSurface, gate_tools=None):
        self.s = surface
        # None means every tool. Naming tools narrows what becomes a claim, which
        # matters because most tools return chatter rather than facts.
        self.gate_tools = set(gate_tools) if gate_tools else None
        self._registered = False

    def register(self):
        from crewai.hooks import (register_after_tool_call_hook,
                                  register_before_llm_call_hook)
        register_after_tool_call_hook(self._after_tool)
        register_before_llm_call_hook(self._before_llm)
        self._registered = True
        return self

    def unregister(self):
        try:
            from crewai.hooks import (unregister_after_tool_call_hook,
                                      unregister_before_llm_call_hook)
            unregister_after_tool_call_hook(self._after_tool)
            unregister_before_llm_call_hook(self._before_llm)
        except Exception:
            pass
        self._registered = False
        return self

    def _after_tool(self, ctx):
        """A tool returned. Record what it established, and flag a contradiction
        back into the result so the agent sees it now rather than at merge time."""
        try:
            tool = getattr(ctx, "tool_name", None) or "tool"
            if self.gate_tools and tool not in self.gate_tools:
                return
            result = str(getattr(ctx, "tool_result", "") or "")[:400]
            if not result.strip():
                return
            v = self.s.on_conclusion(subject=tool, relation="returned",
                                     object=result, claimant=tool)
            if v.get("verdict") == "CONFLICTED" and not self.s.observe_only:
                warn = (f"\n\n[TRUTHGUARD] This contradicts what the fleet already "
                        f"established: {v.get('reason')}. Do not build on either "
                        f"version until it is resolved.")
                try:
                    ctx.tool_result = result + warn
                except Exception:
                    pass
        except Exception:
            pass          # never break the crew

    def _before_llm(self, ctx):
        """Inject what the fleet knows before the model reasons. Without this the
        agent reasons from its own local history and shared memory is write-only."""
        if self.s.observe_only:
            return
        try:
            msgs = getattr(ctx, "messages", None)
            if not msgs:
                return
            last = str(msgs[-1].get("content", "")) if isinstance(msgs[-1], dict) else ""
            block = self.s.as_prompt_block(last[:200])
            if block and isinstance(msgs[-1], dict):
                msgs[-1]["content"] = block + "\n\n" + last
        except Exception:
            pass


def crewai_governance(agent_id: str = None, namespace: str = None,
                      token: str = None, gate_tools=None,
                      observe_only: bool = False) -> CrewAIGovernance:
    """Govern every CrewAI agent in this process with one call."""
    return CrewAIGovernance(
        GovernanceSurface(agent_id, namespace, token, observe_only), gate_tools)


# ── generic: any framework with a callback ───────────────────────────────────

def wrap_tool(fn, surface: GovernanceSurface, name: str = None):
    """Guard a plain callable. The fallback path for frameworks with no hook API —
    equivalent to the toolkit's generic `run_tool` surface."""
    tool_name = name or getattr(fn, "__name__", "tool")

    def guarded(*a, **kw):
        out = fn(*a, **kw)
        surface.on_conclusion(subject=tool_name, relation="returned",
                              object=str(out)[:400], claimant=tool_name)
        return out
    guarded.__name__ = tool_name
    return guarded
