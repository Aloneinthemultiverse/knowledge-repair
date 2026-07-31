"""TruthGuard as a backend for Microsoft's Episodic Memory Kernel.

EMK (`emk`, MIT) stores agent experience as immutable `Episode(goal, action,
result, reflection)` records behind a `VectorStoreAdapter` ABC, with pluggable
backends — JSONL, ChromaDB. Its purpose is that ONE agent learns from ITS OWN
past attempts: "last time I did this, it failed".

Implementing their ABC means any agent already written against EMK gets
TruthGuard underneath without changing a line, which is the cheapest possible
integration path — no adapter per framework, just one class.

What the swap adds, and it is the whole point:

EMK stores what an agent DID. It has no notion of what an agent CONCLUDED, so
two agents recording contradictory results produce two happy episodes and
nobody notices. This backend writes the episode faithfully AND puts the result
through the admission gate as a claim, so a contradiction between agents becomes
visible in a store whose interface never mentioned claims.

The episode is always written, even when the gate raises a conflict. Refusing to
store an experience because it disagrees with another would break EMK's
append-only contract and lose the audit trail — the disagreement belongs beside
both records, not instead of one.

    from truthguard.emk_bridge import TruthGuardStore
    store = TruthGuardStore(agent_id="researcher", namespace="proj")
    store.store(Episode(goal="find rate limit", action="read policy.pdf",
                        result="100 rps", reflection="from 2023 revision"))
"""
import os
import time
import uuid

try:                                        # EMK is an optional dependency
    from emk.store import VectorStoreAdapter
    from emk.schema import Episode
except Exception:                           # pragma: no cover
    VectorStoreAdapter, Episode = object, None


class TruthGuardStore(VectorStoreAdapter):
    """EMK's storage interface, backed by the governed graph.

    Every method EMK declares is implemented so an existing caller sees no
    difference — except that `update` and `delete` are honoured as RETRACTION
    rather than mutation, because episodes are append-only and because a deleted
    experience that other conclusions were built on is exactly what trust
    propagation exists to catch.
    """

    def __init__(self, agent_id: str = None, namespace: str = None,
                 token: str = None, gate: bool = True):
        self.agent_id = agent_id or os.getenv("TG_AGENT_ID", "emk")
        self.namespace = namespace or os.getenv("TG_NAMESPACE", "default")
        self.token = token if token is not None else os.getenv("TG_AGENT_TOKEN", "")
        self.gate = gate
        self._mem = None
        self._episodes = {}                  # id -> Episode, for get_by_id/list
        self._claims = {}                    # episode_id -> claim node id

    @property
    def mem(self):
        if self._mem is None:
            from .fleet_adapter import FleetMemory
            self._mem = FleetMemory(self.agent_id, self.namespace, self.token,
                                    goal="emk-backed agent")
            self._mem_ep = self._mem.episode
        return self._mem

    # ── EMK write path ───────────────────────────────────────────────────────
    def store(self, episode, embedding=None) -> str:
        """Record the experience, and gate what it CONCLUDED.

        goal+action name what was attempted; `result` is the fact obtained, and
        the fact is the part other agents will build on. So the claim is
        subject=goal, relation="resulted in", object=result — which makes two
        agents pursuing the same goal with different results a contradiction
        rather than two unrelated rows.
        """
        eid = getattr(episode, "episode_id", None) or f"ep:{uuid.uuid4().hex[:12]}"
        self._episodes[eid] = episode

        from . import episodes as _ep
        try:                                  # z-plane: what was done
            _ep.log_tool_call(self.mem.episode, "emk.store",
                              {"goal": str(getattr(episode, "goal", ""))[:120]},
                              outcome="ok", result=str(getattr(episode, "result", ""))[:400])
        except Exception:
            pass

        if not self.gate:
            return eid
        try:
            v = self.mem.remember(
                subject=str(getattr(episode, "goal", ""))[:200].strip().lower(),
                relation="resulted in",
                object=str(getattr(episode, "result", ""))[:400],
                confidence=0.8,
                claimant=self.agent_id)
            if v.get("node"):
                self._claims[eid] = v["node"]
            # Surface the verdict on the episode's metadata rather than raising.
            # An EMK caller does not expect store() to fail, and breaking that
            # contract is how a governance layer gets removed.
            meta = getattr(episode, "metadata", None)
            if isinstance(meta, dict):
                meta["truthguard_verdict"] = v.get("verdict")
                if v.get("verdict") == "CONFLICTED":
                    meta["truthguard_conflict"] = v.get("reason")
        except Exception:
            pass
        return eid

    # ── EMK read path ────────────────────────────────────────────────────────
    def retrieve(self, query=None, k: int = 5, **kw) -> list:
        """Similar past experiences — now including OTHER agents', not just this
        one's. EMK backends are per-agent files; the graph is shared, so an agent
        can learn from a peer's attempt instead of repeating it."""
        q = query if isinstance(query, str) else str(query or "")
        out = []
        for f in self.mem.recall(q, k=k):
            out.append({"goal": f.get("subject"), "result": f.get("object"),
                        "claimant": f.get("claimant"),
                        "disputed": f.get("disputed"),
                        "claim_id": f.get("id")})
        return out

    def get_by_id(self, episode_id: str):
        return self._episodes.get(episode_id)

    def retrieve_failures(self, query=None, k: int = 5, **kw) -> list:
        """Past runs that failed, from the z-plane's failure signatures. Clusters
        by failure CLASS rather than by embedding distance, so two crashes with
        different stack text but the same exception type retrieve each other."""
        from . import episodes as _ep
        return _ep.similar_failures(str(query or ""), k=k)

    def retrieve_successes(self, query=None, k: int = 5, **kw) -> list:
        return [r for r in self.retrieve(query, k) if not r.get("disputed")]

    def retrieve_with_anti_patterns(self, query=None, k: int = 5, **kw) -> dict:
        """Successes, failures, and — beyond EMK — what is actively DISPUTED.

        An anti-pattern in EMK is something that went wrong before. Shared memory
        adds a sharper one: a belief two agents disagree about right now. Acting
        on it is a mistake nobody has made yet, so no failure episode exists to
        warn you.
        """
        hits = self.retrieve(query, k)
        return {"successes": [h for h in hits if not h.get("disputed")],
                "failures": self.retrieve_failures(query, k),
                "disputed": [h for h in hits if h.get("disputed")],
                "open_conflicts": self.mem.conflicts()}

    # ── mutation: honoured as retraction ─────────────────────────────────────
    def update(self, episode_id: str, episode) -> bool:
        """Correcting an experience retracts the old claim and admits the new one.

        Overwriting in place would erase what was believed before, and anything
        derived from it would keep standing on a fact that no longer exists. The
        retraction is what makes those dependents get flagged.
        """
        old = self._claims.get(episode_id)
        if old:
            try:
                from . import admission
                admission.retract(self.mem.cg, old, reason="episode updated")
            except Exception:
                pass
        self.store(episode)
        return True

    def delete(self, episode_id: str) -> bool:
        """Retract, never erase. The record of what was believed and when it was
        withdrawn is the audit trail EMK's append-only design exists to protect."""
        node = self._claims.get(episode_id)
        if not node:
            return self._episodes.pop(episode_id, None) is not None
        try:
            from . import admission
            admission.retract(self.mem.cg, node, reason="episode deleted")
        except Exception:
            return False
        return True
