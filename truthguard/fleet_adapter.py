"""Governed shared memory for an external agent framework.

CrewAI, AutoGen, AgenticFleet and the rest all give agents a shared memory that
is last-writer-wins: an agent stores a conclusion, and the store keeps it. That is
the failure this project exists to prevent, so wiring one of those frameworks to
TruthGuard means replacing its memory backend, not adding a tool beside it.

This is the seam. It is deliberately framework-agnostic — four methods, no
imports from any agent library — because every framework spells "memory" and
"tool" differently, and a seam that names one of them has to be rewritten for the
next. The framework-specific part is then a handful of lines at the call site.

    mem = FleetMemory("researcher", namespace="clinical")
    for c in mem.recall("what is the rate limit"):     # what the fleet believes
        ...
    v = mem.remember("rate limit", "is", "100 rps", confidence=0.8,
                     claimant="policy_2024.pdf")
    if v["verdict"] == "CONFLICTED":
        escalate(v["conflict"])                        # a human decides
    mem.done("SUCCESS")

The verdict is returned rather than raised, and CONFLICTED is not an error: both
claims persist and neither wins. An adapter that raised would push the framework
into discarding one side, which is the behaviour being replaced.
"""
import os

from . import admission, episodes, graph_store, identity
from .context_graph import ContextGraph


class FleetMemory:
    """One agent's handle on shared memory. Construct one per agent, per run."""

    def __init__(self, agent_id: str, namespace: str = None, token: str = None,
                 goal: str = None, may_assert: list = None,
                 quota_per_hour: int = 0):
        self.agent_id = agent_id
        self.namespace = namespace or os.getenv("TG_NAMESPACE", "default")
        self.token = token if token is not None else os.getenv("TG_AGENT_TOKEN", "")
        self.cg = ContextGraph()
        # An episode per run, so every claim is traceable to the actions behind it
        # and a failed run's tool sequence is still there to diagnose from.
        self.episode = episodes.start_episode(
            goal or f"{agent_id} run", agent_id=agent_id, namespace=self.namespace)
        if may_assert is not None:
            admission.set_policy(self.cg, agent_id, may_assert, self.namespace,
                                 quota_per_hour=quota_per_hour)

    # ── the write path ───────────────────────────────────────────────────────
    def remember(self, subject: str, relation: str, object: str,
                 confidence: float = 0.6, claimant: str = None,
                 sources: list = None, valid_from: str = None,
                 valid_until: str = None, severity: str = "normal",
                 derived_from: list = None) -> dict:
        """Offer one conclusion to shared memory. Returns the gate's verdict.

        `claimant` is who ASSERTS the fact — pass the source document when the
        agent is relaying what it read, and leave it out when the conclusion is
        the agent's own. Getting this right is what makes a conflict say
        "policy_2023 vs policy_2024" instead of blaming two messengers.

        `derived_from` lists the claim ids this was reasoned FROM. Passing them is
        what lets a later retraction find this conclusion and flag it; without
        them the reasoning chain has a hole and trust propagation stops here.
        """
        claim = {"subject": subject, "relation": relation, "object": object,
                 "confidence": confidence, "claimant": claimant,
                 "sources": list(sources or []) + list(derived_from or []),
                 "valid_from": valid_from, "valid_until": valid_until,
                 "severity": severity}
        v = admission.admit(self.cg, claim, agent_id=self.agent_id,
                            namespace=self.namespace, episode_id=self.episode,
                            token=self.token)
        episodes.log_tool_call(self.episode, "remember",
                               {"subject": subject, "relation": relation},
                               outcome="ok" if v.get("node") else "error",
                               error_text=v.get("reason", "") if not v.get("node") else "",
                               result=v)
        return v

    # ── the read path ────────────────────────────────────────────────────────
    def recall(self, question: str, k: int = 5) -> list:
        """What the fleet already believes about this, as plain dicts.

        Excludes retracted and quarantined claims at the storage layer, so a
        refuted conclusion cannot re-enter as a premise. CONFLICTED ones are
        included, with the verdict attached — an agent reading a disputed fact
        must see that it is disputed rather than one arbitrary side of it.
        """
        import re
        terms = [w for w in re.findall(r"[a-z0-9]{3,}", (question or "").lower())]
        rows = graph_store.search_claims(self.cg.storage_dir, self.namespace,
                                         terms, limit=k * 4)
        out = []
        for nid, d in rows[:k]:
            out.append({"id": nid, "subject": d.get("subject"),
                        "relation": d.get("relation"), "object": d.get("object"),
                        "confidence": d.get("confidence"),
                        "claimant": d.get("claimant") or d.get("agent_id"),
                        "submitted_by": d.get("agent_id"),
                        "disputed": d.get("write_verdict") == "CONFLICTED",
                        "valid_from": d.get("valid_from"),
                        "valid_until": d.get("valid_until")})
        episodes.log_tool_call(self.episode, "recall", {"q": question},
                              outcome="ok", result=out)
        return out

    def conflicts(self, min_severity: str = "low") -> list:
        """Open disagreements in this namespace, most severe first. A reviewer
        agent reads this; it does not resolve them, because the point of
        materialising a conflict is that a human rules on it."""
        self.cg.refresh()
        return admission.open_conflicts(self.cg, self.namespace, min_severity)

    def needs_review(self) -> list:
        """Conclusions flagged because something they were built on was retracted."""
        self.cg.refresh()
        return admission.review_queue(self.cg, self.namespace)

    def done(self, outcome: str = "SUCCESS") -> dict:
        """Close the run and roll it up onto the z-plane."""
        return episodes.end_episode(self.episode, outcome=outcome, cg=self.cg)


def provision(agent_ids: list, namespace: str = None,
              policies: dict = None, quota_per_hour: int = 0) -> dict:
    """Operator-side: register a fleet and return each agent's secret ONCE.

    Registration is separate from FleetMemory because it is the operator's act,
    not the agent's — an agent that could register itself could grant itself any
    identity, which is the hole verified identity exists to close.

    policies maps agent_id -> list of relations it may assert; '*' permits all.
    """
    namespace = namespace or os.getenv("TG_NAMESPACE", "default")
    cg = ContextGraph()
    out = {}
    for aid in agent_ids:
        r = identity.register_agent(cg, aid, namespace, save=False)
        out[aid] = r["token"]
        admission.set_policy(cg, aid, (policies or {}).get(aid, ["*"]), namespace,
                             quota_per_hour=quota_per_hour, save=False)
    cg.save()
    return {"namespace": namespace, "tokens": out,
            "note": "store these now; they are not recoverable from the graph"}
