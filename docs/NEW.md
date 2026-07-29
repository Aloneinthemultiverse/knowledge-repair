# Expansion: Verified Shared Memory for Agent Fleets

Status: **design, not built.** Nothing in this document exists in the codebase yet.
Written 2026-07-25.

---

## The thesis

TruthGuard's assessment gate currently protects one direction — it decides whether a
question may be *answered*. For a fleet of agents sharing one memory, the same machinery
has to protect the other direction: whether a conclusion may be *remembered*.

```
today   retrieve  → GATE → generate          "should this be answered?"
new     conclude  → GATE → shared memory     "should this be remembered?"
```

That symmetry is the whole design. It is available to us and not to mem0/Zep/Letta
because they have no gate to mirror.

## The problem it solves

In any shared-memory system today, the last writer wins. So:

1. Agent A concludes something wrong and writes it.
2. Agent B retrieves it as context and builds on it.
3. Agent B writes a second, dependent, wrong conclusion.

Hallucination compounds silently and the fleet has no immune system. Memory quality
degrades monotonically as the fleet gets busier — which is exactly backwards.

## What we already have that makes this cheap

- **Bi-temporal clash detection** — `same subject + relation + overlapping validity +
  different value` is already implemented in `assess.py`. It currently runs over chunks
  within a single query. Pointing it at *stored conclusions across agents* is a scope
  change, not new logic.
- **Confidence and band on every spine node** — already recorded by `record_turn`.
- **Provenance edges** — `grounds` (to chunks) and `references` (to code) already exist,
  with `EXTRACTED` / `INFERRED` tagging.
- **Cross-process shared state** — the mtime hot-reload means separate MCP clients
  already read and write one graph without restart.

---

## New components

### 1. Write admission controller

Every agent write passes three checks before entering shared memory:

| Check | Verdict |
|---|---|
| Confidence below namespace threshold | **QUARANTINED** — stored, not served |
| Contradicts a stored conclusion over an overlapping validity window | **CONFLICTED** — both kept, neither wins |
| Clean | **ACCEPTED** |

The rule that matters: **no silent overwrite.** Last-writer-wins is the bug being fixed.

### 2. Conflict nodes

When two agents disagree, materialise the disagreement rather than resolving it:

```
ConflictNode {
  subject, relation, validity_window,
  claims: [
    { agent_id, value, confidence, sources[] },
    { agent_id, value, confidence, sources[] }
  ],
  status: OPEN | RESOLVED,
  resolved_by: agent_id | human,
  opened_at
}
```

Any agent retrieving that fact receives **both claims plus the conflict flag** — the
existing dual-answer behaviour, extended from answers to memory.

### 3. Trust propagation

Add a `derived_from` edge whenever a conclusion grounds on another *conclusion* (rather
than on a source chunk). When a conclusion is later refuted or superseded, walk that
edge backwards and mark every dependent conclusion `needs_review`.

This is the containment mechanism: one bad fact stops at the blast radius rather than
diffusing through the fleet. No other agent-memory system does this.

### 4. Agent reliability scoring

Recall already ranks by `similarity × confidence`. Add a third term:

```
score = similarity × confidence × agent_reliability
```

where `agent_reliability` is derived from that agent's accepted / quarantined / later-
refuted ratio. An agent that keeps being wrong is automatically down-weighted. The fleet
self-corrects without human tuning.

### 5. Namespaces

`namespace` on every node, scoping both reads and writes. Teams share infrastructure,
not memory.

### 6. The z-plane — a fourth axis for observation

The three existing planes all answer *what is known*:

```
y+   documents      what the corpus states
x    conversation   what was concluded
y-   code           what the system is made of
```

None of them answer *what was done*. For a single assistant that gap is tolerable — one
actor, one session, and the conclusion is the whole story. For a fleet it is disqualifying:
several agents act concurrently, and a conclusion alone cannot tell you which agent
produced it, through which actions, at what cost, or whether the run succeeded.

**The z-axis is the observation plane.** It is where a fleet becomes visible.

```
         y+  knowledge
             │
  z ─────────┼───────── x  conversation
  action     │
             y-  code
```

| Lives on z | Meaning |
|---|---|
| `EpisodeNode` | one agent run: goal, tool sequence, outcome, duration, retries |
| `failure_signature` | normalised error class, so failures cluster rather than scatter |
| agent activity | which agent, which namespace, when |

**Cross-plane edges from z:**

| Edge | Meaning |
|---|---|
| `produced` → x | this run yielded that conclusion |
| `used` → y− | this run touched that code |
| `consulted` → y+ | this run read that document |
| `similar_to` → z | this run resembles that past run |

The `produced` edge is the one that completes provenance. Today a conclusion links to the
*sources* it grounded on. With z it also links to the *actions* that produced it — so
"why does the system believe this" has two answers: which evidence, and which run.

**Raw traces do not live on the plane.** One agent session emits hundreds of tool calls
against a graph of ~3,300 nodes; ingesting them raw would swamp every traversal. Raw calls
go to a flat append-only table, and only the rolled-up episode enters the graph. This is
the pattern already in use — 603 raw chunks in `chunks.json`, 86 rollup `knowledge` nodes
in the graph.

**Why this belongs in Part I rather than later:** Parts II and III both depend on it.
Cross-agent contradiction needs to know which agent asserted what (z supplies the author
and the run). Reliability learning needs outcome labels (z supplies success and failure).
Without the observation plane there is nothing to attribute a correction *to*.

It also delivers value before any learning exists: embed episodes, and when something
breaks, retrieve the nearest failed runs. *"This resembles three past failures, all in the
Edit-after-Bash chain, all FileNotFoundError."* Diagnosis on day one, no model involved.

---

## Data model deltas

**Spine node — add:**

```
agent_id          which agent/client produced this
namespace         tenant isolation
write_verdict     ACCEPTED | QUARANTINED | CONFLICTED
verified_by[]     other agents that independently reached the same conclusion
superseded_by     lifecycle
```

**New edge types:**

| Edge | Meaning |
|---|---|
| `contradicts` | conclusion ↔ conclusion, cross-agent disagreement |
| `confirms` | an agent independently reached the same conclusion |
| `derived_from` | conclusion built on conclusion — the trust chain |

`confirms` is quietly the most valuable: independent agreement from two agents working
from different sources is real evidence, and should raise stored confidence.

---

## Build order

| Phase | What | Effort | Why |
|---|---|---|---|
| 1 | `agent_id` + `namespace` on writes | days | unlocks everything else |
| 2 | Cross-agent contradiction → ConflictNode | days | **the demo** |
| 3 | `derived_from` + trust propagation + reliability | ~1 week | the containment story |
| 4 | SQLite/Postgres backend | weeks | hard blocker — pickle is single-writer |
| 5 | Access control, audit log | later | necessary, not differentiating |

Phase 4 is the real engineering. Phases 1–3 are mostly scope changes to existing logic.

---

## Known gaps this design does *not* address

Recorded deliberately, so they are not mistaken for solved:

- **Coordination.** This is memory hygiene, not collaboration. Two agents can still start
  the same task; the conflict is detected after the fact, not prevented before it.
- **Shared goals.** The graph stores what is *true*, not what the fleet is *trying to do*.
- **Human escalation.** `OPEN` conflicts need a resolver. With one operator and many
  agents, the human is the bottleneck, so conflicts need ranking and batching.
- **Concurrency.** Until Phase 4, concurrent writes from two agents can corrupt the
  pickle file. This is a correctness blocker for any real fleet.

---

## Positioning

> Fleets share memory, and today the last writer wins — one agent's wrong conclusion
> becomes another agent's premise, and hallucination compounds silently. We put the
> assessment gate on the write path too. Every conclusion is stored with its author, its
> confidence, and its sources. When two agents disagree over the same time window, we
> materialise the conflict instead of overwriting it. And when a conclusion is refuted,
> we walk the provenance chain and flag everything built on it.
>
> Not shared memory — reconciled memory.

---

# Part II — The learned layer

## What the thing is

A **graphical meta neuro-symbolic model for agent fleets.** Each word is load-bearing, so
each is defined:

| Term | What it means here | Where it lives |
|---|---|---|
| **Neuro** | reliability weights, support weights, per-domain thresholds — continuous values learned from outcomes | node and edge attributes |
| **Symbolic** | temporal-overlap rule, contradiction detection, legal-action constraints, provenance semantics | hard rules, never learned |
| **Meta** | the system reasons about the *quality of conclusions* — its own and other agents' — not only about their content | the gate, plus reliability |
| **Graphical** | parameters attach to graph structure, so learning is local, inspectable, and bounded by a neighbourhood | the graph itself |
| **For agent fleets** | the unit of learning is the **agent**, not the user | `agent_id` on every write |

The distinction that matters most is **meta**. An ordinary RAG system reasons about
whether a claim is supported. This one additionally reasons about *how much a conclusion
should be trusted given who produced it, how confidently they asserted it, and how their
prior conclusions have held up*. That second question is what a fleet needs and a single
assistant does not.

## The taste-1 analogue, stated plainly

Command Code's `taste-1` is a meta neuro-symbolic model with continuous RL, learning **one
developer's preferences** from their accept / reject / edit signals. It answers: *what
does this person consider good code?*

This is the fleet analogue. It learns **a fleet's trust topology** from outcome signals
the system generates itself. It answers: *which agent, on which kind of question, should
be believed — and how much?*

| | taste-1 | this |
|---|---|---|
| Unit of learning | a person | **an agent** |
| Question answered | what does this person prefer? | **who should be believed, about what?** |
| Signal | human accept / reject / edit | **supersession, adjudicated contradiction, refusal outcome** |
| Requires a human | yes, structurally | **no** |
| Where parameters live | `.commandcode/taste/` packages | **the graph** |

The last two rows are the position. There is no human clicking accept between two agents,
so a preference-signal design cannot extend to agent-to-agent trust. An outcome-signal
design can — and outcomes are exactly what an assessment gate already produces.

## How the three parts compose into one system

| Part | Contributes |
|---|---|
| **I — verified shared memory** | the substrate: agent identity, conflict nodes, provenance chains. Without `agent_id` there is no unit to learn about. |
| **II — the learned layer** | the formalism: what the state, action, and reward are; why the symbolic layer shields the learned one |
| **III — graph-native learning** | the implementation: parameters as graph attributes, updated by correction, read back at three existing decision points |

Part I must be built first. Parts II and III are not separable — III is how II is
realised in this system rather than as a model beside it.

## Where the field actually is

A survey of 178 neuro-symbolic papers (2020–Nov 2025, published Jan 2026) breaks
integration work down as: learning & inference 63%, knowledge representation 44%, logic
& reasoning 35%, explainability 28%, **meta-cognition 5%**. The survey's own finding is
that meta-cognition demonstrates *greater performance impact than sophisticated
integration patterns alone* while being the least explored.

Meta-cognition — a system reasoning about its own reasoning — is what the assessment
gate already does. We built a meta-cognitive layer without calling it one. That 5% is
the lane.

Separately, enterprise agent orchestration is converging on the Agentic Ontology of Work:
**Agents, Skills, Intents, Contexts, Policies, Memory, Confidence, Outcomes**. We have
Contexts, Memory, Confidence, and partial Outcomes. We lack Agents, Skills, Intents,
Policies — the same gap Phase 1 above identifies, which is reassuring: the industry is
standardising on the thing we already planned to add.

## We are already neuro-symbolic

| Layer | What plays it here |
|---|---|
| **Neural** — interprets raw data | MiniLM embeddings, cross-encoder rerank, LLM triple extraction |
| **Symbolic** — reasons over structure | bi-temporal overlap rules, contradiction dictionary lookup, graph traversal, provenance tagging |

The textbook split — neural as sensory layer, symbolic as cognitive layer — is what we
already run. The expansion is not "become neuro-symbolic". It is "add the meta layer".

## Prior art: Command Code's `taste-1`

Command Code (commandcode.ai, $5M seed led by PWV, beta Feb 2026) ships a model-agnostic
coding-agent harness whose differentiator is `taste-1`, described as a *meta
neuro-symbolic AI model with continuous reinforcement learning*.

| Component | How theirs works |
|---|---|
| Signal | every human accept, reject, edit |
| Loop | "self-aware RL feedback loop to build skills" |
| Neural | intuition learned from observed code texture |
| Symbolic | "enforces the logic of user choices" |
| Storage | `.commandcode/taste/` project · `~/.commandcode/taste/` global · remote sync |
| Output | `/skills` + `/memory` packages, categorised, with confidence values |

Worth noting their storage is folders of confidence-scored packages — the same tell as
Taste 1.0 (a competing "taste model" that is literally a folder of markdown). The
"model" is substantially a structured judgment layer, not a large neural net.

### How ours differs — the axis that matters

| | taste-1 | this design |
|---|---|---|
| Learns whose preference | one **human's** | the **fleet's** |
| Signal source | human accept / reject / edit | **outcomes**: supersession, contradiction, refusal correctness |
| Human in the loop | **required** | **not required** — self-labelling |
| Substrate | folders of skill packages | **the graph itself** |
| Scope | style and conventions | **factual reliability and trust** |

The critical difference: **their reward requires a human.** No accept/reject, no
learning. Ours comes from the gate — a superseded conclusion is a negative signal
whether or not anyone was watching. That is why theirs cannot extend to agent-to-agent
trust: there is no human clicking accept between two agents.

### What to borrow

1. **Confidence-scored, categorised packages** rather than one undifferentiated blob.
2. **Project scope vs global scope** — maps directly onto our namespaces.
3. **"Model-agnostic harness, not a model lab"** positioning. We are the same: MCP,
   bring-your-own-key. Claim the layer, not a foundation model.

### What not to do

Do not lead with "meta neuro-symbolic + RL" — that is now a funded competitor's launch
positioning, and we would be compared directly against a shipped product. Lead with what
they structurally cannot do: **no-human-in-the-loop reward, and cross-agent trust.**

---

## The RL formulation

The graph is not storage for the learner. The graph *is* the environment.

| RL element | In this system |
|---|---|
| **State** | the retrieved neighbourhood — question embedding, top-k chunks, past turns, confidence signals |
| **Action** | the controller's choice: answer · dual-answer · clarify · refuse · rewrite · retrieve-more |
| **Reward** | derived from the graph, automatically |
| **Policy** | which action to take given this graph state |

### The reward signal already exists

| Observation | Reward |
|---|---|
| Answer later superseded | negative — it was wrong |
| Refused, and the fact genuinely was absent | positive — correct abstention |
| Refused, but the fact was present | negative — over-cautious |
| Dual-answered, conflict was real | positive |
| Rewrite loop that then succeeded | positive, credited to the rewrite |
| Answer never contradicted, repeatedly re-accessed | positive |

Reward engineering is normally the hard part of applied RL. Here the gate produces it as
a byproduct. This is the same signal Taste Labs pays expert humans to generate.

### Symbolic shielding — what makes it neuro-symbolic rather than RL-on-a-graph

```
SYMBOLIC LAYER — hard constraints, defines what is LEGAL
  sufficiency below threshold   -> ANSWER is not in the action space
  unresolved contradiction      -> single ANSWER illegal, DUAL only
  no evidence at all            -> REFUSE is the only legal action
            |
            v
NEURAL POLICY — learns to choose among LEGAL actions
  given the graph neighbourhood, which permitted action maximises reward?
```

The gate never becomes learnable. It stays a hard rule; the policy only chooses within
what is already safe. This pattern is known in the literature as shielded RL / symbolic
shielding, and it addresses RL's worst deployment property: the policy cannot learn its
way into an unsafe action, because unsafe actions are never in the action space.

Note what this replaces: the controller's thresholds (`>=0.75 answer`, `0.4-0.75 hedge`,
`<0.4 refuse`, `max 2 rewrites`) are currently hand-picked constants. A learned policy
would derive them from outcome history, per domain, and keep adapting.

### Graph-based state encoding

Use a GNN over the retrieved neighbourhood as the state encoder. The state is then not a
flat vector but the actual subgraph — which chunks, which past turns, which communities,
and how they connect. That is the "graph-based" half, and it is what distinguishes this
from bolting a bandit onto a retrieval score.

---

## Honest challenges

None are fatal; all are real, and none should be discovered on stage.

1. **Reward delay.** A wrong answer is only revealed when superseded — possibly weeks
   later. Classic credit assignment.
2. **Sample efficiency.** RL is data-hungry. 366 turns is far too few. Thousands of
   episodes are needed before a learned policy beats hand-tuned thresholds.
3. **Exploration is dangerous in production.** RL learns by trying suboptimal actions —
   i.e. deliberately giving worse answers sometimes. Unacceptable for a trust product
   without careful gating.
4. **Non-stationarity.** The corpus changes, so the environment shifts under the policy.

## Therefore: stage it, do not start with full RL

| Stage | Method | Episodes needed | Exploration risk |
|---|---|---|---|
| 1 | **Offline** — supervised on logged outcomes; learn thresholds from history | ~500 | **none** |
| 2 | **Contextual bandit** — one-step, no long credit chains | ~2k | low |
| 3 | **Full RL with GNN policy** — multi-step, learns rewrite strategy | ~10k+ | managed |

Stage 1 delivers most of the value at zero risk, and supports an honest claim nobody
else makes: *our controller thresholds are learned from outcome history, not hand-picked.*

---

## Prerequisite: the capture layer

None of the three stages is possible without episode data. This is the thing to build
first, and it is deliberately not a model.

**Raw store — flat, append-only, not in the graph:**

```sql
CREATE TABLE tool_calls (
  id, episode_id, agent_id, ts,
  tool, args_hash, args_json,
  outcome,           -- ok | error | timeout
  error_text, duration_ms, exit_code
);
```

**Rollup — a fourth plane in the same graph:**

```
EpisodeNode {
  id, agent_id, namespace, goal,
  tool_sequence: ["Read", "Edit", "Bash"],
  outcome: SUCCESS | FAILURE | PARTIAL,
  failure_signature,      -- normalised error class, so failures cluster
  n_retries, duration, embedding
}
edges:  produced -> decision (x)   used -> code symbol (y-)   similar_to -> episode
```

This mirrors the pattern already in use: `chunks.json` holds 603 raw chunks while the
graph holds 86 rollup `knowledge` nodes. Raw stays flat; rollup goes in the graph.

**Why not put raw tool calls in the graph:** a single agent session produces hundreds of
tool calls. The entire graph is currently 3,336 nodes. Raw ingestion would swamp the
spine and slow every traversal for no gain.

**Why not a separate database:** the value is precisely the link *"this decision was
produced by these actions"*. That has to be walkable in one hop, which means the rollup
belongs in the graph.

**Immediate use case, no training required:** embed episodes, and when something breaks,
retrieve the nearest failed episodes. *"This resembles three past failures, all in the
Edit-after-Bash chain, all FileNotFoundError; two were fixed by re-reading the file
first."* That is the diagnostic capability, available on day one, and it is also how the
evaluation set for Stage 1 gets built.

## Build order

| # | What | Why |
|---|---|---|
| 1 | `tool_calls` table + capture hook | data starts accruing immediately; cannot be backfilled |
| 2 | Episode rollup into the z-plane, `produced` edges | links actions to decisions |
| 3 | Episode embeddings + similarity retrieval | the diagnostic use case, working, no training |
| 4 | Offline policy learning on accumulated outcomes | Stage 1 above, once ~500 episodes exist |

Capture more than seems necessary — args, durations, exit codes, retry counts. Storage is
cheap; data not captured is gone permanently.

---

# Part III — Graph-native neuro-symbolic learning

## The design principle

Everything in Part II assumed a model *trained on* the graph — a GNN or a policy network,
living beside the graph as a separate artefact. This part proposes something different
and better suited to this system:

> **The graph is the model.** Learned parameters are node and edge attributes. There is
> no separate weights file, no training job, no inference server. Learning is an update
> to the graph, applied when the system corrects itself.

This matters because it is *adaptable to what already exists here*. No new
infrastructure, no serving layer, no model versioning. The graph is already persisted,
already traversed at query time, already carries confidence. Adding learned attributes to
it is an extension of a structure in production, not a parallel system to keep in sync.

## The learnable parameters

| Parameter | Lives on | Meaning |
|---|---|---|
| `agent_reliability` | agent node | how often this agent's conclusions survive |
| `source_reliability` | chunk / document node | how often conclusions grounded here hold up |
| `support_weight` | `grounds` / `references` edge | how predictive this source was of a correct conclusion |
| `threshold_prior` | namespace / community node | the confidence band that has proven correct *in this domain* |

Note the last one. The controller's bands (`>=0.75 answer`, `0.4-0.75 hedge`,
`<0.4 refuse`) are currently three global constants chosen by hand. As node attributes on
communities, they become **per-domain and learned** — a clinical namespace can converge on
a stricter refusal threshold than a documentation namespace, without anyone tuning it.

## Correction as the training signal

The system already corrects itself. Those corrections are the labels.

| Correction event | Already produced by | Credit assignment |
|---|---|---|
| Conclusion superseded | `run_supersede` / bi-temporal check | sources and author of the old conclusion lose weight; sources of the new one gain |
| Contradiction resolved | conflict node adjudicated | the losing claim's author and sources lose weight |
| Refusal, and evidence genuinely absent | gap analysis | reinforce the threshold that fired |
| Refusal, but evidence was present | later successful answer on the same question | relax the threshold for that community |
| Rewrite loop that then succeeded | trace `retrieve -> assess -> rewrite -> answer` | reinforce the rewrite; penalise the original phrasing |

No annotator, no reward model, no human in the loop. This is the property Command Code's
`taste-1` cannot have: its signal is a human clicking accept.

## The update rule

When conclusion `C` is superseded by `C'`:

```
1. sources(C)   := walk grounds edges backward from C
2. for each s in sources(C):
       source_reliability[s]  -= lr * support_weight[s->C]
       support_weight[s->C]   -= lr
3. agent_reliability[author(C)] -= lr * confidence(C)
       # penalty scales with confidence: being confidently wrong costs more
       # than being tentatively wrong
4. symmetric positive update for sources(C') and author(C')
5. for each d in dependents(C) via derived_from:
       mark d.needs_review = true
```

Two properties worth stating explicitly:

- **The penalty scales with the confidence that was asserted.** A conclusion delivered at
  0.9 confidence and later refuted costs its author far more than one delivered at 0.45.
  This is what disciplines calibration rather than merely accuracy.
- **The update is local.** Only nodes on the provenance path of the corrected conclusion
  change. Learning is `O(neighbourhood)` — the same cost profile as recall, and for the
  same structural reason. There is no epoch, no full pass, no retraining window.

## Where this is read back

The learned attributes feed three existing decision points, each of which is currently a
constant:

```
recall ranking      score = similarity x confidence x agent_reliability
retrieval weighting rank contribution scaled by source_reliability
controller bands    threshold_prior[community] instead of the global 0.75 / 0.4
```

Nothing new is invoked at query time. Three constants become three lookups.

## The symbolic / neural split, made precise

| | What it is | Can it be learned? |
|---|---|---|
| **Symbolic** | temporal-overlap rule, contradiction detection, what actions are legal, provenance semantics | **No — hard rules, never learned** |
| **Neural** | reliability scores, support weights, per-domain thresholds | **Yes — continuous, updated by correction** |

The gate is symbolic and stays symbolic. The learned layer only adjusts *weights within
the space the gate permits*. A refuted conclusion can never make the system decide that
contradiction detection should be skipped — that rule is not a parameter.

This is the shielding property from Part II, expressed in the parameterisation rather
than in an action mask: **unsafe behaviour is not merely discouraged, it is
unrepresentable.**

## Why "graphical" is the right word

Not because a GNN is used, but because the parameters are attached to graph structure:

- Learning is **local** — bounded by the provenance neighbourhood
- Learning is **interpretable** — every weight is attached to a named node or edge, so
  "why is this source down-weighted" has a literal answer with a list of the corrections
  that caused it
- Learning is **inspectable** — the model can be read in the 3D view; there is no opaque
  parameter vector

A conventional trained model would give none of the three.

## Honest limits

- **Cold start.** Every reliability begins at a uniform prior. The system is no better
  than today until corrections accumulate. It degrades gracefully — uniform weights
  reduce exactly to the current hand-tuned behaviour.
- **Sparse corrections.** Supersession is rare in a small corpus. This learns slowly by
  design, and that is the correct trade for a trust product.
- **Feedback loops.** A down-weighted source is retrieved less, so it gets fewer chances
  to be vindicated. Needs an exploration floor — never let a weight reach zero.
- **This is not a foundation model.** It is a parameterised graph. Do not describe it as
  training a model; describe it as a graph that adapts. The claim is smaller and true.

## Build order

| # | What | Prerequisite |
|---|---|---|
| 1 | Add the four attributes with uniform priors; read them at the three decision points | none — behaviour identical to today |
| 2 | Implement the update rule on supersession events | Part I supersession working |
| 3 | Extend to contradiction adjudication and refusal outcomes | conflict nodes |
| 4 | Per-community `threshold_prior` | enough per-domain volume |

Step 1 is a no-op at runtime and can ship immediately — it makes the system
*parameterised* before it is *adaptive*, which is the safe order.
