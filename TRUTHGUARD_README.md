# TruthGuard RAG — Self-Correcting RAG Pipeline

**OneInbox AI Internship Hackathon 2026 · AI Engineer track · Problem Statement 1**

Standard RAG answers every question with equal confidence — including the ones it should
refuse. TruthGuard places an **assessment gate between retrieval and generation**: the
generator is never invoked until the retrieved context has been checked for sufficiency,
contradiction, and ambiguity.

**Core thesis: hallucination is blocked structurally, not prompted away.**

🌐 **[Product page](https://truthguard-pink.vercel.app)** · **[Interactive architecture](https://truthguard-pink.vercel.app/architecture)**

---

## Table of contents

- [The problem](#the-problem)
- [How it works](#how-it-works)
- [Install and run locally](#install-and-run-locally)
- [Studio and the web pages](#studio-and-the-web-pages)
- [Use it as an MCP server](#use-it-as-an-mcp-server)
- [Context memory: three planes](#context-memory-three-planes)
- [Architecture](#architecture)
- [Benchmarks](#benchmarks)
- [Reproduction notes](#reproduction-notes)
- [Repo layout](#repo-layout)

---

## The problem

RAG failures are rarely loud. The answer looks confident, cites a source, and reads well —
but the source was missing, contradicted, unreadable, or answering a different question.

| | Failure | What happens |
|---|---|---|
| **P1** | Blindness | Scanned pages carry no text layer. The retriever never sees them, so the fact does not exist as far as the system is concerned. |
| **P2** | Omission | The answer isn't in the corpus at all. Rather than say so, the model composes one from adjacent context. |
| **P3** | Silent arbitration | Two documents disagree. The model picks one, presents it as fact, and never mentions the conflict existed. |
| **P4** | Ambiguity collapse | The question had two readings. The model answers one and behaves as if the other was never possible. |

These are not prompt problems. Asking a model to "be careful" does not fix a missing text
layer or a contradiction between two PDFs. They must be caught structurally, before generation.

## How it works

```
Ingest  ──▶  Retrieve  ──▶  ASSESS  ──▶  Respond
                              │
                       the generator does not
                       run until this passes
```

| Stage | What it does |
|---|---|
| **01 Ingest** | Mixed PDFs, scans and code-in-PDF normalized to Markdown. Pages without a text layer go through an OCR ladder that escalates per page: Tesseract, then docTR, then PaddleOCR. |
| **02 Retrieve** | Three signals scored in parallel — dense vectors, BM25 keywords, exact entity matches — fused by reciprocal rank and reranked by a cross-encoder. |
| **03 Assess** | Sufficiency, fact extraction with validity windows, contradiction detection, then an answerability verdict. |
| **04 Respond** | Answer with citations and a confidence band — or refuse, clarify, or present both sides. |

### Four possible outcomes

| Verdict | When |
|---|---|
| **Answer** | Context is sufficient and consistent. Generate with chunk-level citations. |
| **Dual-answer** | Sources conflict. Present both values with provenance rather than choosing. |
| **Clarify** | The question is ambiguous. Offer the readings, then re-run with the choice. |
| **Refuse** | The corpus cannot support an answer. Decline and report what is missing. |

### Bi-temporal facts

Every extracted fact carries a `valid_from` / `valid_until` window. Two values for the same
subject **only conflict if they claim to be true at the same time**. Non-overlapping windows
are a supersession — a timeline, not a contradiction. `$300` in 2023 followed by `$500` in
2024 is history; both claiming 2024 is a conflict.

## Install and run locally

**Requirements:** Python 3.10+, ~2 GB disk for models and indexes.

```bash
git clone https://github.com/Aloneinthemultiverse/truthguard-rag.git
cd truthguard-rag
pip install -r requirements.txt
```

Tier-1 OCR engine (optional — only needed for scanned pages):

```bash
# Windows
winget install UB-Mannheim.TesseractOCR
# macOS
brew install tesseract
# Debian/Ubuntu
sudo apt install tesseract-ocr
```

Configure a model provider:

```bash
cp .env.example .env
```

```ini
# Any OpenAI-compatible endpoint (NVIDIA NIM, OpenAI, Ollama, vLLM…)
LLM_PROVIDER=openai
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_API_KEY=nvapi-…
LLM_MODEL=deepseek-ai/deepseek-v4-pro

# Fully local alternative — no key, no quota:
# LLM_BASE_URL=http://127.0.0.1:11434/v1
# LLM_MODEL=qwen2.5:3b-instruct
# LLM_API_KEY=ollama

# Tier-2 OCR escalation. Default "none" — Tesseract alone handles ordinary
# scans, with no key, no GPU and no cost. Escalation is opt-in:
#   TG_OCR_TIER2=dots     self-hosted dots.ocr behind vLLM (needs CUDA, ~9-16GB VRAM)
#   TG_OCR_TIER2=ollama   a local vision model
#   TG_OCR_TIER2=mistral  the paid Mistral OCR API
TG_OCR_TIER2=none
# Tier-1 engine. "auto" escalates per page and stops as soon as the result is
# good enough, so clean pages never pay for the heavy models:
#   tesseract  fast, no weights          - most clean scans stop here
#   doctr      detection-then-recognition - poor scans, multi-column order
#   paddleocr  angle-invariant           - rotated/scattered text (maps, photos)
TG_OCR_ENGINE=auto
# TG_OCR_TIER2_URL=http://127.0.0.1:8000/v1
# TG_OCR_TIER2_MODEL=rednote-hilab/dots.ocr
```

### The OCR ladder

| Tier | Engine | Cost | Hardware |
|---|---|---|---|
| **1** | `auto` ladder: **Tesseract → docTR → PaddleOCR** | free | CPU |
| **2** | dots.ocr · a local vision model · Mistral OCR | free if self-hosted | dots.ocr needs a **CUDA GPU with ~9–16 GB VRAM** on the serving host |

Tier 2 fires only when tier 1 returns confidence below `0.85` or a garbage ratio
above `20%`. If the tier-2 backend is unreachable, ingestion continues with the
tier-1 result and the page keeps its real confidence — escalation being
unavailable never fails an ingest.

To serve dots.ocr on a GPU machine:

```bash
vllm serve rednote-hilab/dots.ocr --gpu-memory-utilization 0.9
# then, on the TruthGuard host:
TG_OCR_TIER2=dots TG_OCR_TIER2_URL=http://<gpu-host>:8000/v1 python -m truthguard.main ingest
```

Build the corpus and ask:

```bash
python -m truthguard.make_corpus                 # seeded trap corpus
python -m truthguard.main ingest                 # ingest + build index
python -m truthguard.main ask "What is the travel reimbursement limit per trip?"
python -m truthguard.main stats
```

### What to try (the traps)

| Ask | Watch it |
|---|---|
| `"What is the travel reimbursement limit per trip?"` | **dual-answer** — 2023 says $300, 2024 says $500, both cited, never arbitrated |
| `"Who is the CEO of DataHub Ltd?"` | rewrite loop → **refusal with gap analysis** |
| `"What is the penalty for filing an expense claim late?"` | answered **from a scanned page** (OCR confidence shown in the citation) |
| `"What does the retry_with_backoff function do?"` | code **quoted verbatim** from inside a PDF |
| `"What is the limit?"` | **multiple-choice clarifying question** |
| add `--baseline` to any | the same pipeline with the correction layer off |

### Ingest your own material

```bash
python -m truthguard.main ingest --path /your/documents
python -c "from truthguard.ingest_all import ingest_project; ingest_project('/path/to/repo')"
python -c "from truthguard.import_chat import import_chat; import_chat('session.jsonl')"
```

`ingest_project` absorbs a whole project in one call — the entire codebase, every document
in the repo, and optionally a chat transcript, all cross-linked.

## Studio and the web pages

```bash
uvicorn truthguard.api:app --port 7788      # API + Studio backend
cd studio-ui && npm install && npm run dev  # UI on http://127.0.0.1:5178
```

| Route | What |
|---|---|
| `/` | Studio — chat, trace chips, confidence ring, citations, drag-to-ingest |
| `/` → **3D graph** tab | the live 3-plane context graph |
| `/about` | product page |
| `/architecture` | interactive architecture |

Provider keys and the model can be set from the UI (⚙ → **fetch models** lists what the
provider actually serves). Keys are written to `.env` on the machine running the API.

**Exposing it publicly** (e.g. a laptop behind a tunnel) — set a token and writes stay off:

```bash
TG_API_TOKEN=<random>  TG_ALLOWED_ORIGINS=https://your.site  uvicorn truthguard.api:app --port 7790
```

| Variable | Effect |
|---|---|
| `TG_API_TOKEN` | unset = local mode, no auth. Set = every request needs the token (header, query param, or cookie) |
| `TG_ALLOW_WRITE=1` | re-enables `/config` and `/ingest`, which are **403 by default** when a token is set |
| `TG_ALLOWED_ORIGINS` | comma-separated CORS allowlist |

Graph views are served from a filename whitelist, so the tunnel cannot be used to read
arbitrary files.

## Use it as an MCP server

TruthGuard is a plain **stdio MCP server** — no client-specific code. Every client that
connects shares the same graph on disk, so a conversation started in one tool is recallable
from another.

```bash
python -m truthguard.mcp_server
```

### The twelve tools

| Tool | Arguments | What it does |
|---|---|---|
| `ask` | `question, followup?, baseline?` | Full self-correcting pipeline. Returns verdict, confidence band, citations, reasoning trace. Records the turn. |
| `get_context` | `question` | **The router call.** One ready-to-inject context block — document passages, code bodies, entities, compiled topic truths, past turns — bounded by a token budget. |
| `recall` | `question` | Searches past conversation turns by similarity × confidence over active memories. |
| `ingest_project` | `repo_path, chat_path?` | Absorbs a whole project — codebase, documents, optional transcript — cross-linked. |
| `ingest_document` | `path` | Adds a PDF/DOCX/MD/TXT. Scans run the OCR ladder. Re-wires existing turns to the new document. |
| `ingest_chat` | `path` | Imports a transcript as its own conversation chain, auto-linked to entities, docs and code. |
| `link_code_repo` | `path` | Indexes a git repo as the code plane: structural graph + AST bodies across 22+ languages. |
| `query_code` | `symbol \| cypher` | Structural traversal of the code graph, zero LLM. |
| `graph_query` | `command, name, target?` | `context`, `impact`, `find`, `edit_plan`, `path`, `report` — every edge tagged EXTRACTED or INFERRED. |
| `rebuild_communities` | — | Re-runs the DecisionGraph recipe on all three planes. |
| `graph_stats` | — | Node/edge counts per plane, turn count, community count. |
| `live_view_url` | — | URL of the auto-refreshing 3D graph view. |

### Connecting a client

**Claude Code**

```bash
claude mcp add truthguard -- python -m truthguard.mcp_server
```

Run it from the project folder, or set `cwd` in `~/.claude.json`. Restart Claude Code and
the tools appear.

**OpenCode** — `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "truthguard": {
      "type": "local",
      "command": ["python", "-m", "truthguard.mcp_server"],
      "enabled": true,
      "environment": {
        "PYTHONPATH": "/absolute/path/to/truthguard-rag",
        "TG_STORAGE_DIR": "/absolute/path/to/truthguard-rag/storage/truthguard"
      }
    }
  }
}
```

Setting `PYTHONPATH` and `TG_STORAGE_DIR` lets it run from any folder.

**Antigravity / Cursor / Cherry Studio / Codex / Gemini CLI** — add a local stdio MCP
server with command `python -m truthguard.mcp_server` and the working directory pointed at
this repo.

## Context memory: three planes

Memory here is not a chat log. Conversations, documents and codebases are compiled into a
single global graph, so recall costs proportional to a **neighborhood** rather than to history.

| Plane | Holds |
|---|---|
| **y+ knowledge** | Provenance-tagged chunks, extracted entities, community summaries with a compiled truth per topic |
| **x spine** | Turns as decision memory — confidence, decay, supersession; each session its own thread |
| **y− code** | Call and import structure alongside real function bodies, across 22+ languages |

The planes are cross-wired by `grounds` / `references` / `member_of` / `supersedes`, so any
claim traces back to its source in one hop.

**Transferable context router.** The memory does not live inside a chat window or a single
vendor's account — it lives in a graph on your disk, and any model that speaks MCP can read
and write it. Start a design discussion in one tool, continue it in another a week later,
and the second tool already knows what was decided and why.

## Architecture

Full detail in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the
[interactive version](https://truthguard-pink.vercel.app/architecture) is explorable.

| Layer | Components |
|---|---|
| **Entry** | Corpus source · CLI & eval runner |
| **Ingestion** | Ingestion pipeline (markitdown, Tesseract, PaddleOCR) · Mistral OCR API · embedding model |
| **Storage** | turbovec index (2/4-bit quantized, SIMD) · BM25 index · chunk metadata · figure asset store |
| **Retrieval** | Retrieval engine — vectors + BM25 + entity match → RRF → cross-encoder rerank |
| **Gate** | Assessment engine · query-time knowledge graph · self-correction controller · LLM provider |
| **Memory** | Session context graph · decision audit graph · code graph (GitNexus) · recall engine |
| **Surface** | MCP server (stdio) |

Key behaviours:

- Sufficiency is checked by embedding similarity first — **failures there cost zero LLM tokens**
- Triple extraction runs at query time over the top-10 chunks only, never persisted, never stale
- Confidence bands: `≥0.75` answer · `0.4–0.75` hedge · `<0.4` refuse
- At most 2 rewrites, drift-checked; hard cap of 6 LLM calls per query
- Every response carries a machine-readable trace, so answers are replayable
- Index construction uses **zero LLM calls** — deterministic chunking and local embeddings

## Benchmarks

All numbers below come from the result files in [`eval/`](eval/) in this repo.

### LOCOMO — conversational memory (`eval/locomo_results.json`)

| System | recall@10 |
|---|---|
| **TruthGuard** | **0.657** |
| graphify | 0.497 |
| hybrid RRF | 0.493 |
| dense RAG | 0.439 |
| BM25 | 0.362 |
| mem0 | 0.048 |

`n = 300`. QA accuracy on the same run: **0.735**.

Comparison figures are each system's own published numbers. Our run uses the same public
dataset with a local deterministic embedder and zero LLM cost at index time. **A BM25
baseline inside our own harness scores 0.577** — disclosed so the harness itself can be
calibrated, because 0.362 and 0.577 for the same algorithm is the size of the harness effect.

### LongMemEval (`eval/longmemeval_results.json`)

| Metric | Score | n |
|---|---|---|
| recall@10 | **0.972** | 470 |
| QA accuracy | **0.804** | 51 |

### Adversarial battery (`eval/results.json`)

15 gold questions across five categories — answerable, unanswerable, contradictory,
OCR-dependent, ambiguous.

| Metric | Result |
|---|---|
| Correct behavior | **13 / 15 (86.7%)** |
| Hallucinated | **1 / 15 (6.7%)** |

| Case | Expected | Result |
|---|---|---|
| Fact absent from corpus | refuse with gap analysis | 3 / 3 |
| Two editions disagree | dual-answer with sources | 2 / 3 |
| Ambiguous question | clarify, then answer | 2 / 2 |
| Prompt injection in a document | never surfaced | 0 leaks |
| Superseded value quoted | not treated as conflict | pass |

Reproduce:

```bash
python -m eval.run_eval --both          # ablation: correction layer on vs off
python -m eval.locomo_bench
python -m eval.longmemeval_bench
```

## Reproduction notes

Things that did not work, recorded because the alternative — quoting a favourable number
without noting what we could not reproduce — is exactly the failure mode this project exists
to prevent.

- **The baseline ablation is not currently reproducible from this repo.** `eval/results.json`
  contains only the corrected run. Figures quoted elsewhere for baseline hallucination rate
  come from an earlier run whose rows are not in the file. Re-run
  `python -m eval.run_eval --both` to regenerate both arms before citing a comparison.
- **Head-to-head against graphify is inconclusive.** We ran its published pipeline on the same
  conversational data. Its deterministic path produced no conversational nodes, and its LLM
  path produced nodes with no edges, leaving its graph query unable to traverse
  (`eval/h2h_results.json`: evidence 0.6 vs 0.0). Its published figures rely on a harness not
  included in its repository.
- **OCR behaviour is under-tested.** Only 2 OCR cases exist in the battery. That is too few to
  state an OCR accuracy figure, and none is claimed.
- **On judges.** The same memory system can score anywhere from 27% to 68% on the same
  benchmark depending on whose harness and judge model runs it. Absolute cross-paper
  comparison is unreliable; we publish methodology and a baseline calibration instead.
- **Latency.** A full `ask` is minutes, not seconds, dominated by sequential provider
  round-trips rather than local compute. Warm local retrieval is ~2s; the rest is the model.
  `fast=True` on the API drops the multi-query interpretation call; the eval harness leaves it
  on so benchmark numbers stay comparable.

## Repo layout

```
truthguard/
  main.py            CLI: ingest / ask / stats
  api.py             FastAPI — Studio backend, /ask, /config, /models, /graph
  mcp_server.py      stdio MCP server (12 tools)
  controller.py      the self-correction state machine
  assess.py          the gate: sufficiency, triples, bi-temporal clash detection
  retrieve.py        vectors + BM25 + entity → RRF → rerank
  recall.py          cross-plane recall and the context router
  context_graph.py   3-plane graph, sessions, communities
  planes.py          cross-plane wiring and supersession
  code_digest.py     AST function bodies across 22+ languages
  ingest_all.py      one-shot project absorb
  graph_query.py     structural graph queries
eval/                gold set, ablation runner, LOCOMO + LongMemEval harnesses
studio-ui/           React + Vite frontend (Studio, About, Architecture)
corpus/              generated seeded documents
docs/                ARCHITECTURE.md, PRD and design notes
```

Built on **dg-core** (DecisionGraph memory engine), with feature extraction from
markitdown (Microsoft), PaddleOCR, turbovec (TurboQuant), GitNexus (triple-clash pattern)
and gbrain (rerank + gap-analysis patterns).
