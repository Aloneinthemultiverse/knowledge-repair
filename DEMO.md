# Knowledge Repair Tool — demo script and pitch

## One-line pitch

> It repairs a corrupted knowledge base of people, places, events and relationships, fixes only what the
> data proves, explains every change, and matches or beats published VLDB research systems with zero labels.

## Setup (before the demo)

```bash
pip install -r requirements.txt fastapi uvicorn python-multipart rapidfuzz jellyfish recordlinkage
python -m kb bench --seeds 7          # creates the sample corrupted KB in out_kb_bench/seed7
python -m kb.serve                    # http://127.0.0.1:8765
```

Have open: the upload page, the published benchmark report, and `out_kb_bench/ask_demo.json`.

## The 3-minute demo

**0:00 — The problem (20 s).**
"This is a real knowledge base from Wikidata: 654 Nobel laureates, the places they were born, 351 award events
and the relationships between them. We corrupted it the way real databases decay: duplicate people under
different names, typos, missing values, and contradictions between related records."

**0:20 — The AI that forgot (40 s).** Show `ask_demo.json`.
"Ask an assistant: *When was Alan J. Heeger born?* From the corrupted database it says **1896**. He was born
in **1936**. Across 286 questions about damaged facts, the assistant is right only **35%** of the time."

**1:00 — Repair (40 s).** On the upload page, click **Use the sample corrupted Nobel KB**.
"About one second. Quality goes from **0.961 to 0.993**, measured without any clean copy. 461 issues found."
Open the report.

**1:40 — Explainability and traceability (40 s).** In the report:
1. Filter actions to **flagged only**: "When the data can't prove the answer, it flags instead of guessing."
2. In *Trace an entity*, click a merged example: "Two original rows, *Alan J. Heeger 1936* and
   *ALAN J. HEEGER 1896*, became one entity. Same birthday, so same person; the 1896 copy gives an impossible
   age at his 2000 award, so 1936 is kept. Every repaired value says which original rows it came from."

**2:20 — The AI remembers (20 s).**
"Same assistant, repaired database: **62.6%** correct, up from 35%. Typo-hit questions go from 18% to 100%."

**2:40 — Credibility (20 s).** Show the benchmark report.
"On the standard benchmarks from VLDB papers we beat Baran and HoloClean on Tax and Beers, match them on
Hospital, and tie the classic record-linkage method on FEBRL, with no labels and 5–25× faster than Baran."

## Answers to likely judge questions

- **Where is the AI?** The hard part is deciding what is true without inventing values. Statistical evidence
  (votes over duplicates, learned column dependencies, cross-record constraints) does that explainably; an
  LLM could hallucinate a plausible but wrong value, which is the problem we are repairing.
- **How do you know a fix is right?** Real Wikidata ground truth: only 14 of 35,458 correct cells were broken
  across 5 runs (0.04%). Every change has a confidence; below 0.60 nothing changes, it is flagged.
- **Can it run on our data?** `python -m kb run <folder>` on four CSVs, or the upload page. Column format is in
  `kb/README.md`.
- **Is it tested?** 9 automated tests (`python -m pytest tests/test_kb.py`).

## Limits (say them before a judge finds them)

- Behind Baran on **Flights** (0.677 vs 1.00) and **Rayyan** (0.00 vs 0.52): those need the ~20 hand labels
  Baran asks for, to learn which sources to trust or how dates were reshuffled.
- Places with identical name and country (Tokyo city vs Tokyo prefecture) cannot be told apart from the record.
- Errors with no evidence anywhere (a missing birth date with no second copy) are reported, not filled.

## Numbers at a glance

| | Result |
|---|---|
| Real KB, relationship F1 | 0.929 → 0.990 |
| Real KB, people duplicate F1 | 0.995 (precision 1.00) |
| Real KB, quality without ground truth | 0.96 → 0.99 |
| Correct cells broken by repair | 0.04% |
| Assistant on damaged facts | 35.0% → 62.6% correct |
| Tax / Beers / Hospital repair F1 | 0.846 / 0.940 / 0.902 (Baran 0.81 / 0.90 / 0.87) |
| FEBRL2 / FEBRL3 duplicate F1 | 0.996 / 0.996 (ECM 0.996 / 0.997) |
| Repair time, full KB | ~1 s on CPU |
