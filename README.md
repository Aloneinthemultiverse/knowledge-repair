# Knowledge Repair Tool

**Problem 7 · The AI That Forgot Everything**

Repairs a corrupted knowledge base of **people, places, events and relationships**. It finds duplicate
records (including the same entity under different names), missing and inconsistent values, typos, and
contradictions between related records. It repairs what the data proves, flags what it cannot prove, and
links every repaired value back to the original rows. No labelled data and no AI model are needed, and every
change states its rule and reason.

## Results

| Benchmark | Result |
|---|---|
| Real Wikidata KB (Nobel laureates, 4 linked tables), relationship F1 | 0.929 → **0.990** |
| Same KB, people duplicate F1 | **0.995** (precision 1.00) |
| Same KB, quality score without a clean copy | 0.96 → **0.99** |
| Correct cells broken by the repair | **0.04%** |
| Assistant answering questions about damaged facts | 35.0% → **62.6%** correct |
| Tax / Beers / Hospital repair F1 (Baran, VLDB 2020: 0.81 / 0.90 / 0.87) | **0.846 / 0.940 / 0.902** |
| FEBRL person-record duplicates F1 (classic ECM: 0.996 / 0.997) | **0.996 / 0.996** |

Where it is behind: Flights (0.677 vs Baran 1.00) and Rayyan (0.00 vs 0.52), which need hand labels.
Full details and sources: [benchmark report](out_repair/benchmark_report.html).

## Quick start

```bash
pip install -r requirements.txt
python -m kb bench --seeds 7            # real KB -> inject errors -> repair -> score (writes out_kb_bench/)
python -m kb run out_kb_bench/seed7 --out out_kb   # or point it at your own 4 CSVs
python -m kb.serve                      # upload page at http://127.0.0.1:8765
python -m pytest tests/test_kb.py       # 9 tests
```

`run` writes repaired tables, `lineage.csv`, `issues.csv`, `actions.csv`, `quality.json` and `report.html`.
Input format: [kb/README.md](kb/README.md). Demo script and pitch: [DEMO.md](DEMO.md).

## Layout

| path | contents |
|---|---|
| `kb/` | the knowledge repair tool (load, corrupt, repair, quality, report, CLI, upload page, Q&A demo) |
| `repair/` | single-table benchmarks (Raha/Baran datasets, FEBRL) and the first people-only engine |
| `bench/` | benchmark data (see `bench/README.md`) |
| `tests/` | automated tests |

This repository started from the TruthGuard RAG project; its original README is in
[TRUTHGUARD_README.md](TRUTHGUARD_README.md).
