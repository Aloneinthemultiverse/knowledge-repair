# Knowledge Repair Tool (Problem 7)

Repairs a corrupted knowledge base of **people, places, events and relationships**: finds duplicates,
missing values, typos and contradictions between related records, repairs what the data can prove,
flags the rest, and keeps every repaired value traceable to the original rows.

## Run on your own data

Put four CSVs in a folder:

| file | columns |
|---|---|
| `people.csv` | person_id, name, birth_date (YYYY-MM-DD), death_date, gender, birth_place_id, death_place_id |
| `places.csv` | place_id, name, country |
| `events.csv` | event_id, name, prize, year, place_id |
| `relationships.csv` | src, rel (`received`, `spouse`, `father`, `mother`, `child`, `sibling`), dst, year |

```bash
python -m kb run path/to/folder --out out_kb
```

Outputs in `out_kb/`:

| file | contents |
|---|---|
| `repaired_<table>.csv` | repaired tables; each entity lists its `source_records` |
| `lineage.csv` | every repaired value, the original records it came from, and whether it is original, repaired or derived |
| `issues.csv` | every problem found (table, kind, records, detail) |
| `actions.csv` | every change or flag: before, after, confidence, status, rule, reason |
| `quality.json` | data-quality score before and after (no clean copy needed) |
| `report.html` | readable report with filters and an entity-to-original-records lookup |

## Benchmark

```bash
python -m kb bench                    # real Wikidata Nobel-laureate KB, 5 seeds of injected errors
python -m pytest tests/test_kb.py     # 9 tests
```

## How it decides

- Order: places → events → people → relationships; each step uses the already-cleaned tables as evidence.
- A change is applied at confidence ≥ 0.90, applied and marked for review at 0.60–0.90, and only flagged below 0.60.
- Values are never invented: missing data with no evidence elsewhere stays empty and is reported.
- Input records are never modified.

Modules: `load.py` (Wikidata → 4 tables), `corrupt.py` (seeded error injection with a log),
`repair.py` (repair + lineage), `dq.py` (quality without ground truth), `score.py` (quality against ground truth),
`report.py` (HTML report).
