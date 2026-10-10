# Sample input: corrupted Nobel laureate knowledge base

Real Wikidata data (see `bench/wikidata_kb/clean/`) with about 570 injected errors (seed 7):
duplicate people and places under different names, typos, missing values, wrong years and
broken references. `injected_errors.csv` lists every injected error.

Upload these four files in the web app, or run:

```bash
python -m kb run sample_input --out out_kb
```
