# Benchmark data

| folder | source | in repo |
|---|---|---|
| `wikidata_kb/` | Wikidata SPARQL (Nobel laureates), rebuilt into 4 tables by `python -m kb.load` | yes |
| `hospital/`, `flights/`, `beers/`, `rayyan/` | [Raha/Baran datasets](https://github.com/BigDaMa/raha/tree/master/datasets) | yes |
| `tax/`, `movies_1/` | same source, large | no: download `dirty.csv` and `clean.csv` from the link above |
| FEBRL | bundled with the `recordlinkage` package | no download needed |
