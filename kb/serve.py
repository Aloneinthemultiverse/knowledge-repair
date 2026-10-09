"""Upload page: drop in the 4 CSVs, get the repaired knowledge base and its report.

    python -m kb.serve            # http://127.0.0.1:8765
"""
import io
import os
import tempfile
import zipfile

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .__main__ import TABLES, repair_and_export

app = FastAPI(title="Knowledge Repair")
REQUIRED = {"people": ["person_id", "name", "birth_date", "death_date", "gender", "birth_place_id", "death_place_id"],
            "places": ["place_id", "name", "country"], "events": ["event_id", "name", "prize", "year", "place_id"],
            "relationships": ["src", "rel", "dst", "year"]}
RUNS = os.path.join(tempfile.gettempdir(), "kb_runs")
SAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out_kb_bench", "seed7")

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Knowledge Repair</title>
<style>
:root{--bg:#f6f7f9;--surface:#fff;--fg:#17202b;--muted:#5b6675;--line:#dde2e8;--accent:#1f5fae;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#11161d;--surface:#18202a;--fg:#e6ebf1;--muted:#9aa6b4;--line:#2b3642;--accent:#7fb0f0;--bad:#f08a80;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:720px;margin:0 auto;padding:40px 16px;display:grid;gap:20px}
h1{margin:0;font-size:26px}p{margin:0;color:var(--muted)}
form{display:grid;gap:12px;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:20px}
label{display:grid;gap:4px;font-weight:500}
input[type=file]{font:inherit}
.row{display:flex;gap:10px;flex-wrap:wrap}
button{font:inherit;padding:9px 16px;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:var(--bg);cursor:pointer}
button.alt{background:transparent;color:var(--accent)}
button:focus-visible{outline:2px solid var(--fg);outline-offset:2px}
#status{min-height:1.5em}.err{color:var(--bad)}
code{font-size:13px}
</style></head><body><main>
<h1>Knowledge Repair</h1>
<p>Upload a knowledge base as four CSVs. The tool finds duplicates, missing values, typos and contradictions between related records, repairs what the data proves, flags the rest, and links every repaired value to its original rows.</p>
<form id="f">
<label>people.csv <input type="file" id="people" name="people" accept=".csv" required></label>
<label>places.csv <input type="file" id="places" name="places" accept=".csv" required></label>
<label>events.csv <input type="file" id="events" name="events" accept=".csv" required></label>
<label>relationships.csv <input type="file" id="relationships" name="relationships" accept=".csv" required></label>
<div class="row"><button type="submit">Repair</button><button type="button" class="alt" id="sample">Use the sample corrupted Nobel KB</button></div>
<div id="status" role="status"></div>
</form>
<p>Columns: see <code>kb/README.md</code>. Results open as a report; all output files download as one zip.</p>
</main><script>
const st=document.getElementById('status');
async function go(req){st.textContent='Repairing…';st.className='';
 try{const r=await req;const j=await r.json();if(!r.ok)throw new Error(j.detail||'Repair failed');
  st.innerHTML=`Quality ${j.dq_before} → ${j.dq_after} · ${j.issues} issues · ${j.actions} actions · ${j.seconds}s — <a href="/runs/${j.run}/report.html" target="_blank">open report</a> · <a href="/runs/${j.run}/download">download all files</a>`;}
 catch(e){st.textContent=e.message+'. Check that each file has the columns listed in kb/README.md.';st.className='err';}}
document.getElementById('f').addEventListener('submit',e=>{e.preventDefault();go(fetch('/repair',{method:'POST',body:new FormData(e.target)}));});
document.getElementById('sample').addEventListener('click',()=>go(fetch('/repair-sample',{method:'POST'})));
</script></body></html>"""


def _finish(kb):
    os.makedirs(RUNS, exist_ok=True)
    run = next(tempfile._get_candidate_names())
    out = os.path.join(RUNS, run)
    rep, q = repair_and_export(kb, out)
    return {"run": run, "dq_before": q["before"]["DQ"], "dq_after": q["after"]["DQ"],
            "issues": len(rep.issues), "actions": len(rep.actions), "seconds": q["seconds"]}


def _frame(raw):
    df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    return df.where(df != "", None)


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/repair")
async def repair(people: UploadFile = File(...), places: UploadFile = File(...),
                 events: UploadFile = File(...), relationships: UploadFile = File(...)):
    try:
        kb = {t: _frame(await f.read()) for t, f in zip(TABLES, (people, places, events, relationships))}
        missing = {t: sorted(set(c) - set(kb[t].columns)) for t, c in REQUIRED.items()}
        missing = {t: m for t, m in missing.items() if m}
        if missing:
            msg = "; ".join(f"{t}.csv is missing {', '.join(m)}" for t, m in missing.items())
            return JSONResponse({"detail": msg}, status_code=400)
        return _finish(kb)
    except Exception as e:                                   # bad columns / malformed CSV
        return JSONResponse({"detail": f"Could not repair: {e}"}, status_code=400)


@app.post("/repair-sample")
def repair_sample():
    kb = {t: _frame(open(os.path.join(SAMPLE, f"dirty_{t}.csv"), "rb").read()) for t in TABLES}
    return _finish(kb)


@app.get("/runs/{run}/report.html")
def report(run: str):
    return FileResponse(os.path.join(RUNS, os.path.basename(run), "report.html"))


@app.get("/runs/{run}/download")
def download(run: str):
    d = os.path.join(RUNS, os.path.basename(run))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(d):
            z.write(os.path.join(d, f), f)
    path = os.path.join(d, "..", f"{os.path.basename(run)}.zip")
    with open(path, "wb") as fh:
        fh.write(buf.getvalue())
    return FileResponse(path, filename="knowledge_repair_output.zip")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
