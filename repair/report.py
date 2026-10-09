"""Self-contained HTML repair report: scores, issues, explained actions, lineage lookup."""
import json

import pandas as pd


def _records(df):
    return json.loads(df.astype(object).where(pd.notna(df), None).to_json(orient="records"))


def build(path, metrics, rep, dirty):
    data = {
        "metrics": metrics,
        "issues": _records(pd.DataFrame(rep.issues)),
        "actions": _records(pd.DataFrame(rep.actions)),
        "entities": _records(rep.entities),
        "dirty": _records(dirty),
    }
    html = TEMPLATE.replace("/*DATA*/", json.dumps(data, default=str).replace("</", "<\\/"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Knowledge Repair Report</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--mut:#6b6b66;--line:#e4e3de;--acc:#2f6fdb;
--ok:#1f8a4c;--warn:#b7791f;--bad:#c53030;--chip:#efeee9}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--card:#1e1e1c;--fg:#ecebe6;
--mut:#9d9c96;--line:#33332f;--acc:#6fa0f0;--ok:#4cc27f;--warn:#e0a84a;--bad:#f07070;--chip:#2a2a27}}
:root[data-theme="dark"]{--bg:#141413;--card:#1e1e1c;--fg:#ecebe6;--mut:#9d9c96;--line:#33332f;
--acc:#6fa0f0;--ok:#4cc27f;--warn:#e0a84a;--bad:#f07070;--chip:#2a2a27}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 10px}
.sub{color:var(--mut);margin:0 0 20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.k{color:var(--mut);font-size:12px}.v{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.v small{font-size:13px;color:var(--mut);font-weight:400}
.up{color:var(--ok)}.bad{color:var(--bad)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line)}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top;font-size:13px}
th{color:var(--mut);font-weight:500;background:var(--chip)}
.wrap{overflow-x:auto;border-radius:10px}
.chip{display:inline-block;padding:1px 8px;border-radius:99px;background:var(--chip);font-size:12px}
.applied{color:var(--ok)}.needs_review{color:var(--warn)}.flagged_only{color:var(--bad)}
.bar{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 10px}
select,input{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit}
.bars div{display:grid;grid-template-columns:130px 1fr 80px;gap:8px;align-items:center;margin:4px 0}
.track{height:10px;background:var(--chip);border-radius:5px;position:relative}
.fill{position:absolute;height:10px;border-radius:5px;background:var(--acc)}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:12px}
a{color:var(--acc)}
</style></head><body><main>
<h1>Knowledge Repair Report</h1>
<p class="sub">Every change links back to the original record IDs. Nothing in the source data was overwritten.</p>
<div class="grid" id="kpis"></div>
<h2>Issues detected vs errors injected</h2>
<div class="card bars" id="bars"></div>
<h2>Data quality without ground truth</h2>
<div class="wrap"><table id="dq"></table></div>
<h2>Actions taken</h2>
<div class="bar"><select id="fk"></select><select id="fs"></select><input id="fq" placeholder="search record / text"></div>
<div class="wrap"><table id="acts"></table></div><p class="sub" id="actn"></p>
<h2>Lineage: repaired entity &rarr; original records</h2>
<div class="bar"><input id="le" placeholder="entity id (E00012) or record id (R00031)" size="40"></div>
<div id="lin"></div>
</main>
<script>
const D=/*DATA*/;const $=s=>document.querySelector(s);
const esc=v=>v==null?'<span class="k">null</span>':String(v).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const m=D.metrics.truth_metrics,pct=x=>(x*100).toFixed(1)+'%';
$('#kpis').innerHTML=[
 ['Cell accuracy',`${pct(m.cell_accuracy_before)} &rarr; <span class="up">${pct(m.cell_accuracy_after)}</span>`],
 ['Cells fixed',m.cells_fixed],
 ['Cells broken (harm)',`<span class="${m['cells_broken (harm)']?'bad':'up'}">${m['cells_broken (harm)']}</span> <small>/ ${m.cells_total}</small>`],
 ['Duplicate F1',m.dup_f1.toFixed(3)+` <small>P ${m.dup_precision.toFixed(2)} &middot; R ${m.dup_recall.toFixed(2)}</small>`],
 ['Records &rarr; entities',`${m.records_before} &rarr; ${m.entities_after} <small>(true ${m.true_entities})</small>`],
 ['Data-quality score',`${D.metrics.dq_before.DQ.toFixed(3)} &rarr; <span class="up">${D.metrics.dq_after.DQ.toFixed(3)}</span>`]
].map(([k,v])=>`<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
const inj=D.metrics.injected,det=D.metrics.detected,kinds=[...new Set([...Object.keys(inj),...Object.keys(det)])];
const mx=Math.max(...kinds.map(k=>Math.max(inj[k]||0,det[k]||0)));
$('#bars').innerHTML='<div class="k" style="margin-bottom:6px">bar = detected &middot; number = detected / injected. FD_VIOLATION shows up as CONTRADICTION; MISSING also counts placeholders normalised to null.</div>'+
 kinds.map(k=>`<div><span>${k}</span><div class="track"><div class="fill" style="width:${100*(det[k]||0)/mx}%"></div></div><span class="mono">${det[k]||0} / ${inj[k]||0}</span></div>`).join('');
const b=D.metrics.dq_before,a=D.metrics.dq_after;
$('#dq').innerHTML='<tr><th>dimension</th><th>before</th><th>after</th></tr>'+Object.keys(b).map(k=>`<tr><td>${k}</td><td class="mono">${b[k].toFixed(3)}</td><td class="mono">${a[k].toFixed(3)}</td></tr>`).join('');
const opt=(el,vals,lab)=>el.innerHTML=`<option value="">${lab}</option>`+vals.map(v=>`<option>${v}</option>`).join('');
opt($('#fk'),[...new Set(D.actions.map(x=>x.kind))],'all kinds');opt($('#fs'),[...new Set(D.actions.map(x=>x.status))],'all statuses');
function drawActs(){const k=$('#fk').value,s=$('#fs').value,q=$('#fq').value.toLowerCase();
 const rows=D.actions.filter(x=>(!k||x.kind==k)&&(!s||x.status==s)&&(!q||JSON.stringify(x).toLowerCase().includes(q)));
 $('#acts').innerHTML='<tr><th>id</th><th>kind</th><th>records</th><th>col</th><th>before &rarr; after</th><th>conf</th><th>status</th><th>why</th></tr>'+
 rows.slice(0,300).map(x=>`<tr><td class="mono">${x.action_id}</td><td><span class="chip">${x.kind}</span></td><td class="mono">${esc(x.records)}</td><td>${esc(x.col)}</td><td class="mono">${esc(x.before)} &rarr; ${esc(x.after)}</td><td class="mono">${x.confidence}</td><td class="${x.status}">${x.status}</td><td>${esc(x.explanation)}</td></tr>`).join('');
 $('#actn').textContent=`${rows.length} actions`+(rows.length>300?' (first 300 shown)':'');}
['#fk','#fs','#fq'].forEach(s=>$(s).addEventListener('input',drawActs));drawActs();
const dirty=Object.fromEntries(D.dirty.map(r=>[r.record_id,r]));
function drawLin(){const q=$('#le').value.trim().toUpperCase();
 const e=D.entities.find(x=>x.entity_id==q)||D.entities.find(x=>x.source_records.split(',').includes(q));
 if(!e){$('#lin').innerHTML='<p class="sub">Type an entity or record id. Merged examples: '+D.entities.filter(x=>x.source_records.includes(',')).slice(0,6).map(x=>`<a href="#" data-e="${x.entity_id}">${x.entity_id}</a>`).join(', ')+'</p>';
  document.querySelectorAll('[data-e]').forEach(a=>a.onclick=ev=>{ev.preventDefault();$('#le').value=a.dataset.e;drawLin();});return}
 const recs=e.source_records.split(','),cols=['name','birth_year','city','country','email'];
 const acts=D.actions.filter(x=>x.records&&x.records.split(',').some(r=>recs.includes(r)||r==e.entity_id));
 $('#lin').innerHTML=`<div class="wrap"><table><tr><th>row</th>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>`+
 recs.map(r=>`<tr><td class="mono">${r} (original)</td>${cols.map(c=>`<td>${esc(dirty[r][c])}</td>`).join('')}</tr>`).join('')+
 `<tr><td class="mono"><b>${e.entity_id} (repaired)</b></td>${cols.map(c=>`<td><b>${esc(e[c])}</b></td>`).join('')}</tr></table></div>
 <h2 style="font-size:14px">Actions touching these records</h2><div class="wrap"><table><tr><th>kind</th><th>col</th><th>before &rarr; after</th><th>status</th><th>why</th></tr>`+
 acts.map(x=>`<tr><td><span class="chip">${x.kind}</span></td><td>${esc(x.col)}</td><td class="mono">${esc(x.before)} &rarr; ${esc(x.after)}</td><td class="${x.status}">${x.status}</td><td>${esc(x.explanation)}</td></tr>`).join('')+'</table></div>';}
$('#le').addEventListener('input',drawLin);drawLin();
</script></body></html>"""
