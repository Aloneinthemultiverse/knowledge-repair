"""Self-contained HTML repair report for a 4-table knowledge base."""
import json

import pandas as pd

IDC = {"people": "person_id", "places": "place_id", "events": "event_id"}


def _rec(df):
    df = df.astype(object).where(pd.notna(df), None)
    return json.loads(df.to_json(orient="records", force_ascii=False))


def build(path, kb, rep, quality):
    fixed = rep.output()
    data = {
        "quality": quality,
        "dirty": {t: _rec(kb[t]) for t in kb},
        "fixed": {t: _rec(fixed[t]) for t in fixed},
        "issues": _rec(pd.DataFrame(rep.issues)) if rep.issues else [],
        "actions": _rec(pd.DataFrame(rep.actions)) if rep.actions else [],
        "lineage": _rec(pd.DataFrame(rep.lineage)) if rep.lineage else [],
        "canon": rep.canon,
    }
    blob = json.dumps(data, default=str, ensure_ascii=False).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.replace("/*DATA*/", blob))


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Knowledge Repair Report</title>
<style>
/* Layout: one column; summary tiles, then tables in their own scroll boxes */
:root{--bg:#f6f7f9;--surface:#fff;--fg:#17202b;--muted:#5b6675;--line:#dde2e8;--accent:#1f5fae;
--good:#1e7a46;--warn:#9a6a00;--bad:#b3261e;--tint:#eef3fa;
--body:system-ui,-apple-system,"Segoe UI",sans-serif;--mono:ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#11161d;--surface:#18202a;--fg:#e6ebf1;
--muted:#9aa6b4;--line:#2b3642;--accent:#7fb0f0;--good:#5cc68a;--warn:#e2b44f;--bad:#f08a80;--tint:#1c2836;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#11161d;--surface:#18202a;--fg:#e6ebf1;--muted:#9aa6b4;--line:#2b3642;
--accent:#7fb0f0;--good:#5cc68a;--warn:#e2b44f;--bad:#f08a80;--tint:#1c2836;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 var(--body)}
main{max-width:1150px;margin:0 auto;padding:24px 16px 64px;display:grid;gap:28px}
section{display:grid;gap:10px;min-width:0}
h1{font-size:24px;margin:0}h2{font-size:17px;margin:0}
.sub{color:var(--muted);margin:0;max-width:75ch}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 14px;display:grid;gap:2px}
.tile .v{font:600 20px/1.3 var(--mono);font-variant-numeric:tabular-nums}.tile .k{color:var(--muted);font-size:12px}
.wrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top;font-size:13px}
th{font:500 11px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.05em;color:var(--muted);background:var(--tint);white-space:nowrap}
.mono{font-family:var(--mono);font-size:12px;font-variant-numeric:tabular-nums}
.up{color:var(--good)}.down{color:var(--bad)}
.chip{display:inline-block;padding:1px 8px;border-radius:99px;background:var(--tint);font-size:12px;white-space:nowrap}
.applied{color:var(--good)}.needs_review{color:var(--warn)}.flagged_only{color:var(--bad)}
.controls{display:flex;gap:8px;flex-wrap:wrap}
select,input{background:var(--surface);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit}
select:focus-visible,input:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
a{color:var(--accent)}
.orig td{color:var(--muted)}.rep td{font-weight:600}
</style></head><body><main>
<header style="display:grid;gap:6px"><h1>Knowledge Repair Report</h1>
<p class="sub">Repair of a knowledge base of people, places, events and relationships. Original records are kept unchanged; every repaired entity lists the records it came from, and every change below states its rule and reason.</p></header>
<section><div class="tiles" id="tiles"></div></section>
<section><h2>Data quality (no clean copy needed)</h2>
<p class="sub">The same checks run on the input and on the repaired output.</p>
<div class="wrap"><table id="dq"></table></div></section>
<section><h2>Cross-record rule violations</h2><div class="wrap"><table id="viol"></table></div></section>
<section id="truthsec" hidden><h2>Against the real data (benchmark run)</h2><div class="wrap"><table id="truth"></table></div></section>
<section><h2>Issues found</h2><div class="wrap"><table id="issues"></table></div></section>
<section><h2>Actions taken</h2>
<div class="controls"><select id="ft" aria-label="Table"></select><select id="fk" aria-label="Action"></select><select id="fs" aria-label="Status"></select><input id="fq" placeholder="Search records or text" aria-label="Search"></div>
<div class="wrap"><table id="acts"></table></div><p class="sub" id="actn"></p></section>
<section><h2>Trace an entity back to its original records</h2>
<p class="sub">Enter a repaired entity ID or any original record ID from people, places or events.</p>
<div class="controls"><input id="le" size="34" placeholder="e.g. Q7186 or XP1234567" aria-label="Entity or record ID"></div>
<div id="lin" style="display:grid;gap:10px"></div></section>
</main>
<script>
const D=/*DATA*/;const $=s=>document.querySelector(s);
const esc=v=>v==null?'<span class="sub">empty</span>':String(v).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const Q=D.quality,B=Q.before,A=Q.after,T=['people','places','events','relationships'];
const IDC={people:'person_id',places:'place_id',events:'event_id'};
const cnt=(arr,f)=>arr.reduce((m,x)=>(m[f(x)]=(m[f(x)]||0)+1,m),{});
const st=cnt(D.actions,a=>a.status);
$('#tiles').innerHTML=[
 ['Quality score',`${B.DQ.toFixed(3)} &rarr; <span class="up">${A.DQ.toFixed(3)}</span>`],
 ['Issues found',D.issues.length],
 ['Changes applied',`${st.applied||0} <span class="sub">+ ${st.needs_review||0} to review</span>`],
 ['Flagged, not changed',st.flagged_only||0],
 ['Records &rarr; entities',T.slice(0,3).map(t=>`${Q.records_before[t]}&rarr;${Q.records_after[t]}`).join(' · ')],
 ['Repair time',Q.seconds+' s']
].map(([k,v])=>`<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
const dims=['completeness','validity','uniqueness','consistency','integrity','DQ'];
$('#dq').innerHTML='<tr><th>Dimension</th><th>Input</th><th>Repaired</th><th>Change</th></tr>'+dims.map(d=>{const dlt=A[d]-B[d];
 return `<tr><td>${d==='DQ'?'<b>Overall</b>':d}</td><td class="mono">${B[d].toFixed(3)}</td><td class="mono">${A[d].toFixed(3)}</td><td class="mono ${dlt>=0?'up':'down'}">${dlt>=0?'+':''}${dlt.toFixed(3)}</td></tr>`}).join('');
const vk=[...new Set([...Object.keys(B.violations),...Object.keys(A.violations)])];
$('#viol').innerHTML='<tr><th>Rule</th><th>Input</th><th>Repaired</th></tr>'+(vk.length?vk.map(k=>`<tr><td>${esc(k)}</td><td class="mono">${B.violations[k]||0}</td><td class="mono">${A.violations[k]||0}</td></tr>`).join(''):'<tr><td colspan="3">No violations found.</td></tr>');
if(Q.against_ground_truth){const g=Q.against_ground_truth;$('#truthsec').hidden=false;
 $('#truth').innerHTML='<tr><th>Measure</th><th>Value</th></tr>'+Object.entries(g).filter(([k])=>!['actions','fix_rate_by_error','detected_by_error'].includes(k)).map(([k,v])=>`<tr><td>${esc(k)}</td><td class="mono">${esc(v)}</td></tr>`).join('')+
 Object.entries(g.fix_rate_by_error||{}).map(([k,v])=>`<tr><td>fixed: ${esc(k)}</td><td class="mono">${esc(v)}</td></tr>`).join('');}
const kinds=[...new Set(D.issues.map(i=>i.kind))].sort();
const it=cnt(D.issues,i=>i.table+'|'+i.kind);
$('#issues').innerHTML='<tr><th>Table</th>'+kinds.map(k=>`<th>${k}</th>`).join('')+'</tr>'+T.map(t=>`<tr><td>${t}</td>${kinds.map(k=>`<td class="mono">${it[t+'|'+k]||''}</td>`).join('')}</tr>`).join('');
const opt=(el,vals,lab)=>el.innerHTML=`<option value="">${lab}</option>`+vals.map(v=>`<option>${v}</option>`).join('');
opt($('#ft'),T,'All tables');opt($('#fk'),[...new Set(D.actions.map(a=>a.kind))].sort(),'All actions');opt($('#fs'),['applied','needs_review','flagged_only'],'All statuses');
function drawActs(){const t=$('#ft').value,k=$('#fk').value,s=$('#fs').value,q=$('#fq').value.toLowerCase();
 const rows=D.actions.filter(a=>(!t||a.table==t)&&(!k||a.kind==k)&&(!s||a.status==s)&&(!q||JSON.stringify(a).toLowerCase().includes(q)));
 $('#acts').innerHTML='<tr><th>Table</th><th>Action</th><th>Records</th><th>Field</th><th>Before &rarr; after</th><th>Conf.</th><th>Status</th><th>Reason</th></tr>'+
 rows.slice(0,400).map(a=>`<tr><td>${a.table}</td><td><span class="chip">${a.kind}</span></td><td class="mono">${esc(a.records)}</td><td>${esc(a.col)}</td><td class="mono">${esc(typeof a.before==='object'&&a.before?JSON.stringify(a.before):a.before)} &rarr; ${esc(a.after)}</td><td class="mono">${a.confidence}</td><td class="${a.status}">${a.status.replace('_',' ')}</td><td>${esc(a.explanation)}</td></tr>`).join('');
 $('#actn').textContent=`${rows.length} actions`+(rows.length>400?' (first 400 shown)':'');}
['#ft','#fk','#fs','#fq'].forEach(s=>$(s).addEventListener('input',drawActs));drawActs();
const dirtyBy={},fixedBy={};
for(const t of ['people','places','events']){dirtyBy[t]=Object.fromEntries(D.dirty[t].map(r=>[r[IDC[t]],r]));fixedBy[t]=Object.fromEntries(D.fixed[t].map(r=>[r[IDC[t]],r]));}
function drawLin(){const q=$('#le').value.trim();const box=$('#lin');
 let hit=null;for(const t of ['people','places','events']){const cid=D.canon[t][q];if(cid){hit=[t,cid];break}}
 if(!hit){const ex=D.fixed.people.filter(r=>(r.source_records||'').includes(',')).slice(0,6);
  box.innerHTML='<p class="sub">No match yet. Merged examples: '+ex.map(r=>`<a href="#" data-e="${esc(r.person_id)}">${esc(r.person_id)}</a>`).join(', ')+'</p>';
  box.querySelectorAll('[data-e]').forEach(a=>a.onclick=ev=>{ev.preventDefault();$('#le').value=a.dataset.e;drawLin();});return}
 const [t,cid]=hit,ent=fixedBy[t][cid],srcs=(ent.source_records||cid).split(','),cols=Object.keys(ent).filter(c=>c!=='source_records');
 const lin=D.lineage.filter(l=>l.table==t&&l.entity_id==cid);
 const acts=D.actions.filter(a=>(a.records||'').split(',').some(r=>srcs.includes(r)||r==cid));
 const rels=D.fixed.relationships.filter(r=>r.src==cid||r.dst==cid);
 box.innerHTML=`<div class="wrap"><table><tr><th>Row</th>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>`+
  srcs.map(r=>`<tr class="orig"><td class="mono">${esc(r)} (original)</td>${cols.map(c=>`<td>${esc((dirtyBy[t][r]||{})[c])}</td>`).join('')}</tr>`).join('')+
  `<tr class="rep"><td class="mono">${esc(cid)} (repaired)</td>${cols.map(c=>`<td>${esc(ent[c])}</td>`).join('')}</tr></table></div>`+
  `<div class="wrap"><table><tr><th>Field</th><th>Repaired value</th><th>Taken from</th><th>Status</th></tr>${lin.map(l=>`<tr><td>${l.column}</td><td>${esc(l.value)}</td><td class="mono">${esc(l.from_records)}</td><td>${l.status}</td></tr>`).join('')}</table></div>`+
  (rels.length?`<div class="wrap"><table><tr><th>Relationship</th><th>From</th><th>To</th><th>Year</th></tr>${rels.map(r=>`<tr><td>${r.rel}</td><td class="mono">${esc(r.src)}</td><td class="mono">${esc(r.dst)}</td><td class="mono">${esc(r.year)}</td></tr>`).join('')}</table></div>`:'')+
  `<div class="wrap"><table><tr><th>Action</th><th>Field</th><th>Before &rarr; after</th><th>Status</th><th>Reason</th></tr>${acts.map(a=>`<tr><td><span class="chip">${a.kind}</span></td><td>${esc(a.col)}</td><td class="mono">${esc(typeof a.before==='object'&&a.before?JSON.stringify(a.before):a.before)} &rarr; ${esc(a.after)}</td><td class="${a.status}">${a.status.replace('_',' ')}</td><td>${esc(a.explanation)}</td></tr>`).join('')||'<tr><td colspan="5">No changes touched these records.</td></tr>'}</table></div>`;}
$('#le').addEventListener('input',drawLin);drawLin();
</script></body></html>"""
