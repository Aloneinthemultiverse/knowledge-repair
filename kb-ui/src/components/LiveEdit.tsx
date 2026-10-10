import { useEffect, useState } from 'react'
import { api, fmt, type AskResult, type EditResult, type Person, type Summary } from '../api'
import { Chip, ErrorNote, Panel, Section, StatusPill } from './ui'

const FIELDS: [keyof Person, string][] = [['name', 'Name'], ['birth_date', 'Birth date'], ['death_date', 'Death date'], ['gender', 'Gender']]

function shiftYear(d: string | null, by: number) {
  return d && /^\d{4}/.test(d) ? `${Number(d.slice(0, 4)) + by}${d.slice(4)}` : '1850-01-01'
}
function typo(n: string | null) {
  if (!n || n.length < 5) return n ?? ''
  const i = Math.floor(n.length / 2)
  return n.slice(0, i) + n[i + 1] + n[i] + n.slice(i + 2)
}

export function LiveEdit({ s, onSummary, onTrace }: { s: Summary; onSummary: (s: Summary) => void; onTrace: (id: string) => void }) {
  const [q, setQ] = useState('Karik')
  const [rows, setRows] = useState<Person[]>([])
  const [places, setPlaces] = useState<{ id: string; name: string }[]>([])
  const [sel, setSel] = useState<Person | null>(null)
  const [draft, setDraft] = useState<Partial<Person>>({})
  const [res, setRes] = useState<EditResult | null>(null)
  const [answer, setAnswer] = useState<AskResult | null>(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    const t = setTimeout(() => api.records(s.id, q).then(d => { setRows(d.rows); setPlaces(d.places) }).catch(e => setError(e.message)), 200)
    return () => clearTimeout(t)
  }, [s.id, q, res])

  function pick(p: Person) { setSel(p); setDraft({ ...p }); setRes(null); setAnswer(null); setError('') }

  async function apply(label: string, run: () => Promise<EditResult>) {
    setBusy(label); setError(''); setAnswer(null)
    try {
      const r = await run()
      setRes(r)
      onSummary(r.summary)
      const name = r.trace.repaired.name ?? sel?.name
      if (name) setAnswer(await api.ask(s.id, `When was ${name} born?`).catch(() => null))
    } catch (e) { setError((e as Error).message) } finally { setBusy('') }
  }
  const edit = (changes: Partial<Person>, label = 'Applying edit') =>
    sel && apply(label, () => api.edit(s.id, 'people', sel.person_id, changes as Record<string, string | null>))

  const changed = sel ? Object.fromEntries(Object.entries(draft).filter(([k, v]) => (sel as Record<string, unknown>)[k] !== v)) : {}
  const q0 = s.quality.after.DQ

  return (
    <Section title="Edit the data, watch it repair"
      hint="Break a record yourself. The full repair re-runs on every edit and shows what it detected, what it changed and why, and what the AI now answers."
      action={<button onClick={async () => { setBusy('Resetting'); const r = await api.reset(s.id); onSummary(r); setRes(null); setAnswer(null); setSel(null); setBusy('') }}
        className="rounded-lg border border-white/15 px-3 py-1.5 text-sm text-white/70 hover:border-white/40">Reset the sample</button>}>
      {error && <ErrorNote message={error} />}
      <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
        <Panel className="grid content-start gap-2 p-3">
          <label htmlFor="ps" className="text-xs text-mute">Find a person</label>
          <input id="ps" value={q} onChange={e => setQ(e.target.value)} className="rounded-lg border border-white/15 bg-white/[0.03] px-3 py-1.5 text-sm" placeholder="Name" />
          <ul className="grid max-h-[420px] gap-0.5 overflow-y-auto">
            {rows.map(p => (
              <li key={p.person_id}>
                <button onClick={() => pick(p)} className={`w-full rounded-md px-2 py-1.5 text-left text-sm ${sel?.person_id === p.person_id ? 'bg-teal/15 text-white' : 'text-white/70 hover:bg-white/5'}`}>
                  {p.name}<span className="ml-2 font-mono text-[11px] text-white/35">{p.birth_date?.slice(0, 4) ?? '—'}</span>
                </button>
              </li>
            ))}
            {!rows.length && <li className="px-2 py-3 text-sm text-mute">No one matches.</li>}
          </ul>
        </Panel>

        {!sel ? <Panel className="grid place-items-center p-10 text-center text-mute">Pick a person on the left to edit their record.</Panel> : (
          <div className="grid content-start gap-4 min-w-0">
            <Panel className="grid gap-3 p-4">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h3 className="font-display text-2xl text-white">{sel.name}</h3><span className="font-mono text-xs text-mute">{sel.person_id}</span>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                {FIELDS.map(([k, label]) => (
                  <label key={k} htmlFor={`e-${k}`} className="grid gap-1 text-xs text-mute">{label}
                    <input id={`e-${k}`} value={(draft[k] as string) ?? ''} onChange={e => setDraft({ ...draft, [k]: e.target.value })}
                      className={`rounded-lg border bg-white/[0.03] px-3 py-1.5 text-sm text-white ${k in changed ? 'border-amber/60' : 'border-white/15'}`} />
                  </label>
                ))}
                <label htmlFor="e-bp" className="grid gap-1 text-xs text-mute sm:col-span-2">Birthplace
                  <select id="e-bp" value={draft.birth_place_id ?? ''} onChange={e => setDraft({ ...draft, birth_place_id: e.target.value || null })}
                    className={`rounded-lg border bg-[#0e131e] px-3 py-1.5 text-sm text-white ${'birth_place_id' in changed ? 'border-amber/60' : 'border-white/15'}`}>
                    <option value="">— none —</option>{places.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                  </select>
                </label>
              </div>
              <div className="flex flex-wrap gap-2">
                <button disabled={!!busy || !Object.keys(changed).length} onClick={() => edit(changed)}
                  className="rounded-lg bg-white px-4 py-1.5 text-sm font-medium text-[#06080e] disabled:opacity-40">{busy === 'Applying edit' ? 'Repairing…' : 'Apply edit'}</button>
                <span className="self-center text-xs text-mute">or break it in one click:</span>
                {[
                  ['Impossible birth year', () => edit({ birth_date: shiftYear(sel.birth_date, -90) }, 'b1')],
                  ['Add a duplicate copy', () => apply('b2', () => api.duplicate(s.id, sel.person_id))],
                  ['Blank the birth date', () => edit({ birth_date: null }, 'b3')],
                  ['Typo in the name', () => edit({ name: typo(sel.name) }, 'b4')],
                ].map(([label, fn]) => (
                  <button key={label as string} disabled={!!busy} onClick={fn as () => void}
                    className="rounded-full border border-coral/40 px-3 py-1 text-xs text-coral hover:bg-coral/10 disabled:opacity-40">{label as string}</button>
                ))}
              </div>
              {busy && <p role="status" className="text-sm text-teal">Re-running the full repair…</p>}
            </Panel>

            {res && (
              <>
                <div className="grid gap-3 sm:grid-cols-3">
                  <Panel className="grid gap-0.5 p-3"><span className="text-xs text-mute">Full repair re-ran in</span><span className="font-mono text-xl text-white">{res.repair_ms} ms</span></Panel>
                  <Panel className="grid gap-0.5 p-3"><span className="text-xs text-mute">Quality after repair</span><span className="font-mono text-xl text-white">{res.summary.quality.after.DQ.toFixed(4)}</span></Panel>
                  <Panel className="grid gap-0.5 p-3"><span className="text-xs text-mute">Input quality with your edit</span><span className="font-mono text-xl text-amber">{res.summary.quality.before.DQ.toFixed(4)}</span></Panel>
                </div>
                <Panel className="grid gap-2 p-4">
                  <h4 className="text-sm font-medium text-white/80">You changed</h4>
                  <p className="font-mono text-xs text-white/70">
                    {res.edited.duplicate_of ? <>added a copy of {res.edited.duplicate_of} named “{fmt(res.edited.after.name)}”</>
                      : Object.keys(res.edited.after).map(k => <span key={k} className="mr-4">{k}: {fmt(res.edited.before[k])} → <span className="text-amber">{fmt(res.edited.after[k])}</span></span>)}
                  </p>
                  <h4 className="mt-2 text-sm font-medium text-white/80">The repair now says</h4>
                  <ul className="grid gap-1.5">
                    {res.actions.map(a => (
                      <li key={a.action_id} className="flex flex-wrap items-start gap-2 text-sm">
                        <StatusPill status={a.status} /><Chip>{a.kind}</Chip>
                        {a.new && <span className="rounded bg-teal/15 px-1.5 text-[11px] text-teal">new</span>}
                        <span className="min-w-0 flex-1 text-white/65">{a.col ? `${a.col}: ` : ''}{a.explanation}</span>
                      </li>
                    ))}
                    {!res.actions.length && <li className="text-sm text-mute">Nothing to fix: the edited record is consistent with the rest of the data.</li>}
                  </ul>
                  <div className="mt-1 text-sm">Repaired record: <b className="text-white">{fmt(res.trace.repaired.name)}</b>, born <b className="text-teal">{fmt(res.trace.repaired.birth_date)}</b>
                    <button onClick={() => onTrace(res.edited.record_id)} className="ml-3 text-teal underline underline-offset-2">trace it</button></div>
                </Panel>
                {answer && (
                  <Panel className="grid gap-2 p-4">
                    <h4 className="text-sm font-medium text-white/80">The AI, asked “{answer.question}”</h4>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <div><div className="text-[11px] uppercase tracking-wider text-mute">from the corrupted data (with your edit)</div><div className="font-display text-3xl text-white/80">{answer.before.answer ?? '—'}</div></div>
                      <div><div className="text-[11px] uppercase tracking-wider text-teal">from the repaired data</div><div className="font-display text-3xl text-white">{answer.after.answer ?? '—'}</div></div>
                    </div>
                  </Panel>
                )}
              </>
            )}
          </div>
        )}
      </div>
      <p className="text-xs text-mute">Quality of the repaired sample before your edits: {q0.toFixed(4)}. Edits update the search records at once; the turbovec index rebuilds in the background so renamed or added people become searchable.</p>
    </Section>
  )
}
