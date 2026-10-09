import { useEffect, useState, type FormEvent } from 'react'
import { api, fmt, type Summary, type Trace as T } from '../api'
import { Chip, ErrorNote, Section, Skeleton, StatusPill, Table } from './ui'

const LINEAGE: Record<string, string> = { original: 'text-mute', repaired: 'text-teal', derived: 'text-purple' }

export function Trace({ s, target }: { s: Summary; target: string }) {
  const [id, setId] = useState(target || s.examples[0] || '')
  const [data, setData] = useState<T | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  async function load(eid: string) {
    if (!eid.trim()) return
    setBusy(true)
    setError('')
    try { setData(await api.trace(s.id, eid.trim())) }
    catch (e) { setData(null); setError((e as Error).message) }
    finally { setBusy(false) }
  }
  useEffect(() => { if (target) { setId(target); load(target) } else if (id) load(id) }, [target]) // eslint-disable-line
  const submit = (e: FormEvent) => { e.preventDefault(); load(id) }
  const show = (c: string, v: string | null | undefined) =>
    c.endsWith('place_id') && v ? `${data?.place_names[v] ?? v}` : fmt(v)

  return (
    <Section title="Trace a repaired entity"
      hint="Every repaired person, place or event lists the original records it came from, which value was taken from where, and every change that touched it.">
      <form onSubmit={submit} className="flex flex-wrap gap-2">
        <label htmlFor="eid" className="sr-only">Entity or record ID</label>
        <input id="eid" value={id} onChange={e => setId(e.target.value)} placeholder="Entity or original record ID, e.g. Q237160"
          className="min-w-0 flex-1 rounded-lg border border-white/15 bg-white/[0.03] px-3 py-2 font-mono text-sm" />
        <button className="rounded-lg bg-white px-4 py-2 text-sm font-medium text-[#06080e]">Trace</button>
      </form>
      <div className="flex flex-wrap items-center gap-2 text-sm text-mute">
        <span>Merged examples:</span>
        {s.examples.map(e => <button key={e} onClick={() => { setId(e); load(e) }} className="font-mono text-teal underline underline-offset-2">{e}</button>)}
      </div>
      {error && <ErrorNote message={error} />}
      {busy && <Skeleton rows={4} />}
      {data && !busy && (
        <div className="grid gap-6">
          <div className="grid gap-2">
            <h3 className="text-sm font-medium text-white/70">{data.originals.length} original record{data.originals.length > 1 ? 's' : ''} → 1 repaired {({ people: 'person', places: 'place', events: 'event' } as Record<string, string>)[data.table]}</h3>
            <Table head={['Row', ...data.columns]}>
              {data.originals.map(o => (
                <tr key={o.record_id} className="text-mute">
                  <td className="whitespace-nowrap font-mono text-xs">{o.record_id}<span className="ml-1 text-white/30">original</span></td>
                  {data.columns.map(c => <td key={c} className={o[c] !== data.repaired[c] ? 'text-coral' : ''}>{show(c, o[c])}</td>)}
                </tr>
              ))}
              <tr className="bg-teal/[0.06]">
                <td className="whitespace-nowrap font-mono text-xs text-teal">{data.entity_id} repaired</td>
                {data.columns.map(c => <td key={c} className="font-medium text-white">{show(c, data.repaired[c])}</td>)}
              </tr>
            </Table>
            <p className="text-xs text-mute">Values in coral differ from the repaired entity.</p>
          </div>
          <div className="grid gap-2">
            <h3 className="text-sm font-medium text-white/70">Where each value came from</h3>
            <Table head={['Field', 'Value', 'Taken from', 'Status']}>
              {data.lineage.map(l => (
                <tr key={l.column}><td>{l.column}</td><td className="text-white">{fmt(l.value)}</td>
                  <td className="font-mono text-xs">{l.from_records || '—'}</td><td className={LINEAGE[l.status] ?? ''}>{l.status}</td></tr>
              ))}
            </Table>
          </div>
          {data.relationships.length > 0 && (
            <div className="grid gap-2">
              <h3 className="text-sm font-medium text-white/70">Relationships</h3>
              <Table head={['Relation', 'From', 'To', 'Year']}>
                {data.relationships.map((r, i) => (
                  <tr key={i}><td><Chip>{r.rel}</Chip></td><td>{r.src_name}</td><td>{r.dst_name}</td><td className="font-mono tabular">{fmt(r.year)}</td></tr>
                ))}
              </Table>
            </div>
          )}
          <div className="grid gap-2">
            <h3 className="text-sm font-medium text-white/70">Changes that touched these records</h3>
            {data.actions.length === 0 ? <p className="text-sm text-mute">No changes: these records were already consistent.</p> : (
              <Table head={['Status', 'Action', 'Field', 'Before → after', 'Reason']}>
                {data.actions.map(a => (
                  <tr key={a.action_id}><td><StatusPill status={a.status} /></td><td><Chip>{a.kind}</Chip></td><td>{fmt(a.col)}</td>
                    <td className="font-mono text-xs">{fmt(a.before)} → <span className="text-white">{fmt(a.after)}</span></td>
                    <td className="min-w-[16rem] text-mute">{a.explanation}</td></tr>
                ))}
              </Table>
            )}
          </div>
        </div>
      )}
    </Section>
  )
}
