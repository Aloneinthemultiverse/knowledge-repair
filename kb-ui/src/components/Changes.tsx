import { useEffect, useState } from 'react'
import { api, fmt, type Action, type Summary } from '../api'
import { Chip, ErrorNote, Section, Skeleton, StatusPill, Table } from './ui'

const PAGE = 50

export function Changes({ s, onTrace }: { s: Summary; onTrace: (id: string) => void }) {
  const [f, setF] = useState({ table: '', kind: '', status: '', q: '' })
  const [page, setPage] = useState(0)
  const [data, setData] = useState<{ total: number; kinds: string[]; rows: Action[] } | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let live = true
    setError('')
    const t = setTimeout(() => {
      api.actions(s.id, { ...f, limit: PAGE, offset: page * PAGE })
        .then(d => live && setData(d))
        .catch(e => live && setError(e.message))
    }, 150)
    return () => { live = false; clearTimeout(t) }
  }, [s.id, f, page])

  const set = (k: keyof typeof f) => (e: { target: { value: string } }) => { setPage(0); setF({ ...f, [k]: e.target.value }) }
  const select = 'rounded border border-rule bg-surface px-2 py-1.5 text-sm'
  const pages = data ? Math.ceil(data.total / PAGE) : 0

  return (
    <Section title="Every change, with its reason"
      hint="Applied at confidence 0.90 or higher, applied and marked for review from 0.60, and only flagged below that. Click a record ID to trace it.">
      <div className="flex flex-wrap gap-2">
        <label className="sr-only" htmlFor="ft">Table</label>
        <select id="ft" className={select} value={f.table} onChange={set('table')}>
          <option value="">All tables</option>{['people', 'places', 'events', 'relationships'].map(t => <option key={t}>{t}</option>)}
        </select>
        <label className="sr-only" htmlFor="fk">Action</label>
        <select id="fk" className={select} value={f.kind} onChange={set('kind')}>
          <option value="">All actions</option>{data?.kinds.map(k => <option key={k}>{k}</option>)}
        </select>
        <label className="sr-only" htmlFor="fs">Status</label>
        <select id="fs" className={select} value={f.status} onChange={set('status')}>
          <option value="">All statuses</option><option value="applied">Applied</option>
          <option value="needs_review">Needs review</option><option value="flagged_only">Flagged only</option>
        </select>
        <label className="sr-only" htmlFor="fq">Search</label>
        <input id="fq" className={`${select} min-w-0 flex-1`} placeholder="Search a name, ID or reason" value={f.q} onChange={set('q')} />
      </div>
      {error && <ErrorNote message={error} />}
      {!data ? <Skeleton rows={6} /> : data.total === 0 ? (
        <p role="status" className="rounded border border-rule bg-surface p-6 text-center text-mute">No changes match these filters.</p>
      ) : (
        <>
          <Table head={['Status', 'Table', 'Action', 'Records', 'Field', 'Before → after', 'Conf.', 'Reason']}>
            {data.rows.map(a => (
              <tr key={a.action_id}>
                <td><StatusPill status={a.status} /></td>
                <td>{a.table}</td>
                <td><Chip>{a.kind}</Chip></td>
                <td className="max-w-[16rem]">
                  <div className="flex flex-wrap gap-1">
                    {a.records.split(',').slice(0, 4).map(r => (
                      <button key={r} onClick={() => onTrace(r)} className="font-mono text-xs text-accent underline underline-offset-2">{r}</button>
                    ))}
                  </div>
                </td>
                <td className="whitespace-nowrap">{fmt(a.col)}</td>
                <td className="max-w-[18rem] font-mono text-xs">{fmt(a.before)} → <span className="text-ink font-medium">{fmt(a.after)}</span></td>
                <td className="font-mono tabular">{a.confidence.toFixed(2)}</td>
                <td className="min-w-[16rem] text-mute">{a.explanation}</td>
              </tr>
            ))}
          </Table>
          <div className="flex items-center gap-3 text-sm">
            <span className="text-mute">{data.total} changes · page {page + 1} of {pages}</span>
            <button className="rounded border border-rule px-2 py-1 disabled:opacity-40" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
            <button className="rounded border border-rule px-2 py-1 disabled:opacity-40" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Next</button>
          </div>
        </>
      )}
    </Section>
  )
}
