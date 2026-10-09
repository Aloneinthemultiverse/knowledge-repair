import type { Summary } from '../api'
import { api } from '../api'
import { Panel, Section, Table } from './ui'

const DIMS: [keyof Summary['quality']['before'], string, string][] = [
  ['completeness', 'Completeness', 'required fields that hold a value'],
  ['validity', 'Validity', 'filled values that are well-formed'],
  ['uniqueness', 'Uniqueness', 'records that are not copies'],
  ['consistency', 'Consistency', 'entities that break no cross-record rule'],
  ['integrity', 'Integrity', 'references that point to a real record'],
]
const TABLES = ['people', 'places', 'events', 'relationships'] as const
const FILES = ['repaired_people.csv', 'repaired_places.csv', 'repaired_events.csv', 'repaired_relationships.csv',
  'lineage.csv', 'actions.csv', 'issues.csv']

function Meter({ before, after }: { before: number; after: number }) {
  const pct = (x: number) => `${Math.max(0, Math.min(1, x)) * 100}%`
  return (
    <div className="relative h-2 rounded bg-sunk" aria-hidden>
      <div className="absolute inset-y-0 left-0 rounded bg-mute/40" style={{ width: pct(before) }} />
      <div className="absolute inset-y-0 left-0 rounded bg-accent" style={{ width: pct(after), opacity: 0.85 }} />
    </div>
  )
}

export function Overview({ s }: { s: Summary }) {
  const q = s.quality
  const kinds = [...new Set(s.issues.map(i => i.kind))]
  const count = (t: string, k: string) => s.issues.find(i => i.table === t && i.kind === k)?.count
  const vKeys = [...new Set([...Object.keys(q.before.violations), ...Object.keys(q.after.violations)])]
  return (
    <div className="grid gap-10">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Panel className="grid gap-1 p-4">
          <span className="text-xs text-mute">Quality score</span>
          <span className="font-mono text-2xl tabular">{q.before.DQ.toFixed(3)} → <span className="text-ok">{q.after.DQ.toFixed(3)}</span></span>
        </Panel>
        <Panel className="grid gap-1 p-4">
          <span className="text-xs text-mute">Changes applied</span>
          <span className="font-mono text-2xl tabular">{s.actions.applied}<span className="ml-2 text-sm text-warn">+{s.actions.needs_review} to review</span></span>
        </Panel>
        <Panel className="grid gap-1 p-4">
          <span className="text-xs text-mute">Flagged, left unchanged</span>
          <span className="font-mono text-2xl tabular text-bad">{s.actions.flagged_only}</span>
        </Panel>
        <Panel className="grid gap-1 p-4">
          <span className="text-xs text-mute">People: records → entities</span>
          <span className="font-mono text-2xl tabular">{s.records.people[0]} → {s.records.people[1]}</span>
        </Panel>
      </div>

      <Section title="Data quality" hint="Measured without a clean copy: the same checks run on the input (grey) and the repaired output (green).">
        <Panel className="grid gap-4 p-4">
          {DIMS.map(([k, label, what]) => (
            <div key={k} className="grid gap-1.5 sm:grid-cols-[180px_1fr_120px] sm:items-center sm:gap-4">
              <div><div className="text-sm font-medium">{label}</div><div className="text-xs text-mute">{what}</div></div>
              <Meter before={q.before[k] as number} after={q.after[k] as number} />
              <div className="font-mono text-sm tabular sm:text-right">{(q.before[k] as number).toFixed(3)} → {(q.after[k] as number).toFixed(3)}</div>
            </div>
          ))}
        </Panel>
      </Section>

      <div className="grid gap-10 lg:grid-cols-2">
        <Section title="Issues found" hint="By table and type. Contradictions include conflicts between related records.">
          <Table head={['Table', ...kinds]}>
            {TABLES.map(t => (
              <tr key={t}><td className="font-medium">{t}</td>
                {kinds.map(k => <td key={k} className="font-mono tabular">{count(t, k) ?? ''}</td>)}</tr>
            ))}
          </Table>
        </Section>
        <Section title="Cross-record rules" hint="Rule violations before and after repair.">
          <Table head={['Rule', 'Input', 'Repaired']}>
            {vKeys.map(k => (
              <tr key={k}><td>{k}</td><td className="font-mono tabular">{q.before.violations[k] ?? 0}</td>
                <td className="font-mono tabular">{q.after.violations[k] ?? 0}</td></tr>
            ))}
          </Table>
        </Section>
      </div>

      <Section title="Records" hint="Duplicates merge into one entity; every entity keeps the IDs of the records it came from.">
        <Table head={['Table', 'Input records', 'Repaired']}>
          {TABLES.map(t => <tr key={t}><td>{t}</td><td className="font-mono tabular">{s.records[t][0]}</td><td className="font-mono tabular">{s.records[t][1]}</td></tr>)}
        </Table>
        <div className="flex flex-wrap gap-2">
          {FILES.map(f => (
            <a key={f} href={api.downloadUrl(s.id, f)} className="rounded border border-rule bg-surface px-2.5 py-1 font-mono text-xs hover:border-accent">{f}</a>
          ))}
        </div>
      </Section>
    </div>
  )
}
