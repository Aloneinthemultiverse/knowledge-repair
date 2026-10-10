import { useEffect, useMemo, useRef, useState } from 'react'
import { api, fmt, type Summary, type Timeline } from '../api'
import { Chip, ErrorNote, Panel, Section, Skeleton, StatusPill } from './ui'

type Event = { phase: 'corrupt' | 'repair'; stage: string; text: string; kind: string; status?: 'applied' | 'needs_review' | 'flagged_only' }

/** Plays the corruption (sample only) and then the repair, stage by stage. Quality is real at every
 *  stage boundary; between boundaries the meter moves linearly through that stage's events. */
export function Simulation({ s }: { s: Summary }) {
  const [tl, setTl] = useState<Timeline | null>(null)
  const [error, setError] = useState('')
  const [i, setI] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(40)
  const feed = useRef<HTMLOListElement>(null)

  useEffect(() => { api.timeline(s.id).then(setTl).catch(e => setError(e.message)) }, [s.id])

  const { events, marks } = useMemo(() => {
    const ev: Event[] = []
    const mk: { at: number; quality: number; label: string }[] = []
    if (!tl) return { events: ev, marks: mk }
    if (tl.corruption) {
      mk.push({ at: 0, quality: tl.corruption.clean_quality, label: 'Clean Wikidata' })
      // the injection log is grouped by error type; interleave it (fixed order) so it plays like real decay
      const errs = tl.corruption.errors.map((e, n) => ({ e, key: (n * 2654435761) % 4294967296 })).sort((a, b) => a.key - b.key).map(x => x.e)
      for (const e of errs)
        ev.push({ phase: 'corrupt', stage: e.table, kind: e.kind, text: `${e.name}${e.col ? ` · ${e.col}: ${fmt(e.old)} → ${fmt(e.new)}` : ''}` })
    }
    mk.push({ at: ev.length, quality: tl.repair[0].quality, label: 'Corrupted' })
    for (const st of tl.repair.slice(1)) {
      for (const a of st.actions)
        ev.push({ phase: 'repair', stage: st.label, kind: a.kind, status: a.status, text: `${a.col ? `${a.col}: ` : ''}${a.explanation}` })
      mk.push({ at: ev.length, quality: st.quality, label: st.label })
    }
    return { events: ev, marks: mk }
  }, [tl])

  useEffect(() => {
    if (!playing) return
    if (i >= events.length) { setPlaying(false); return }
    const t = setTimeout(() => setI(n => Math.min(events.length, n + Math.max(1, Math.round(speed / 20)))), 1000 / speed * Math.max(1, Math.round(speed / 20)))
    return () => clearTimeout(t)
  }, [playing, i, events.length, speed])
  useEffect(() => { feed.current?.scrollTo({ top: feed.current.scrollHeight }) }, [i])

  if (error) return <ErrorNote message={error} />
  if (!tl) return <Skeleton rows={5} />

  const k = marks.findIndex(m => m.at >= i)
  const hi = marks[Math.max(0, k)] ?? marks[marks.length - 1]
  const lo = marks[Math.max(0, k - 1)] ?? hi
  const quality = hi.at === lo.at ? hi.quality : lo.quality + (hi.quality - lo.quality) * (i - lo.at) / (hi.at - lo.at)
  const cur = events[Math.max(0, i - 1)]
  const shown = events.slice(Math.max(0, i - 80), i)
  const counts = events.slice(0, i).reduce<Record<string, number>>((m, e) => ((m[`${e.phase}:${e.kind}`] = (m[`${e.phase}:${e.kind}`] || 0) + 1), m), {})
  const qMin = Math.min(...marks.map(m => m.quality)) - 0.005
  const pct = (x: number) => `${((x - qMin) / (1 - qMin)) * 100}%`

  return (
    <Section title="Watch it break, then watch it heal"
      hint={tl.corruption ? 'First the real Wikidata knowledge base is corrupted error by error, then the repair runs stage by stage. Quality is measured at every stage boundary.'
        : 'The repair runs stage by stage on your data. Quality is measured at every stage boundary.'}>
      <Panel className="grid gap-4 p-4">
        <div className="flex flex-wrap items-center gap-3">
          <button onClick={() => { if (i >= events.length) setI(0); setPlaying(p => !p) }}
            className="rounded-lg bg-white px-4 py-1.5 text-sm font-medium text-[#06080e]">{playing ? 'Pause' : i >= events.length ? 'Replay' : i ? 'Resume' : 'Play'}</button>
          <button onClick={() => { setPlaying(false); setI(0) }} className="rounded-lg border border-white/15 px-3 py-1.5 text-sm text-white/70">Restart</button>
          <label htmlFor="sp" className="flex items-center gap-2 text-xs text-mute">Speed
            <input id="sp" type="range" min={10} max={400} value={speed} onChange={e => setSpeed(Number(e.target.value))} /></label>
          <span className="ml-auto font-mono text-xs text-mute">{i} / {events.length} events</span>
        </div>
        <div className="grid gap-1.5">
          <div className="flex items-baseline justify-between">
            <span className={`text-sm ${cur?.phase === 'corrupt' ? 'text-coral' : 'text-teal'}`}>{!i ? 'Ready' : cur.phase === 'corrupt' ? 'Corrupting the knowledge base' : `Repairing: ${cur.stage}`}</span>
            <span className="font-mono text-3xl text-white">{quality.toFixed(4)}</span>
          </div>
          <div className="relative h-3 rounded bg-white/[0.06]">
            <div className={`absolute inset-y-0 left-0 rounded transition-[width] duration-150 ${cur?.phase === 'corrupt' ? 'bg-coral' : 'bg-teal'}`} style={{ width: pct(quality) }} />
            {marks.map(m => <div key={m.label} title={`${m.label}: ${m.quality.toFixed(4)}`} className="absolute -top-1 h-5 w-px bg-white/40" style={{ left: pct(m.quality) }} />)}
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-mute">
            {marks.map(m => <span key={m.label} className={i >= m.at ? 'text-white/70' : ''}>{m.label} {m.quality.toFixed(3)}</span>)}
          </div>
        </div>
      </Panel>
      <div className="grid gap-4 lg:grid-cols-[1fr_240px]">
        <Panel className="min-w-0 p-2">
          <ol ref={feed} className="grid max-h-[380px] gap-0.5 overflow-y-auto" aria-live="off">
            {shown.map((e, n) => (
              <li key={i - shown.length + n} className="flex items-start gap-2 rounded px-2 py-1 text-[13px] animate-rise">
                {e.phase === 'corrupt' ? <span className="whitespace-nowrap rounded border border-coral/40 bg-coral/10 px-1.5 py-0.5 text-xs text-coral">✕ Error</span> : <StatusPill status={e.status!} />}
                <Chip>{e.kind}</Chip><span className="min-w-0 flex-1 text-white/60">{e.text}</span>
              </li>
            ))}
            {!i && <li className="p-6 text-center text-sm text-mute">Press Play.</li>}
          </ol>
        </Panel>
        <Panel className="grid content-start gap-1 p-3 text-sm">
          <div className="mb-1 text-xs text-mute">So far</div>
          {Object.entries(counts).sort().map(([key, n]) => (
            <div key={key} className="flex justify-between gap-2"><span className={key.startsWith('corrupt') ? 'text-coral/90' : 'text-teal/90'}>{key.split(':')[1].toLowerCase()}</span><span className="font-mono tabular text-white/70">{n}</span></div>
          ))}
        </Panel>
      </div>
    </Section>
  )
}
