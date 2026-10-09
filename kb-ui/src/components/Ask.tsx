import { useEffect, useState, type FormEvent } from 'react'
import { api, type AskResult, type Side, type Summary } from '../api'
import { AiSentence } from './AiSentence'
import { Retrieval } from './Retrieval'
import { ErrorNote, Panel, Section } from './ui'

const SAMPLE_QUESTIONS = [
  'When was Alan J. Heeger born?',
  'When was Katalin Karikó born?',
  'When was Robert Andrews Millikan born?',
  'Where was Giulio Natta born?',
  'When was Daniel Nathans born?',
  'In which year did Richard Feynman receive the Nobel Prize in Physics?',
]

function Answer({ side, title, tone, onTrace, run, question }: {
  side: Side; title: string; tone: 'before' | 'after'; onTrace: (id: string) => void; run: string; question: string
}) {
  const border = tone === 'after' ? 'border-accent/50' : 'border-rule'
  return (
    <Panel className={`grid content-start gap-3 p-4 ${border}`}>
      <div className="font-mono text-[11px] uppercase tracking-wider text-mute">{title}</div>
      {side.not_found ? (
        <div className="grid gap-1">
          <div className="text-lg">No one by that name in this knowledge base.</div>
          <div className="text-sm text-mute">Closest: {side.alternatives.map(a => a.name).join(', ')}</div>
        </div>
      ) : (
        <>
          <div className="font-display text-3xl font-semibold leading-tight">{side.answer ?? 'No value recorded'}</div>
          {side.conflicting.length > 0 && (
            <div className="rounded border border-warn/40 bg-warn/10 px-2.5 py-1.5 text-sm text-warn">
              Another record for this person says {side.conflicting.join(', ')}.
            </div>
          )}
          <p className="text-sm text-mute"><span className="font-medium text-ink">{side.name}</span> · {side.record}</p>
          {tone === 'after' && side.record_id && (
            <button onClick={() => onTrace(side.record_id!)} className="justify-self-start text-sm text-accent underline underline-offset-2">
              Show where this came from
            </button>
          )}
        </>
      )}
      {!side.not_found && <AiSentence run={run} question={question} side={tone} />}
      {side.retrieval && <Retrieval r={side.retrieval} chosen={side.record_id} />}
    </Panel>
  )
}

export function Ask({ s, onTrace }: { s: Summary; onTrace: (id: string) => void }) {
  const [ready, setReady] = useState(false)
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[0])
  const [result, setResult] = useState<AskResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let live = true
    const poll = async () => {
      try {
        const r = await api.askReady(s.id)
        if (!live) return
        if (r.ready) setReady(true)
        else setTimeout(poll, 1500)
      } catch (e) {
        if (live) setError((e as Error).message)
      }
    }
    poll()
    return () => { live = false }
  }, [s.id])

  async function ask(q: string) {
    setQuestion(q)
    setBusy(true)
    setError('')
    try {
      setResult(await api.ask(s.id, q))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault()
    ask(question)
  }

  const fixed = result && !result.before.not_found && result.before.answer !== result.after.answer

  return (
    <Section title="Ask the knowledge base"
      hint="The same turbovec retriever (4-bit quantized vectors fused with BM25 keywords) answers from the corrupted and from the repaired knowledge base. The value is read from the retrieved record; a free LLM then phrases it, and its sentence is kept only if it states that same value. Any difference between the two sides comes from the repair.">
      <form onSubmit={submit} className="flex flex-wrap gap-2">
        <label htmlFor="q" className="sr-only">Question</label>
        <input id="q" value={question} onChange={e => setQuestion(e.target.value)}
          className="min-w-0 flex-1 rounded border border-rule bg-surface px-3 py-2" placeholder="When was Alan J. Heeger born?" />
        <button disabled={!ready || busy} className="rounded bg-accent px-4 py-2 font-medium text-surface disabled:opacity-60">
          {!ready ? 'Indexing…' : busy ? 'Asking…' : 'Ask'}
        </button>
      </form>
      {!ready && <p className="text-sm text-mute" role="status">Building the two search indexes (about 30 seconds on first use).</p>}
      {s.label.startsWith('Sample') && (
        <div className="flex flex-wrap gap-2">
          {SAMPLE_QUESTIONS.map(q => (
            <button key={q} disabled={!ready || busy} onClick={() => ask(q)}
              className="rounded-full border border-rule bg-surface px-3 py-1 text-sm hover:border-accent disabled:opacity-50">{q}</button>
          ))}
        </div>
      )}
      {error && <ErrorNote message={error} onRetry={() => ask(question)} />}
      {result && (
        <div className="grid gap-3">
          <p className="text-sm text-mute">Asked for: <span className="text-ink">{result.asked_for}</span>
            {fixed && <span className="ml-2 font-medium text-ok">The repair changed this answer.</span>}</p>
          <div className="grid gap-3 md:grid-cols-2">
            <Answer side={result.before} title="Corrupted knowledge base" tone="before" onTrace={onTrace} run={s.id} question={result.question} />
            <Answer side={result.after} title="Repaired knowledge base" tone="after" onTrace={onTrace} run={s.id} question={result.question} />
          </div>
        </div>
      )}
    </Section>
  )
}
