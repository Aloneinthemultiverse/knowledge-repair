import { useEffect, useState } from 'react'
import { api, type Explanation } from '../api'

/** The LLM's one-sentence answer for one side, written only from the retrieved record. */
export function AiSentence({ run, question, side }: { run: string; question: string; side: 'before' | 'after' }) {
  const [x, setX] = useState<Explanation | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let live = true
    setX(null)
    setError('')
    api.explain(run, question, side).then(v => live && setX(v)).catch(e => live && setError(e.message))
    return () => { live = false }
  }, [run, question, side])

  const model = x?.model?.replace(':free', '').split('/').pop()
  return (
    <div className="grid gap-1 rounded-lg border border-purple/25 bg-purple/[0.05] px-3 py-2.5" aria-live="polite">
      <div className="flex items-center justify-between gap-2 text-[11px] uppercase tracking-wider text-purple/80">
        <span>AI answer</span>
        {model && <span className="font-mono normal-case tracking-normal text-white/40">{model} · free</span>}
      </div>
      {error ? <p className="text-sm text-bad">{error}</p>
        : !x ? <p className="animate-pulse text-sm text-white/40">Writing from the retrieved record…</p>
        : x.status === 'ok' ? <p className="text-[15px] text-white/90">“{x.sentence}”</p>
        : <p className="text-sm text-white/50">{x.status === 'rejected' && x.sentence ? <>Not shown: “{x.sentence}”. </> : null}{x.note}</p>}
    </div>
  )
}
