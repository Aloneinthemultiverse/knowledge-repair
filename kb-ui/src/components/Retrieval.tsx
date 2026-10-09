import type { Retrieval as R } from '../api'

/** What the retriever did for one side: turbovec engine, index size, latency, ranked hits. */
export function Retrieval({ r, chosen }: { r: R; chosen: string | null }) {
  const ratio = r.index_kb ? r.float32_kb / r.index_kb : 0
  return (
    <div className="grid gap-2 rounded-lg border border-white/[0.06] bg-white/[0.02] p-3 text-xs">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-white/55">
        <span className="font-mono text-teal">{r.engine}</span>
        <span>{r.vectors} vectors</span>
        <span>index {r.index_kb} KB{ratio > 1 && <> · {ratio.toFixed(1)}× smaller than float32 ({r.float32_kb} KB)</>}</span>
        <span>{r.query_ms} ms</span>
      </div>
      <ol className="grid gap-1">
        {r.hits.map(h => (
          <li key={h.id} className={`grid grid-cols-[18px_1fr_auto] items-center gap-2 ${h.id === chosen ? 'text-white' : 'text-white/45'}`}>
            <span className="font-mono">{h.rank}</span>
            <span className="truncate">{h.name}{h.id === chosen && <span className="ml-1.5 text-teal">← answer</span>}</span>
            <span className="font-mono tabular">{h.vector_score === null ? 'keyword' : `cos ${h.vector_score.toFixed(3)}`}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}
