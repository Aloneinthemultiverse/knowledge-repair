import type { ReactNode } from 'react'

export function Section({ title, hint, children, action }: {
  title: string; hint?: string; children: ReactNode; action?: ReactNode
}) {
  return (
    <section className="grid gap-3 min-w-0">
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div className="grid gap-0.5">
          <h2 className="font-display text-[30px] leading-tight text-white">{title}</h2>
          {hint && <p className="text-mute text-sm max-w-[70ch]">{hint}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  )
}

export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`glass border border-white/[0.08] rounded-xl ${className}`}>{children}</div>
}

const STATUS = {
  applied: { label: 'Applied', cls: 'text-ok border-ok/40 bg-ok/10', mark: '✓' },
  needs_review: { label: 'Review', cls: 'text-warn border-warn/40 bg-warn/10', mark: '!' },
  flagged_only: { label: 'Flagged', cls: 'text-bad border-bad/40 bg-bad/10', mark: '⚑' },
} as const

export function StatusPill({ status }: { status: keyof typeof STATUS }) {
  const s = STATUS[status]
  return (
    <span className={`inline-flex items-center gap-1 whitespace-nowrap rounded border px-1.5 py-0.5 text-xs font-medium ${s.cls}`}>
      <span aria-hidden>{s.mark}</span>{s.label}
    </span>
  )
}

export function Chip({ children }: { children: ReactNode }) {
  return <span className="inline-block whitespace-nowrap rounded bg-sunk px-1.5 py-0.5 font-mono text-xs">{children}</span>
}

export function Table({ head, children }: { head: ReactNode[]; children: ReactNode }) {
  return (
    <div className="glass overflow-x-auto rounded-xl border border-white/[0.08]">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="bg-white/[0.03]">
            {head.map((h, i) => (
              <th key={i} className="whitespace-nowrap border-b border-rule px-3 py-2 text-left font-mono text-[11px] font-medium uppercase tracking-wider text-mute">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="[&_td]:border-b [&_td]:border-rule [&_td]:px-3 [&_td]:py-2 [&_td]:align-top [&_tr:last-child_td]:border-0">
          {children}
        </tbody>
      </table>
    </div>
  )
}

export function ErrorNote({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-wrap items-center gap-3 rounded-md border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad">
      <span>{message}</span>
      {onRetry && <button className="underline underline-offset-2" onClick={onRetry}>Try again</button>}
    </div>
  )
}

export function Skeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="grid gap-2" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }).map((_, i) => <div key={i} className="h-9 animate-pulse rounded bg-sunk" />)}
    </div>
  )
}
