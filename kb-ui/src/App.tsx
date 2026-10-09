import { useEffect, useState } from 'react'
import type { Summary } from './api'
import { Ask } from './components/Ask'
import { Changes } from './components/Changes'
import { KineticGrid } from './components/magicui/kinetic-grid'
import { Overview } from './components/Overview'
import { Start } from './components/Start'
import { Trace } from './components/Trace'

const TABS = [['overview', 'Overview'], ['ask', 'Ask the AI'], ['changes', 'Changes'], ['trace', 'Trace']] as const
type Tab = typeof TABS[number][0]

function Logo() {
  return (
    <div className="relative h-6 w-6 rounded-md" style={{ background: 'conic-gradient(from 210deg,#39d2c0,#bc8cff,#ff8c66,#39d2c0)' }}>
      <div className="absolute inset-[5px] rounded-[3px] bg-[#06080e]" />
    </div>
  )
}

export default function App() {
  const [run, setRun] = useState<Summary | null>(null)
  const [tab, setTab] = useState<Tab>('overview')
  const [target, setTarget] = useState('')
  const trace = (id: string) => { setTarget(id); setTab('trace') }
  useEffect(() => { window.scrollTo({ top: 0 }) }, [run, tab])

  return (
    <div className="relative min-h-screen">
      <div className="pointer-events-none fixed inset-0 z-0">
        <KineticGrid dotColor="#7f93b3" lineColor="#39d2c0" trailColor="#bc8cff" spacing={34} radius={260} strength={4} />
      </div>
      <div className="relative z-10">
        <nav className="sticky top-0 z-30 border-b border-white/[0.06] bg-[#06080e]/85 backdrop-blur-xl">
          <div className="mx-auto flex h-[60px] max-w-[1120px] items-center gap-3 px-6">
            <button onClick={() => setRun(null)} className="flex items-center gap-3" aria-label="Back to start">
              <Logo /><b className="text-[15px] font-semibold text-white">Knowledge Repair</b>
            </button>
            {run ? (
              <div className="ml-auto flex items-center gap-1 overflow-x-auto text-[13.5px]" role="tablist">
                {TABS.map(([k, label]) => (
                  <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}
                    className={`whitespace-nowrap rounded-md px-3 py-1.5 transition ${tab === k ? 'bg-white/10 text-white' : 'text-white/45 hover:text-white'}`}>
                    {label}
                  </button>
                ))}
              </div>
            ) : (
              <div className="ml-auto hidden gap-7 text-[13.5px] text-white/45 sm:flex">
                <a href="#problem" className="transition hover:text-white">Problem</a>
                <a href="#how" className="transition hover:text-white">How it works</a>
                <a href="#results" className="transition hover:text-white">Results</a>
                <a href="#upload" className="transition hover:text-white">Upload</a>
              </div>
            )}
          </div>
        </nav>
        {!run ? <Start onRun={s => { setRun(s); setTab('overview') }} /> : (
          <main className="mx-auto grid max-w-[1120px] gap-8 px-6 py-10">
            <div className="grid gap-1">
              <div className="text-[11.5px] font-medium uppercase tracking-[0.18em] text-teal">Repair run {run.id}</div>
              <h1 className="font-display text-[44px] leading-tight text-white">{run.label}</h1>
            </div>
            {tab === 'overview' && <Overview s={run} />}
            {tab === 'ask' && <Ask s={run} onTrace={trace} />}
            {tab === 'changes' && <Changes s={run} onTrace={trace} />}
            {tab === 'trace' && <Trace s={run} target={target} />}
          </main>
        )}
      </div>
    </div>
  )
}
