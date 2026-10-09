import { useState, type FormEvent, type ReactNode } from 'react'
import { api, type Summary } from '../api'
import { BlurFade } from './magicui/blur-fade'
import { BorderBeam } from './magicui/border-beam'
import { NumberTicker } from './magicui/number-ticker'
import TextMorph from './TextMorph'
import { ErrorNote } from './ui'

const PROBLEMS = [
  ['P1', 'Duplicates', 'The same laureate appears twice: "Alan J. Heeger" and "ALAN J. HEEGER". Each copy carries different facts, so the AI answers from whichever it finds first.'],
  ['P2', 'Typos', '"Artuhr Holly Compton", "Gneeva", "Nobbel Prize". A misspelled name is a record the search never matches.'],
  ['P3', 'Missing values', 'Birth dates, birthplaces and countries replaced by "N/A" or left blank. The fact is gone unless another record still holds it.'],
  ['P4', 'Contradictions', 'Born in 1896 but awarded in 2000. Died before being born. An award year that disagrees with its own event. Related records that cannot all be true.'],
]
const STEPS: [string, string, string, boolean][] = [
  ['01', 'Places', 'Fix country typos, merge "Stanford" with "Stanford, United States", and keep real namesakes like Cambridge UK and Cambridge US apart.', false],
  ['02', 'Events', 'Settle each award year by vote: the event label, its copies, and every award link that points to it.', false],
  ['03', 'People', 'Merge copies by name and shared birthday; keep the birth year that gives a possible age at the award; flag what cannot be proven.', true],
  ['04', 'Relationships', 'Re-point links to merged entities, correct award years, restore missing parent and child links, flag broken references.', false],
]
const BENCH: [string, number, boolean][] = [
  ['Ours (no labels)', 0.846, true], ['Baran + pretraining', 0.83, false], ['Baran', 0.81, false], ['HoloClean', 0.11, false],
]

function Section({ id, children }: { id?: string; children: ReactNode }) {
  return <section id={id} className="border-t border-white/[0.06] py-20"><div className="mx-auto max-w-[880px] px-6">{children}</div></section>
}
const Eyebrow = ({ children }: { children: ReactNode }) =>
  <div className="mb-4 text-[11.5px] font-medium uppercase tracking-[0.18em] text-white/40">{children}</div>
const H3 = ({ children }: { children: ReactNode }) =>
  <h3 className="font-display mb-6 text-[40px] leading-[1.12] text-white">{children}</h3>
const P = ({ children }: { children: ReactNode }) =>
  <p className="mb-5 max-w-[70ch] text-[17px] leading-[1.75] text-white/55">{children}</p>

const FILES = [
  ['people', 'person_id, name, birth_date, death_date, gender, birth_place_id, death_place_id'],
  ['places', 'place_id, name, country'],
  ['events', 'event_id, name, prize, year, place_id'],
  ['relationships', 'src, rel, dst, year'],
]

export function Start({ onRun }: { onRun: (s: Summary) => void }) {
  const [busy, setBusy] = useState<'sample' | 'upload' | null>(null)
  const [error, setError] = useState('')

  async function run(kind: 'sample' | 'upload', form?: FormData) {
    setBusy(kind)
    setError('')
    try { onRun(kind === 'sample' ? await api.sample() : await api.upload(form!)) }
    catch (e) { setError((e as Error).message) }
    finally { setBusy(null) }
  }
  const submit = (e: FormEvent<HTMLFormElement>) => { e.preventDefault(); run('upload', new FormData(e.currentTarget)) }

  return (
    <div>
      <div className="mx-auto max-w-[880px] px-6 pb-16 pt-24">
        <div className="-mt-4 mb-1 h-[86px]">
          <TextMorph words={'FORGOT\nMERGE\nREPAIR\nREMEMBER'} color="rgb(235,242,252)"
            transition={{ duration: 0.9, delay: 1.6, ease: 'easeInOut' }}
            font={{ fontFamily: "'Instrument Serif', serif", fontSize: 60, lineHeight: '1.2em', textAlign: 'left' }} />
        </div>
        <BlurFade inView={false}>
          <div className="mb-6 text-[11.5px] font-medium uppercase tracking-[0.18em] text-teal">Problem 7 · The AI that forgot everything</div>
        </BlurFade>
        <BlurFade inView={false} delay={0.08}>
          <h1 className="font-display mb-7 text-[52px] leading-[0.98] text-white sm:text-[84px]">
            A knowledge base<br /><span className="shimmer">that repairs itself</span>
          </h1>
        </BlurFade>
        <BlurFade inView={false} delay={0.16}>
          <p className="max-w-[700px] text-[19px] leading-[1.7] text-white/55">
            When the records about people, places, events and relationships decay, the AI that reads them starts answering
            wrong. This tool finds duplicates, typos, missing values and contradictions between related records,
            <span className="text-white/90"> fixes only what the data proves</span>, flags the rest, and links every
            repaired value back to the rows it came from.
          </p>
        </BlurFade>
        <BlurFade inView={false} delay={0.24}>
          <div className="mt-9 flex flex-wrap gap-3">
            <button onClick={() => run('sample')} disabled={busy !== null}
              className="relative overflow-hidden rounded-lg bg-white px-5 py-2.5 text-[14.5px] font-medium text-[#06080e] transition hover:bg-white/90 disabled:opacity-60">
              {busy === 'sample' ? 'Repairing…' : 'Repair the sample knowledge base →'}
            </button>
            <a href="#upload" className="rounded-lg border border-white/15 px-5 py-2.5 text-[14.5px] text-white/80 transition hover:border-white/40">Upload your own</a>
          </div>
          {error && <div className="mt-5"><ErrorNote message={error} /></div>}
        </BlurFade>
      </div>

      <Section id="problem">
        <Eyebrow>The problem</Eyebrow>
        <H3>An AI is only as right as the records it reads</H3>
        <P>We took a real knowledge base of 654 Nobel laureates from Wikidata and corrupted it the way databases decay. Asked about the damaged facts, an assistant answered correctly only 35% of the time.</P>
        <div className="mt-8 grid gap-3 sm:grid-cols-2">
          {PROBLEMS.map(([k, t, d], i) => (
            <BlurFade key={k} delay={i * 0.06}>
              <div className="h-full rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
                <div className="mb-2 flex items-baseline gap-3"><span className="font-mono text-xs text-coral">{k}</span><b className="text-white">{t}</b></div>
                <p className="text-[14.5px] leading-[1.65] text-white/50">{d}</p>
              </div>
            </BlurFade>
          ))}
        </div>
      </Section>

      <Section id="how">
        <Eyebrow>How it works</Eyebrow>
        <H3>Each table is repaired with evidence from the ones before it</H3>
        <div className="mt-8 grid gap-3">
          {STEPS.map(([n, t, d, hl], i) => (
            <BlurFade key={n} delay={i * 0.06}>
              <div className={`relative grid gap-4 overflow-hidden rounded-xl border p-5 sm:grid-cols-[60px_160px_1fr] ${hl ? 'border-teal/30 bg-teal/[0.04]' : 'border-white/[0.08] bg-white/[0.02]'}`}>
                {hl && <BorderBeam size={120} duration={8} />}
                <span className="font-mono text-sm text-white/30">{n}</span>
                <b className="text-white">{t}</b>
                <p className="text-[14.5px] leading-[1.65] text-white/55">{d}</p>
              </div>
            </BlurFade>
          ))}
        </div>
        <P><br />Every change gets a confidence. At 0.90 or higher it is applied, from 0.60 it is applied and marked for review, and below that nothing changes: the issue is flagged with the evidence.</P>
      </Section>

      <Section id="results">
        <Eyebrow>Results</Eyebrow>
        <H3>Measured against the real data</H3>
        <div className="grid gap-3 sm:grid-cols-3">
          {[
            [<><NumberTicker value={286} /> questions</>, 'about damaged facts: 35.0% correct before repair, 62.6% after'],
            [<><NumberTicker value={14} /> cells</>, 'of 35,458 correct ones changed by mistake across 5 runs (0.04%)'],
            [<><NumberTicker value={570} /> errors</>, 'injected per run into people, places, events and relationships'],
          ].map(([v, d], i) => (
            <div key={i} className="rounded-xl border border-white/[0.08] bg-white/[0.02] p-5">
              <div className="font-display text-[34px] text-white">{v}</div>
              <p className="text-[14px] text-white/50">{d}</p>
            </div>
          ))}
        </div>
        <div className="mt-10">
          <Eyebrow>Tax benchmark (200,000 person records) · repair F1, VLDB 2020</Eyebrow>
          <div className="grid gap-2.5">
            {BENCH.map(([n, v, ours]) => (
              <div key={n} className="grid grid-cols-[150px_1fr_52px] items-center gap-3 text-[14px] sm:grid-cols-[190px_1fr_60px]">
                <span className={ours ? 'text-white' : 'text-white/50'}>{n}</span>
                <div className="h-2.5 rounded bg-white/[0.05]"><div className={`h-full rounded ${ours ? 'bg-teal' : 'bg-white/25'}`} style={{ width: `${v * 100}%` }} /></div>
                <span className="font-mono tabular text-white/70">{v.toFixed(3)}</span>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[13px] text-white/40">Baran needs about 20 hand-labelled rows per dataset; this tool uses none. It is behind Baran on Flights and Rayyan.</p>
        </div>
      </Section>

      <Section id="upload">
        <Eyebrow>Your data</Eyebrow>
        <H3>Repair your own knowledge base</H3>
        <form onSubmit={submit} className="grid gap-4 rounded-xl border border-white/[0.08] bg-white/[0.02] p-6">
          {FILES.map(([id, cols]) => (
            <label key={id} htmlFor={`f-${id}`} className="grid gap-1.5 text-sm">
              <span className="font-medium text-white">{id}.csv</span>
              <input id={`f-${id}`} name={id} type="file" accept=".csv" required
                className="text-sm text-white/70 file:mr-3 file:rounded-md file:border file:border-white/15 file:bg-white/5 file:px-3 file:py-1 file:text-white/80" />
              <span className="font-mono text-[11px] text-white/35">{cols}</span>
            </label>
          ))}
          <button type="submit" disabled={busy !== null}
            className="justify-self-start rounded-lg border border-teal/50 px-5 py-2.5 text-[14.5px] text-teal transition hover:bg-teal/10 disabled:opacity-60">
            {busy === 'upload' ? 'Repairing…' : 'Repair my files'}
          </button>
        </form>
      </Section>
    </div>
  )
}
