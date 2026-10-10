export type Quality = {
  completeness: number; validity: number; uniqueness: number; consistency: number; integrity: number; DQ: number
  violations: Record<string, number>
}
export type Summary = {
  id: string; label: string
  quality: { before: Quality; after: Quality }
  records: Record<'people' | 'places' | 'events' | 'relationships', [number, number]>
  issues: { table: string; kind: string; count: number }[]
  actions: { applied: number; needs_review: number; flagged_only: number; total: number }
  examples: string[]
}
export type Action = {
  action_id: string; table: string; kind: string; records: string; col: string | null
  before: unknown; after: unknown; confidence: number; status: 'applied' | 'needs_review' | 'flagged_only'
  rule: string; explanation: string
}
export type Side = {
  answer: string | null; record_id: string | null; record: string | null; name: string | null
  conflicting: string[]; not_found: boolean; alternatives: { id: string; name: string }[]
  retrieval: Retrieval
}
export type Retrieval = {
  engine: string; vectors: number; index_kb: number; float32_kb: number; query_ms: number
  hits: { id: string; name: string; rank: number; vector_score: number | null }[]
}
export type Explanation = { sentence: string | null; model: string | null; status: 'ok' | 'rejected' | 'unavailable' | 'error'; note: string | null }
export type Person = { person_id: string; name: string | null; birth_date: string | null; death_date: string | null; gender: string | null; birth_place_id: string | null; death_place_id: string | null }
export type EditResult = {
  summary: Summary; trace: Trace; actions: (Action & { new: boolean })[]; repair_ms: number
  edited: { table: string; record_id: string; before: Record<string, unknown>; after: Record<string, unknown>; duplicate_of?: string }
}
export type Stage = { stage: string; label: string; what?: string; quality: number; actions: Action[] }
export type Timeline = {
  corruption: null | { clean_quality: number; errors: { table: string; kind: string; id: string; name: string; col: string | null; old: string | null; new: string | null }[] }
  repair: Stage[]
}
export type AskResult = { question: string; asked_for: string; before: Side; after: Side }
export type Trace = {
  table: string; entity_id: string; columns: string[]
  repaired: Record<string, string | null>
  originals: Record<string, string | null>[]
  lineage: { column: string; value: string | null; from_records: string; status: string }[]
  relationships: { rel: string; src: string; src_name: string; dst: string; dst_name: string; year: string | null }[]
  actions: Action[]
  place_names: Record<string, string>
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, init)
  } catch {
    throw new Error('Cannot reach the repair service. Start it with: python -m kb.api')
  }
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`)
  return body as T
}

export const api = {
  sample: () => call<Summary>('/api/runs/sample', { method: 'POST' }),
  upload: (form: FormData) => call<Summary>('/api/runs', { method: 'POST', body: form }),
  actions: (id: string, p: Record<string, string | number>) =>
    call<{ total: number; kinds: string[]; rows: Action[] }>(
      `/api/runs/${id}/actions?` + new URLSearchParams(Object.entries(p).map(([k, v]) => [k, String(v)]))),
  trace: (id: string, eid: string) => call<Trace>(`/api/runs/${id}/trace/${encodeURIComponent(eid)}`),
  askReady: (id: string) => call<{ ready: boolean }>(`/api/runs/${id}/ask/ready`),
  ask: (id: string, question: string) => call<AskResult>(`/api/runs/${id}/ask`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }),
  }),
  explain: (id: string, question: string, side: 'before' | 'after') => call<Explanation>(`/api/runs/${id}/explain`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, side }),
  }),
  llm: () => call<{ available: boolean }>('/api/llm'),
  records: (id: string, q: string) => call<{ total: number; rows: Person[]; places: { id: string; name: string }[] }>(`/api/runs/${id}/records?q=${encodeURIComponent(q)}`),
  edit: (id: string, table: string, rid: string, changes: Record<string, string | null>) => call<EditResult>(`/api/runs/${id}/records/${table}/${encodeURIComponent(rid)}`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ changes }),
  }),
  duplicate: (id: string, rid: string) => call<EditResult>(`/api/runs/${id}/records/people/${encodeURIComponent(rid)}/duplicate`, { method: 'POST' }),
  reset: (id: string) => call<Summary>(`/api/runs/${id}/reset`, { method: 'POST' }),
  index: (id: string) => call<{ ready: boolean; stale: boolean }>(`/api/runs/${id}/index`),
  timeline: (id: string) => call<Timeline>(`/api/runs/${id}/timeline`),
  downloadUrl: (id: string, name: string) => `/api/runs/${id}/download/${name}`,
}

export const fmt = (v: unknown): string =>
  v === null || v === undefined || v === '' ? '—' : typeof v === 'object' ? JSON.stringify(v) : String(v)
