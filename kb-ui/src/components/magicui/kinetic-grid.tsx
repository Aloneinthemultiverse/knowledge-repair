import { useRef, useEffect, type CSSProperties } from 'react'

interface KineticGridProps {
  background?: string
  dotColor?: string
  lineColor?: string
  trailColor?: string
  spacing?: number
  radius?: number
  strength?: number
  trail?: boolean
  className?: string
  style?: CSSProperties
}

type Dot = { hx: number; hy: number; x: number; y: number; vx: number; vy: number }

/** Kinetic Grid — reactive dot mesh pulled toward the cursor, with a fading trail. */
export function KineticGrid({
  background = 'transparent',
  dotColor = '#ffffff',
  lineColor = '#39d2c0',
  trailColor = '#39d2c0',
  spacing = 30,
  radius = 320,
  strength = 4,
  trail = true,
  className,
  style,
}: KineticGridProps) {
  const hostRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mouseRef = useRef({ x: -9999, y: -9999, active: false })
  const trailRef = useRef<{ x: number; y: number; t: number }[]>([])

  useEffect(() => {
    const host = hostRef.current
    const canvas = canvasRef.current
    if (!host || !canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const GAP = Math.max(8, spacing)
    const R = Math.max(1, radius)
    const PULL = (Math.max(1, Math.min(10, strength)) / 10) * 4

    let W = 1, H = 1
    let cols: Dot[][] = []
    let dots: Dot[] = []

    const build = (mw?: number, mh?: number) => {
      const r = host.getBoundingClientRect()
      W = Math.max(1, Math.floor(mw ?? r.width))
      H = Math.max(1, Math.floor(mh ?? r.height))
      const dpr = window.devicePixelRatio || 1
      canvas.width = Math.floor(W * dpr)
      canvas.height = Math.floor(H * dpr)
      canvas.style.width = W + 'px'
      canvas.style.height = H + 'px'
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      cols = []; dots = []
      const nCols = Math.floor(W / GAP) + 2
      const nRows = Math.floor(H / GAP) + 2
      for (let c = 0; c < nCols; c++) {
        const col: Dot[] = []
        for (let i = 0; i < nRows; i++) {
          const d = { hx: c * GAP, hy: i * GAP, x: c * GAP, y: i * GAP, vx: 0, vy: 0 }
          col.push(d); dots.push(d)
        }
        cols.push(col)
      }
    }
    build()

    const ro = typeof ResizeObserver !== 'undefined'
      ? new ResizeObserver(e => build(e[0]?.contentRect?.width, e[0]?.contentRect?.height))
      : null
    ro?.observe(host)

    const setMouse = (clientX: number, clientY: number) => {
      const r = canvas.getBoundingClientRect()
      const mx = clientX - r.left, my = clientY - r.top
      mouseRef.current = { x: mx, y: my, active: true }
      const tr = trailRef.current
      tr.push({ x: mx, y: my, t: performance.now() })
      if (tr.length > 80) tr.shift()
    }
    const onMove = (e: MouseEvent) => setMouse(e.clientX, e.clientY)
    const onLeave = () => { mouseRef.current = { x: -9999, y: -9999, active: false } }
    const onTouch = (e: TouchEvent) => { const t = e.touches[0]; if (t) setMouse(t.clientX, t.clientY) }

    // listen on window so the grid reacts even though the canvas is behind content
    window.addEventListener('mousemove', onMove)
    host.addEventListener('mouseleave', onLeave)
    window.addEventListener('touchmove', onTouch, { passive: true })
    window.addEventListener('touchend', onLeave)

    let raf = 0
    const frame = () => {
      const m = mouseRef.current
      ctx.clearRect(0, 0, W, H)

      for (const d of dots) {
        let ax = (d.hx - d.x) * 0.08
        let ay = (d.hy - d.y) * 0.08
        if (m.active) {
          const dx = m.x - d.x, dy = m.y - d.y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < R && dist > 0.001) {
            const f = (1 - dist / R) * PULL
            ax += (dx / dist) * f; ay += (dy / dist) * f
          }
        }
        d.vx = (d.vx + ax) * 0.82; d.vy = (d.vy + ay) * 0.82
        d.x += d.vx; d.y += d.vy
      }

      for (let c = 0; c < cols.length; c++) {
        for (let i = 0; i < cols[c].length; i++) {
          const d = cols[c][i]
          const right = cols[c + 1]?.[i]
          const down = cols[c]?.[i + 1]
          const prox = m.active
            ? Math.max(0, 1 - Math.sqrt((m.x - d.x) ** 2 + (m.y - d.y) ** 2) / R) : 0
          ctx.strokeStyle = lineColor
          if (right) {
            ctx.globalAlpha = 0.04 + prox * 0.55
            ctx.lineWidth = 0.5 + prox * 1.3
            ctx.beginPath(); ctx.moveTo(d.x, d.y); ctx.lineTo(right.x, right.y); ctx.stroke()
          }
          if (down) {
            ctx.globalAlpha = 0.04 + prox * 0.55
            ctx.lineWidth = 0.5 + prox * 1.3
            ctx.beginPath(); ctx.moveTo(d.x, d.y); ctx.lineTo(down.x, down.y); ctx.stroke()
          }
        }
      }

      for (const d of dots) {
        const prox = m.active
          ? Math.max(0, 1 - Math.sqrt((m.x - d.x) ** 2 + (m.y - d.y) ** 2) / R) : 0
        ctx.globalAlpha = 0.14 + prox * 0.7
        ctx.fillStyle = dotColor
        ctx.beginPath(); ctx.arc(d.x, d.y, 0.7 + prox * 2, 0, 2 * Math.PI); ctx.fill()
      }

      if (trail) {
        const now = performance.now()
        const tr = trailRef.current
        ctx.lineCap = 'round'; ctx.lineJoin = 'round'
        for (let i = 1; i < tr.length; i++) {
          const a = tr[i - 1], b = tr[i]
          const age = now - b.t
          if (age > 260) continue
          ctx.globalAlpha = Math.max(0, 1 - age / 260) * 0.7
          ctx.strokeStyle = trailColor; ctx.lineWidth = 2
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke()
        }
      }

      ctx.globalAlpha = 1
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)

    return () => {
      cancelAnimationFrame(raf)
      ro?.disconnect()
      window.removeEventListener('mousemove', onMove)
      host.removeEventListener('mouseleave', onLeave)
      window.removeEventListener('touchmove', onTouch)
      window.removeEventListener('touchend', onLeave)
    }
  }, [background, dotColor, lineColor, trailColor, spacing, radius, strength, trail])

  return (
    <div ref={hostRef} className={className}
      style={{ position: 'absolute', inset: 0, overflow: 'hidden', background, ...style }}>
      <canvas ref={canvasRef}
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />
    </div>
  )
}
