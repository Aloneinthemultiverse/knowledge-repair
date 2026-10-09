import { useId, useMemo } from 'react'

function mapEaseToCSS(ease: any): string {
  if (Array.isArray(ease) && ease.length === 4) return `cubic-bezier(${ease.join(',')})`
  switch (ease) {
    case 'linear': return 'linear'
    case 'easeIn': return 'ease-in'
    case 'easeOut': return 'ease-out'
    case 'easeInOut': return 'ease-in-out'
    case 'circOut': return 'cubic-bezier(0.075,0.82,0.165,1)'
    case 'backOut': return 'cubic-bezier(0.175,0.885,0.32,1.275)'
    default: return 'ease-in-out'
  }
}

interface Props {
  words?: string
  color?: string
  font?: Record<string, any>
  transition?: { duration?: number; delay?: number; ease?: any }
  tag?: any
}

export default function TextMorph(props: Props = {}) {
  const words = props.words ?? 'TRUTHGUARD\nANSWER\nCLARIFY\nREFUSE'
  const color = props.color ?? '#ffffff'
  const font = props.font ?? { fontFamily: 'Instrument Serif', fontSize: 72, lineHeight: '1.2em', textAlign: 'left' }
  const transition = props.transition ?? { duration: 1, delay: 1.4, ease: 'easeInOut' }
  const Tag = (props.tag ?? 'div') as any

  const morph = Math.max(0.1, transition.duration ?? 1)
  const hold = Math.max(0, transition.delay ?? 1)
  const easeCSS = mapEaseToCSS(transition.ease ?? 'easeInOut')

  const wordList = useMemo(() => words.split(/\r?\n|,/).map(w => w.trim()).filter(Boolean), [words])
  const rawId = useId()
  const safeId = rawId.replace(/[:]/g, '')
  const filterId = `tm-thr-${safeId}`
  const animName = `tm-rot-${safeId}`

  const count = Math.max(1, wordList.length)
  const slot = morph + hold
  const cycle = slot * count
  const pct = (s: number) => Math.min(100, (s / cycle) * 100).toFixed(4)
  const mIn = pct(morph), mHold = pct(morph + hold), mOut = pct(2 * morph + hold)

  const keyframes = `
@keyframes ${animName} {
  0% { opacity:0; filter:blur(20px); transform:translate(-50%,-50%) scale(0.8); }
  ${mIn}% { opacity:1; filter:blur(0px); transform:translate(-50%,-50%) scale(1); }
  ${mHold}% { opacity:1; filter:blur(0px); transform:translate(-50%,-50%) scale(1); }
  ${mOut}%, 100% { opacity:0; filter:blur(20px); transform:translate(-50%,-50%) scale(1.2); }
}`

  const textAlign: string = (font as any)?.textAlign ?? 'center'
  const fontStyle = Object.fromEntries(Object.entries(font).filter(([k]) => k !== 'textAlign'))
  const longest = wordList.reduce((a, w) => (w.length > a.length ? w : a), '')
  const justify = textAlign === 'left' ? 'flex-start' : textAlign === 'right' ? 'flex-end' : 'center'

  return (
    <Tag style={{ position: 'relative', width: '100%', height: '100%', display: 'flex',
      justifyContent: justify, alignItems: 'center', overflow: 'hidden', userSelect: 'none' }}>
      <style>{keyframes}</style>
      <svg style={{ position: 'absolute', width: 0, height: 0, pointerEvents: 'none' }} aria-hidden>
        <defs><filter id={filterId}>
          <feColorMatrix in="SourceGraphic" type="matrix"
            values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 25 -9" result="goo" />
          <feComposite in="SourceGraphic" in2="goo" operator="atop" />
        </filter></defs>
      </svg>
      <div style={{ position: 'relative', filter: `url(#${filterId})`, height: '100%',
        display: 'flex', justifyContent: justify, alignItems: 'center',
        textAlign: textAlign as any, ...fontStyle }}>
        <div style={{ position: 'relative', display: 'inline-flex', justifyContent: 'center',
          alignItems: 'center', lineHeight: 1.2, minHeight: '1.2em' }}>
          <span style={{ visibility: 'hidden', whiteSpace: 'nowrap', display: 'inline-block' }}>{longest || ' '}</span>
          {wordList.map((word, i) => (
            <span key={`${word}-${i}`} style={{
              position: 'absolute', top: '50%', left: '50%',
              transform: 'translate(-50%,-50%)', opacity: 0, color, whiteSpace: 'nowrap',
              animation: `${animName} ${cycle}s ${(slot * i).toFixed(3)}s infinite ${easeCSS}`,
              willChange: 'opacity, filter, transform',
            }}>{word}</span>
          ))}
        </div>
      </div>
    </Tag>
  )
}
