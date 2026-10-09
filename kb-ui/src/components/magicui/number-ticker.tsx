import { useEffect, useRef } from 'react'
import { useInView, useMotionValue, useSpring } from 'motion/react'
export function NumberTicker({ value, className }:{ value:number; className?:string }) {
 const ref = useRef<HTMLSpanElement>(null)
 const mv = useMotionValue(0)
 const spring = useSpring(mv, { damping:60, stiffness:120 })
 const inView = useInView(ref, { once:false })
 useEffect(()=>{ if(inView) mv.set(value) }, [mv, inView, value])
 useEffect(()=> spring.on('change', (l)=>{ if(ref.current) ref.current.textContent = Math.round(l).toLocaleString() }), [spring])
 return <span ref={ref} className={className}>0</span>
}
