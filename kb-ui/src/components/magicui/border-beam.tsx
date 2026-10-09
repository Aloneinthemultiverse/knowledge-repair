import { motion } from 'motion/react'
import { cn } from '@/lib/utils'
export function BorderBeam({ className, size=50, duration=6, delay=0, colorFrom='#39d2c0', colorTo='#bc8cff' }:
 { className?:string; size?:number; duration?:number; delay?:number; colorFrom?:string; colorTo?:string }) {
 return (<div className="pointer-events-none absolute inset-0 rounded-[inherit] [border:1px_solid_transparent] ![mask-clip:padding-box,border-box] ![mask-composite:intersect] [mask:linear-gradient(transparent,transparent),linear-gradient(#000,#000)]">
  <motion.div className={cn('absolute aspect-square','bg-gradient-to-l from-[var(--cf)] via-[var(--ct)] to-transparent', className)}
   style={{ width:size, offsetPath:`rect(0 auto auto 0 round ${size}px)`, '--cf':colorFrom, '--ct':colorTo } as any}
   initial={{ offsetDistance:'0%' }} animate={{ offsetDistance:'100%' }}
   transition={{ repeat:Infinity, ease:'linear', duration, delay:-delay }}/>
 </div>)
}
