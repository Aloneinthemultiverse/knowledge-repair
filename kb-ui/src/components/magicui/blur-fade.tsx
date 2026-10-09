import { useRef } from 'react'
import { AnimatePresence, motion, useInView, type Variants } from 'motion/react'
export function BlurFade({ children, className, duration=0.5, delay=0, offset=8,
  direction='down', inView=true, inViewMargin='-60px', blur='6px' }: any) {
  const ref = useRef(null)
  const inViewResult = useInView(ref, { once: true, margin: inViewMargin as any })
  const isInView = !inView || inViewResult
  const variants: Variants = {
    hidden: { [direction === 'left' || direction === 'right' ? 'x' : 'y']:
        direction === 'right' || direction === 'down' ? -offset : offset,
      opacity: 0, filter: `blur(${blur})` },
    visible: { [direction === 'left' || direction === 'right' ? 'x' : 'y']: 0,
      opacity: 1, filter: 'blur(0px)' },
  }
  return (<AnimatePresence>
    <motion.div ref={ref} initial="hidden" animate={isInView ? 'visible' : 'hidden'}
      exit="hidden" variants={variants}
      transition={{ delay: 0.04 + delay, duration, ease: 'easeOut' }}
      className={className}>{children}</motion.div>
  </AnimatePresence>)
}
