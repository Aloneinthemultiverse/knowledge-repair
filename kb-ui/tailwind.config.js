/** Matches the TruthGuard site: navy-black ground, Instrument Serif display, teal / purple / coral accents. */
const v = (n) => `rgb(var(--${n}) / <alpha-value>)`
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: v('canvas'), surface: v('surface'), sunk: v('sunk'), ink: v('ink'), mute: v('mute'),
        rule: v('rule'), accent: v('accent'), ok: v('ok'), warn: v('warn'), bad: v('bad'),
        teal: '#39d2c0', purple: '#bc8cff', coral: '#ff8c66', amber: '#e3b341',
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'Consolas', 'monospace'],
        display: ['"Instrument Serif"', 'ui-serif', 'Georgia', 'serif'],
      },
      keyframes: {
        'border-beam': { '100%': { 'offset-distance': '100%' } },
        rise: { from: { opacity: '0', transform: 'translateY(8px)' }, to: { opacity: '1', transform: 'none' } },
      },
      animation: {
        'border-beam': 'border-beam calc(var(--duration)*1s) infinite linear',
        rise: 'rise .3s ease',
      },
    },
  },
}
