import { useEffect, useRef, useState } from 'react'
import { Maximize2 } from 'lucide-react'
import type { Data, Layout } from 'plotly.js'
import { ErrorBox } from './ui'

const EMPTY_LAYOUT: Partial<Layout> = {}

export function Chart({
  data,
  layout = EMPTY_LAYOUT,
  height = 360,
  onRange,
  onCursor,
  cursorGroup,
  title = 'Chart',
}: {
  data: Data[]
  layout?: Partial<Layout>
  height?: number
  onRange?: (start: number, end: number) => void
  onCursor?: (position: number | null) => void
  cursorGroup?: string
  title?: string
}) {
  const ref = useRef<HTMLDivElement>(null),
    cursorLine = useRef<HTMLDivElement>(null),
    range = useRef(onRange),
    cursor = useRef(onCursor)
  const [expanded, setExpanded] = useState(false),
    [error, setError] = useState<unknown>()
  range.current = onRange
  cursor.current = onCursor
  const options = useRef({ data, layout, height, expanded, cursorGroup })
  const requestRender = useRef<() => void>(() => {})
  useEffect(() => {
    const container = ref.current
    if (!container) return
    // Each mount owns its plot node, including during StrictMode's effect replay.
    const node = document.createElement('div')
    container.appendChild(node)
    let alive = true
    let running = false
    let dirty = false
    let Plotly: typeof import('plotly.js-dist-min').default | undefined
    let cleanup: (() => void) | undefined
    const render = async () => {
      if (running || !alive) return
      running = true
      try {
        Plotly ??= (await import('plotly.js-dist-min')).default
        while (alive && dirty) {
          dirty = false
          const { data, layout, height, expanded } = options.current
          const chart = await Plotly.react(
            node,
            data,
            {
              autosize: true,
              height: expanded ? window.innerHeight * 0.78 : height,
              paper_bgcolor: 'transparent',
              plot_bgcolor: 'transparent',
              margin: { l: 58, r: 20, t: 20, b: 50 },
              font: { family: 'Segoe UI, system-ui, sans-serif', size: 11, color: '#65746e' },
              colorway: ['#087d71', '#dd8d37', '#6279ba', '#bd6380', '#60a89b'],
              xaxis: { gridcolor: '#edf1ee', zeroline: false },
              yaxis: { gridcolor: '#edf1ee', zeroline: false },
              showlegend: data.length > 1,
              legend: { orientation: 'h', y: 1.1 },
              ...layout,
            },
            {
              responsive: true,
              displaylogo: false,
              toImageButtonOptions: { format: 'png', filename: 'di-validator-chart', scale: 2 },
              modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            },
          )
          if (!alive) break
          setError(undefined)
          if (cleanup) continue
          chart.on('plotly_relayout', (event) => {
            const a = event['xaxis.range[0]'],
              b = event['xaxis.range[1]']
            if (typeof a === 'number' && typeof b === 'number') range.current?.(a, b)
          })
          chart.on('plotly_hover', (event) => {
            const x = event.points[0]?.x
            const { cursorGroup } = options.current
            if (typeof x === 'number') cursor.current?.(x)
            if (cursorGroup && typeof x === 'number')
              window.dispatchEvent(new CustomEvent('di-chart-cursor', { detail: { group: cursorGroup, x } }))
          })
          chart.on('plotly_unhover', () => {
            const { cursorGroup } = options.current
            cursor.current?.(null)
            if (cursorGroup)
              window.dispatchEvent(
                new CustomEvent('di-chart-cursor', { detail: { group: cursorGroup, x: null } }),
              )
          })
          const synchronize = (event: Event) => {
            const { group, x } = (event as CustomEvent<{ group: string; x: number | null }>).detail
            if (group !== options.current.cursorGroup || !cursorLine.current) return
            const axes = (
              node as unknown as {
                _fullLayout?: {
                  xaxis: { l2p: (x: number) => number; _offset: number; _length: number }
                  yaxis: { _offset: number; _length: number }
                }
              }
            )._fullLayout
            if (!axes || x == null) {
              cursorLine.current.style.display = 'none'
              return
            }
            const pixel = axes.xaxis.l2p(x)
            Object.assign(cursorLine.current.style, {
              display: pixel >= 0 && pixel <= axes.xaxis._length ? 'block' : 'none',
              left: `${axes.xaxis._offset + pixel}px`,
              top: `${axes.yaxis._offset}px`,
              height: `${axes.yaxis._length}px`,
            })
          }
          window.addEventListener('di-chart-cursor', synchronize)
          const observer = new ResizeObserver(() => {
            if (alive && !running && node.isConnected) void Plotly?.Plots.resize(node)
          })
          observer.observe(node)
          cleanup = () => {
            observer.disconnect()
            window.removeEventListener('di-chart-cursor', synchronize)
            chart.removeAllListeners('plotly_relayout')
            chart.removeAllListeners('plotly_hover')
            chart.removeAllListeners('plotly_unhover')
          }
        }
      } catch (cause) {
        if (alive) setError(cause)
      } finally {
        running = false
        if (!alive) Plotly?.purge(node)
      }
    }
    requestRender.current = () => {
      // Coalesce polling updates to one latest snapshot; never overlap Plotly.react.
      dirty = true
      void render()
    }
    return () => {
      alive = false
      requestRender.current = () => {}
      cleanup?.()
      node.remove()
      if (!running) Plotly?.purge(node)
    }
  }, [])
  useEffect(() => {
    options.current = { data, layout, height, expanded, cursorGroup }
    requestRender.current()
  }, [data, layout, height, expanded, cursorGroup])
  return (
    <div className={expanded ? 'chart-shell expanded' : 'chart-shell'}>
      <button
        className="chart-expand icon-button"
        onClick={() => setExpanded(!expanded)}
        aria-label={expanded ? 'Exit full screen' : 'Expand chart'}
      >
        <Maximize2 size={15} />
      </button>
      <ErrorBox error={error} />
      <div ref={ref} role="img" aria-label={title} style={{ minHeight: height }} />
      <div
        ref={cursorLine}
        aria-hidden="true"
        style={{
          position: 'absolute',
          pointerEvents: 'none',
          borderLeft: '1px solid #73827c',
          display: 'none',
          zIndex: 2,
        }}
      />
    </div>
  )
}
