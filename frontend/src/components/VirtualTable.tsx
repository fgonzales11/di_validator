import { useRef } from 'react'
import type { ReactNode } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'

export function VirtualTable<T>({
  rows,
  columns,
  onRow,
}: {
  rows: T[]
  columns: { key: string; title: string; render: (r: T) => ReactNode; width?: string }[]
  onRow?: (r: T) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => ref.current,
    estimateSize: () => 44,
    overscan: 8,
  })
  const template = columns.map((c) => c.width || 'minmax(110px,1fr)').join(' ')
  return (
    <div className="table-scroll" ref={ref} role="table" aria-rowcount={rows.length + 1}>
      <div role="row" className="table-header" style={{ gridTemplateColumns: template }}>
        {columns.map((c) => (
          <div role="columnheader" key={c.key}>
            {c.title}
          </div>
        ))}
      </div>
      <div
        style={{ height: virtualizer.getTotalSize(), position: 'relative', minWidth: columns.length * 110 }}
      >
        {virtualizer.getVirtualItems().map((item) => (
          <div
            role="row"
            key={item.key}
            className={'table-row ' + (onRow ? 'clickable' : '')}
            style={{
              gridTemplateColumns: template,
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: 44,
              transform: `translateY(${item.start}px)`,
            }}
          >
            {columns.map((c, i) => (
              <div role="cell" key={c.key}>
                {onRow && i === 0 ? (
                  <button className="text-button" onClick={() => onRow(rows[item.index])}>
                    {c.render(rows[item.index])}
                  </button>
                ) : (
                  c.render(rows[item.index])
                )}
              </div>
            ))}
          </div>
        ))}
      </div>
      {!rows.length ? <div className="table-none">No matching records</div> : null}
    </div>
  )
}
