export const number = (n: number | undefined) =>
  new Intl.NumberFormat('en-US', {
    notation: (n ?? 0) > 999999 ? 'compact' : 'standard',
    maximumFractionDigits: 1,
  }).format(n ?? 0)
export const percent = (n: number | null | undefined) => (n == null ? '—' : `${(n * 100).toFixed(1)}%`)
export const date = (v: string | undefined) =>
  v ? new Date(v).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : '—'
