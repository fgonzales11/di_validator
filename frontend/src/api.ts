export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    '/api/v1' + path,
    body === undefined
      ? {}
      : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
  )
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(typeof error.detail === 'string' ? error.detail : JSON.stringify(error.detail))
  }
  return response.json()
}
export async function upload(file: File): Promise<{ path: string; name: string }> {
  const form = new FormData()
  form.append('file', file)
  const response = await fetch('/api/v1/uploads', { method: 'POST', body: form })
  const data = await response.json()
  if (!response.ok) throw new Error(data.detail)
  return data
}
