import { useState } from 'react'

export function useStored<T>(key: string, initial: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => {
    try {
      return JSON.parse(localStorage.getItem('di.v1.' + key) || 'null') ?? initial
    } catch {
      return initial
    }
  })
  return [
    value,
    (next: T) => {
      setValue(next)
      localStorage.setItem('di.v1.' + key, JSON.stringify(next))
    },
  ]
}
