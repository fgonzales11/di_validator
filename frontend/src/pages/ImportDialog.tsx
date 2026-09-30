import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Plus, Upload, X } from 'lucide-react'
import { api, upload } from '../api'
import { ErrorBox, Field } from '../components/ui'
import { Modal } from '../components/Modal'
import type { Channel } from '../types'

const initial = {
  path: '',
  name: '',
  format: 'wide_ami',
  asset_level: 'transformer',
  circuit: '',
  asset_id: 'recording-asset',
  timestamp_column: 'REPORTED_DTTM',
  timezone: 'America/Los_Angeles',
  interval_seconds: 3600,
  interval_position: 'unknown',
  sample_rate: 1000,
  start_time: '2025-01-01T00:00:00Z',
  time_mode: 'timestamp',
  nominal_frequency: 60,
  nominal_voltage: 120,
  channels: [{ name: 'net_energy', kind: 'net_energy', unit: 'kWh', phase: 'A' }] as Channel[],
}
type Config = typeof initial
const units: Record<string, string> = {
  net_energy: 'kWh',
  power: 'kW',
  voltage: 'V',
  current: 'A',
  voltage_rms: 'V',
  current_rms: 'A',
}
export default function ImportDialog({ onClose }: { onClose: () => void }) {
  const [form, setForm] = useState<Config>(initial),
    [error, setError] = useState<unknown>(),
    [busy, setBusy] = useState(false)
  const client = useQueryClient()
  const { data: presets = [] } = useQuery({
    queryKey: ['presets'],
    queryFn: () => api<{ id: string; name: string; kind: string; config: Config }[]>('/presets'),
  })
  function set<K extends keyof Config>(key: K, value: Config[K]) {
    setForm((f) => ({ ...f, [key]: value }))
  }
  async function choose(file?: File) {
    if (!file) return
    setBusy(true)
    try {
      const uploaded = await upload(file)
      setForm((f) => ({ ...f, ...uploaded }))
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }
  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(undefined)
    try {
      await api('/datasets', form)
      void client.invalidateQueries()
      onClose()
    } catch (e) {
      setError(e)
      setBusy(false)
    }
  }
  return (
    <Modal title="Import measurement data" onClose={onClose} wide>
      <form onSubmit={submit}>
        <ErrorBox error={error} />
        <div className="upload-zone">
          <Upload size={25} />
          <strong>Choose a CSV or Parquet file</strong>
          <span>Files are copied to local storage. Original files stay intact.</span>
          <input
            aria-label="Upload measurement file"
            type="file"
            accept=".csv,.parquet"
            onChange={(e) => choose(e.target.files?.[0])}
          />
        </div>
        <div className="form-grid">
          <Field label="Or register a local path">
            <input
              value={form.path}
              required
              onChange={(e) => set('path', e.target.value)}
              placeholder="C:\data\measurements.csv"
            />
          </Field>
          <Field label="Dataset name">
            <input
              value={form.name}
              onChange={(e) => set('name', e.target.value)}
              placeholder="Use filename"
            />
          </Field>
          <Field label="Input layout">
            <select
              value={form.format}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  format: e.target.value,
                  timestamp_column: e.target.value === 'recording' ? 'timestamp' : 'REPORTED_DTTM',
                  channels:
                    e.target.value === 'recording'
                      ? [
                          { name: 'power', kind: 'power', unit: 'kW', phase: 'A' },
                          { name: 'voltage_rms', kind: 'voltage_rms', unit: 'V', phase: 'A' },
                        ]
                      : initial.channels,
                }))
              }
            >
              <option value="wide_ami">Wide interval table · one asset per column</option>
              <option value="recording">Recording · one channel per column</option>
            </select>
          </Field>
          <Field label="Asset level">
            <select value={form.asset_level} onChange={(e) => set('asset_level', e.target.value)}>
              <option value="transformer">Transformer</option>
              <option value="meter">Meter</option>
            </select>
          </Field>
          <Field label="Circuit">
            <input
              value={form.circuit}
              onChange={(e) => set('circuit', e.target.value)}
              placeholder="Infer from filename"
            />
          </Field>
          <Field label="Timestamp column">
            <input value={form.timestamp_column} onChange={(e) => set('timestamp_column', e.target.value)} />
          </Field>
          <Field label="Source timezone">
            <input
              required
              value={form.timezone}
              onChange={(e) => set('timezone', e.target.value)}
              list="timezones"
            />
            <datalist id="timezones">
              <option>America/Los_Angeles</option>
              <option>UTC</option>
            </datalist>
          </Field>
          {form.format === 'wide_ami' ? (
            <>
              <Field label="Interval duration (seconds)">
                <input
                  type="number"
                  min="1"
                  value={form.interval_seconds}
                  onChange={(e) => set('interval_seconds', +e.target.value)}
                />
              </Field>
              <Field
                label="Timestamp position"
                hint="Required before aggregating or aligning with another source."
              >
                <select
                  value={form.interval_position}
                  onChange={(e) => set('interval_position', e.target.value)}
                >
                  <option value="unknown">Unverified · retain source labels</option>
                  <option value="start">Interval start</option>
                  <option value="end">Interval end</option>
                </select>
              </Field>
              <Field label="Energy unit">
                <select
                  value={form.channels[0].unit}
                  onChange={(e) => set('channels', [{ ...form.channels[0], unit: e.target.value }])}
                >
                  <option>kWh</option>
                  <option>Wh</option>
                </select>
              </Field>
            </>
          ) : (
            <>
              <Field label="Asset identifier">
                <input required value={form.asset_id} onChange={(e) => set('asset_id', e.target.value)} />
              </Field>
              <Field label="Sample rate (Hz)">
                <input
                  type="number"
                  min="1"
                  value={form.sample_rate}
                  onChange={(e) => set('sample_rate', +e.target.value)}
                />
              </Field>
              <Field label="Time representation">
                <select value={form.time_mode} onChange={(e) => set('time_mode', e.target.value)}>
                  <option value="timestamp">Absolute timestamps</option>
                  <option value="offset_seconds">Seconds from recording start</option>
                  <option value="sample_index">Sample index</option>
                </select>
              </Field>
              {form.time_mode !== 'timestamp' ? (
                <Field label="Recording start (ISO timestamp)">
                  <input
                    required
                    value={form.start_time}
                    onChange={(e) => set('start_time', e.target.value)}
                  />
                </Field>
              ) : null}
              <Field label="Nominal voltage (V)">
                <input
                  type="number"
                  min="1"
                  value={form.nominal_voltage}
                  onChange={(e) => set('nominal_voltage', +e.target.value)}
                />
              </Field>
              <Field label="Nominal frequency (Hz)">
                <input
                  type="number"
                  min="1"
                  value={form.nominal_frequency}
                  onChange={(e) => set('nominal_frequency', +e.target.value)}
                />
              </Field>
            </>
          )}
        </div>
        {form.format === 'recording' ? (
          <section className="channel-editor">
            <h3>Recording channels</h3>
            {form.channels.map((c, i) => (
              <div className="channel-row" key={i}>
                <Field label="Column name">
                  <input
                    required
                    value={c.name}
                    onChange={(e) =>
                      set(
                        'channels',
                        form.channels.map((ch, j) => (j === i ? { ...ch, name: e.target.value } : ch)),
                      )
                    }
                  />
                </Field>
                <Field label="Measurement">
                  <select
                    value={c.kind}
                    onChange={(e) =>
                      set(
                        'channels',
                        form.channels.map((ch, j) =>
                          j === i ? { ...ch, kind: e.target.value, unit: units[e.target.value] } : ch,
                        ),
                      )
                    }
                  >
                    {Object.keys(units)
                      .filter((k) => k !== 'net_energy')
                      .map((k) => (
                        <option key={k} value={k}>
                          {k.replace('_', ' ')}
                        </option>
                      ))}
                  </select>
                </Field>
                <Field label="Unit">
                  <select
                    value={c.unit}
                    onChange={(e) =>
                      set(
                        'channels',
                        form.channels.map((ch, j) => (j === i ? { ...ch, unit: e.target.value } : ch)),
                      )
                    }
                  >
                    {(c.kind === 'power' ? ['kW', 'W'] : [units[c.kind]]).map((u) => (
                      <option key={u}>{u}</option>
                    ))}
                  </select>
                </Field>
                <Field label="Phase">
                  <input
                    value={c.phase}
                    onChange={(e) =>
                      set(
                        'channels',
                        form.channels.map((ch, j) => (j === i ? { ...ch, phase: e.target.value } : ch)),
                      )
                    }
                  />
                </Field>
                <button
                  type="button"
                  className="icon-button"
                  aria-label="Remove channel"
                  disabled={form.channels.length === 1}
                  onClick={() =>
                    set(
                      'channels',
                      form.channels.filter((_, j) => j !== i),
                    )
                  }
                >
                  <X size={16} />
                </button>
              </div>
            ))}
            <button
              type="button"
              className="button secondary small"
              onClick={() =>
                set('channels', [
                  ...form.channels,
                  { name: 'channel' + (form.channels.length + 1), kind: 'power', unit: 'kW', phase: 'A' },
                ])
              }
            >
              <Plus size={14} />
              Add channel
            </button>
          </section>
        ) : null}
        <div className="notice">
          Invalid or unresolved local timestamps are preserved in the source audit and excluded from analysis.
          Missing measurements remain missing.
        </div>
        <div className="form-actions">
          <select
            aria-label="Load import preset"
            value=""
            onChange={(e) => {
              const p = presets.find((p) => p.id === e.target.value)
              if (p) setForm({ ...p.config, path: form.path })
            }}
          >
            <option value="">Load a saved preset</option>
            {presets
              .filter((p) => p.kind === 'import')
              .map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
          </select>
          <button
            className="button secondary"
            type="button"
            onClick={async () => {
              try {
                await api('/presets', { kind: 'import', name: form.name || 'Import preset', config: form })
                void client.invalidateQueries({ queryKey: ['presets'] })
              } catch (e) {
                setError(e)
              }
            }}
          >
            Save preset
          </button>
          <button className="button" disabled={busy || !form.path} type="submit">
            {busy ? 'Working…' : 'Import dataset'}
          </button>
        </div>
      </form>
    </Modal>
  )
}
