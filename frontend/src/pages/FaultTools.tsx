import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowRight, Download } from 'lucide-react'
import { api } from '../api'
import { Badge, ErrorBox, Field, Panel } from '../components/ui'
import { Modal } from '../components/Modal'
import { VirtualTable } from '../components/VirtualTable'
import type { Dataset, Experiment, Job, NetworkLocation } from '../types'
import { faultRecordName } from '../faultReference'

export type FaultParameters = {
  channel_map: Record<string, string>
  start: number
  end: number | null
  manual_onset: number | null
  eta_i: number
  eta_u: number
  pre_seconds: number
  post_seconds: number
  fault_loop: string
  line_length_km: number
  nominal_voltage_kv: number
  base_power_mva: number
  r1_ohm_km: number
  x1_ohm_km: number
  r0_ohm_km: number | null
  x0_ohm_km: number | null
  line_parameters_verified: boolean
  settle_cycles: number
  average_cycles: number
  min_current_ka: number
  known_distance_km: number | null
}
export const defaultFaultParameters: FaultParameters = {
  channel_map: {},
  start: 0,
  end: null,
  manual_onset: null,
  eta_i: 0.5,
  eta_u: 0.85,
  pre_seconds: 0.05,
  post_seconds: 0.15,
  fault_loop: 'AUTO',
  line_length_km: 50,
  nominal_voltage_kv: 110,
  base_power_mva: 100,
  r1_ohm_km: 0.1,
  x1_ohm_km: 0.4,
  r0_ohm_km: null,
  x0_ohm_km: null,
  line_parameters_verified: false,
  settle_cycles: 1,
  average_cycles: 3,
  min_current_ka: 0.001,
  known_distance_km: null,
}
const phases = ['IA', 'IB', 'IC', 'UA', 'UB', 'UC']
export function FaultSettings({
  dataset,
  value,
  onChange,
}: {
  dataset: Dataset
  value: FaultParameters
  onChange: (p: FaultParameters) => void
}) {
  function set<K extends keyof FaultParameters>(key: K, next: FaultParameters[K]) {
    onChange({ ...value, [key]: next })
  }
  const mapping = {
    ...Object.fromEntries(
      phases.map((p) => [
        p,
        dataset.channels.find((c) => c.kind === (p[0] === 'I' ? 'current' : 'voltage') && c.phase === p[1])
          ?.name || '',
      ]),
    ),
    ...value.channel_map,
  }
  return (
    <div className="fault-settings">
      <p className="notice amber-notice">
        Offline waveform analysis. Line values below are notebook examples until verified. Distances are
        apparent estimates from the recording terminal.
      </p>
      <Field label="Fault loop">
        <select value={value.fault_loop} onChange={(e) => set('fault_loop', e.target.value)}>
          {['AUTO', 'AG', 'BG', 'CG', 'AB', 'BC', 'CA', 'POS'].map((v) => (
            <option key={v} value={v}>
              {v === 'AUTO' ? 'AUTO · heuristic phase selection' : v}
            </option>
          ))}
        </select>
      </Field>
      <div className="form-grid">
        <Field label="Line length (km)">
          <input
            type="number"
            min=".001"
            step="any"
            value={value.line_length_km}
            onChange={(e) => set('line_length_km', +e.target.value)}
          />
        </Field>
        <Field label="Nominal line voltage (kV)">
          <input
            type="number"
            min=".001"
            step="any"
            value={value.nominal_voltage_kv}
            onChange={(e) => set('nominal_voltage_kv', +e.target.value)}
          />
        </Field>
        <Field label="R1 (Ω/km)">
          <input
            type="number"
            min="0"
            step="any"
            value={value.r1_ohm_km}
            onChange={(e) => set('r1_ohm_km', +e.target.value)}
          />
        </Field>
        <Field label="X1 (Ω/km)">
          <input
            type="number"
            min=".000001"
            step="any"
            value={value.x1_ohm_km}
            onChange={(e) => set('x1_ohm_km', +e.target.value)}
          />
        </Field>
        <Field label="R0 (Ω/km)" hint="Optional; supply both R0 and X0.">
          <input
            type="number"
            min="0"
            step="any"
            value={value.r0_ohm_km ?? ''}
            onChange={(e) => set('r0_ohm_km', e.target.value === '' ? null : +e.target.value)}
          />
        </Field>
        <Field label="X0 (Ω/km)" hint="For ground-current compensation.">
          <input
            type="number"
            min=".000001"
            step="any"
            value={value.x0_ohm_km ?? ''}
            onChange={(e) => set('x0_ohm_km', e.target.value === '' ? null : +e.target.value)}
          />
        </Field>
      </div>
      <label className="fault-check">
        <input
          type="checkbox"
          checked={value.line_parameters_verified}
          onChange={(e) => set('line_parameters_verified', e.target.checked)}
        />
        Line parameters verified for this recording
      </label>
      <details>
        <summary>Channels, detection and analysis window</summary>
        <p className="muted">
          Select synchronized phase waveforms with consistent current direction. RMS channels cannot be used.
        </p>
        <div className="form-grid">
          {phases.map((p) => (
            <Field key={p} label={p + ' waveform'}>
              <select
                value={mapping[p]}
                onChange={(e) => set('channel_map', { ...mapping, [p]: e.target.value })}
              >
                <option value="">Select channel</option>
                {dataset.channels
                  .filter((c) => c.kind === (p[0] === 'I' ? 'current' : 'voltage'))
                  .map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name} · {c.unit}
                    </option>
                  ))}
              </select>
            </Field>
          ))}
        </div>
        <div className="form-grid">
          <Field label="Analysis start (s)">
            <input
              type="number"
              min="0"
              step="any"
              value={value.start}
              onChange={(e) => set('start', +e.target.value)}
            />
          </Field>
          <Field label="Analysis end (s)" hint="Blank: end of recording; 200,000 sample limit.">
            <input
              type="number"
              min="0"
              step="any"
              value={value.end ?? ''}
              onChange={(e) => set('end', e.target.value === '' ? null : +e.target.value)}
            />
          </Field>
          <Field label="Current rise fraction">
            <input
              type="number"
              min=".001"
              step=".05"
              value={value.eta_i}
              onChange={(e) => set('eta_i', +e.target.value)}
            />
          </Field>
          <Field label="Fault voltage ratio">
            <input
              type="number"
              min=".001"
              max=".999"
              step=".01"
              value={value.eta_u}
              onChange={(e) => set('eta_u', +e.target.value)}
            />
          </Field>
          <Field label="Pre-fault window (s)">
            <input
              type="number"
              min=".001"
              max="2"
              step=".005"
              value={value.pre_seconds}
              onChange={(e) => set('pre_seconds', +e.target.value)}
            />
          </Field>
          <Field label="Post-fault window (s)">
            <input
              type="number"
              min=".001"
              max="2"
              step=".005"
              value={value.post_seconds}
              onChange={(e) => set('post_seconds', +e.target.value)}
            />
          </Field>
          <Field label="Settling cycles">
            <input
              type="number"
              min="0"
              max="20"
              step=".5"
              value={value.settle_cycles}
              onChange={(e) => set('settle_cycles', +e.target.value)}
            />
          </Field>
          <Field label="Averaging cycles">
            <input
              type="number"
              min="1"
              max="50"
              value={value.average_cycles}
              onChange={(e) => set('average_cycles', +e.target.value)}
            />
          </Field>
          <Field label="Base power (MVA)">
            <input
              type="number"
              min=".001"
              step="any"
              value={value.base_power_mva}
              onChange={(e) => set('base_power_mva', +e.target.value)}
            />
          </Field>
          <Field label="Minimum loop current (kA)">
            <input
              type="number"
              min=".000001"
              step=".001"
              value={value.min_current_ka}
              onChange={(e) => set('min_current_ka', +e.target.value)}
            />
          </Field>
        </div>
        <Field
          label="Manual fault onset (s)"
          hint="Blank: detect automatically. Manual onset disables detection metrics."
        >
          <input
            type="number"
            min="0"
            step=".0005"
            value={value.manual_onset ?? ''}
            onChange={(e) => set('manual_onset', e.target.value === '' ? null : +e.target.value)}
          />
        </Field>
        <Field
          label="Known fault distance (km)"
          hint="Optional independent ground truth; never inferred from filenames."
        >
          <input
            type="number"
            min="0"
            step="any"
            value={value.known_distance_km ?? ''}
            onChange={(e) => set('known_distance_km', e.target.value === '' ? null : +e.target.value)}
          />
        </Field>
      </details>
    </div>
  )
}

type Source = { id: string; name: string; bytes: number; dataset_id: string | null }
export function ComtradeDialog({ onClose, onOpen }: { onClose: () => void; onOpen: (id: string) => void }) {
  const client = useQueryClient(),
    [chosen, setChosen] = useState<string[]>([]),
    [jobId, setJobId] = useState(''),
    [error, setError] = useState<unknown>(),
    [busy, setBusy] = useState(false)
  const catalog = useQuery({
    queryKey: ['comtrade-sources'],
    queryFn: () => api<Source[]>('/faults/sources'),
    refetchInterval: 3000,
  })
  const job = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => api<Job>('/jobs/' + jobId),
    enabled: !!jobId,
    refetchInterval: 1500,
  })
  const sources = catalog.data || []
  return (
    <Modal title="COMTRADE recordings" onClose={onClose} wide>
      <p>
        Recordings from <code>fault-distance/data/data_test</code>. Imports retain the original CFG/DAT pair,
        native sample timing and calibration metadata.
      </p>
      <p className="notice amber-notice">
        RTDS simulation records. Recording timezone and line parameters are unverified. Filenames and trigger
        times are not fault labels.
      </p>
      <ErrorBox
        error={
          error ||
          catalog.error ||
          job.error ||
          (job.data?.status === 'failed' ? job.data.message : undefined)
        }
      />
      <div className="form-actions">
        <button className="button secondary" onClick={() => setChosen(sources.map((s) => s.id))}>
          Select all {sources.length}
        </button>
        <button className="button secondary" onClick={() => setChosen([])}>
          Clear selection
        </button>
      </div>
      <VirtualTable
        rows={sources}
        columns={[
          {
            key: 'choose',
            title: 'Import',
            width: '70px',
            render: (s) => (
              <input
                type="checkbox"
                aria-label={'Import ' + s.name}
                checked={chosen.includes(s.id)}
                onChange={(e) =>
                  setChosen((previous) =>
                    e.target.checked ? [...previous, s.id] : previous.filter((id) => id !== s.id),
                  )
                }
              />
            ),
          },
          { key: 'name', title: 'Recording', render: (s) => s.name },
          {
            key: 'status',
            title: 'Status',
            render: (s) =>
              s.dataset_id ? (
                <button
                  className="text-button"
                  onClick={() => {
                    onOpen(s.dataset_id!)
                    onClose()
                  }}
                >
                  Open in Event Lab
                </button>
              ) : (
                <Badge>Available</Badge>
              ),
          },
          {
            key: 'source',
            title: 'Original files',
            render: (s) => (
              <span className="inline-actions">
                <a href={`/api/v1/faults/sources/${s.id}/cfg`} aria-label={'Download ' + s.name + ' CFG'}>
                  CFG
                </a>
                <a href={`/api/v1/faults/sources/${s.id}/dat`} aria-label={'Download ' + s.name + ' DAT'}>
                  DAT
                </a>
              </span>
            ),
          },
        ]}
      />
      {job.data ? (
        <div role="status">
          <p>{job.data.message}</p>
          <progress value={job.data.progress} max="1" />
          {job.data.status === 'completed' ? (
            <p className="success">Imported recordings are ready to open above.</p>
          ) : null}
        </div>
      ) : null}
      <div className="form-actions">
        <button
          className="button"
          disabled={!chosen.length || busy || ['queued', 'running'].includes(job.data?.status || '')}
          onClick={async () => {
            setBusy(true)
            setError(undefined)
            try {
              const queued = await api<Job>('/faults/imports', { source_ids: chosen })
              setJobId(queued.id)
              void client.invalidateQueries({ queryKey: ['datasets'] })
            } catch (e) {
              setError(e)
            } finally {
              setBusy(false)
            }
          }}
        >
          Import {chosen.length} selected
          <ArrowRight size={15} />
        </button>
      </div>
    </Modal>
  )
}

type WavewinSource = { id: string; name: string; station: string; bytes: number; dataset_id: string | null }
export function WavewinDialog({ onClose, onOpen }: { onClose: () => void; onOpen: (id: string) => void }) {
  const client = useQueryClient(),
    [chosen, setChosen] = useState<string[]>([]),
    [jobId, setJobId] = useState(''),
    [error, setError] = useState<unknown>(),
    [busy, setBusy] = useState(false),
    [stationFilter, setStationFilter] = useState('')
  const catalog = useQuery({
    queryKey: ['wavewin-sources'],
    queryFn: () => api<WavewinSource[]>('/faults/wavewin/sources'),
    refetchInterval: 3000,
  })
  const jobs = useQuery({
    queryKey: ['jobs'],
    queryFn: () => api<Job[]>('/jobs'),
    refetchInterval: 1500,
  })
  // Reopening the dialog must still show an import that is already running.
  const job =
    jobs.data?.find((j) => j.id === jobId) ||
    jobs.data?.find((j) => j.kind === 'wavewin_import' && ['queued', 'running'].includes(j.status))
  const running = ['queued', 'running'].includes(job?.status || '')
  const all = catalog.data || []
  const imported = all.filter((s) => s.dataset_id).length
  const stations = [...new Set(all.map((s) => s.station))].sort()
  const sources = stationFilter ? all.filter((s) => s.station === stationFilter) : all
  const pending = sources.filter((s) => !s.dataset_id)
  return (
    <Modal title="Wavewin event logs" onClose={onClose} wide>
      <p>
        Relay-captured events from <code>fault-distance/data/Wavewin_Logs</code>. Imports parse the SEL
        Compressed ASCII Event (.CEV) report and retain the original file.
      </p>
      <p className="notice amber-notice">
        Values are relay-reported primary amps and kV (SEL-451 reports flag PRIM_VAL=YES; SEL-351S voltages
        match the 13.2 kV feeder level), sampled as fundamental-filtered RMS values at 4 samples per cycle.
        Recorder clock timezone is unverified.
      </p>
      <ErrorBox
        error={error || catalog.error || jobs.error || (job?.status === 'failed' ? job.message : undefined)}
      />
      <div role="status">
        <p>
          <strong>
            {imported} of {all.length}
          </strong>{' '}
          events imported.
          {busy ? ' Queuing import…' : running ? ' ' + (job?.message || 'Import queued…') : ''}
        </p>
        {busy || running ? <progress value={busy ? undefined : job?.progress} max="1" /> : null}
        {job?.status === 'completed' && job.id === jobId ? (
          <p className="success">Import complete. Imported events can be opened below.</p>
        ) : null}
      </div>
      <div className="form-actions">
        <Field label="Station">
          <select value={stationFilter} onChange={(e) => setStationFilter(e.target.value)}>
            <option value="">All stations ({all.length})</option>
            {stations.map((s) => (
              <option key={s} value={s}>
                {s} ({all.filter((x) => x.station === s).length})
              </option>
            ))}
          </select>
        </Field>
        <button className="button secondary" onClick={() => setChosen(pending.map((s) => s.id))}>
          Select {pending.length} not imported
        </button>
        <button className="button secondary" onClick={() => setChosen([])}>
          Clear selection
        </button>
        <button
          className="button"
          disabled={!chosen.length || busy || running}
          onClick={async () => {
            setBusy(true)
            setError(undefined)
            try {
              const queued = await api<Job>('/faults/wavewin/imports', { source_ids: chosen })
              setJobId(queued.id)
              setChosen([])
              void client.invalidateQueries({ queryKey: ['jobs'] })
              void client.invalidateQueries({ queryKey: ['datasets'] })
            } catch (e) {
              setError(e)
            } finally {
              setBusy(false)
            }
          }}
        >
          {busy ? 'Queuing…' : running ? 'Import running…' : `Import ${chosen.length} selected`}
          <ArrowRight size={15} />
        </button>
      </div>
      <VirtualTable
        rows={sources}
        columns={[
          {
            key: 'choose',
            title: 'Import',
            width: '70px',
            render: (s) => (
              <input
                type="checkbox"
                aria-label={'Import ' + s.name}
                checked={chosen.includes(s.id)}
                disabled={!!s.dataset_id}
                onChange={(e) =>
                  setChosen((previous) =>
                    e.target.checked ? [...previous, s.id] : previous.filter((id) => id !== s.id),
                  )
                }
              />
            ),
          },
          { key: 'station', title: 'Station', width: '140px', render: (s) => s.station },
          { key: 'name', title: 'Event', render: (s) => s.name },
          {
            key: 'status',
            title: 'Status',
            render: (s) =>
              s.dataset_id ? (
                <button
                  className="text-button"
                  onClick={() => {
                    onOpen(s.dataset_id!)
                    onClose()
                  }}
                >
                  Open in Event Lab
                </button>
              ) : (
                <Badge>Available</Badge>
              ),
          },
          {
            key: 'source',
            title: 'Original file',
            render: (s) => (
              <a
                href={`/api/v1/faults/wavewin/sources/${s.id}/cev`}
                aria-label={'Download ' + s.name + ' CEV'}
              >
                CEV
              </a>
            ),
          },
        ]}
      />
    </Modal>
  )
}

const KM_PER_MILE = 1.609344
function miles(km: number | undefined, digits = 3) {
  return km === undefined ? '—' : `${(km / KM_PER_MILE).toFixed(digits)} mi (${km.toFixed(digits)} km)`
}

function RelayLocation({ location: n }: { location: NetworkLocation }) {
  return n.relay_location_mi !== undefined ? (
    <p className="muted">
      Relay-reported location: {n.relay_location_mi.toFixed(2)} mi ({n.relay_location_km?.toFixed(2)} km).
      Read as miles from the relay's line-length setting; this is the relay's own estimate, not verified.
    </p>
  ) : null
}

function NetworkLocationView({ location: n }: { location: NetworkLocation }) {
  if (n.status !== 'located') {
    return (
      <>
        <p className="notice">
          Network model location unavailable{n.feeder ? ` (feeder ${n.feeder})` : ''}: {n.reason}
        </p>
        <RelayLocation location={n} />
      </>
    )
  }
  const best = n.candidates?.[0]
  return (
    <div className="fault-network">
      <div className="fault-distance-value">
        <span>Network-model location · feeder {n.feeder}</span>
        <strong>
          {best?.distance_mi.toFixed(3)} <small>mi along feeder</small>
        </strong>
        <span>{best?.distance_km.toFixed(3)} km</span>
        <span>
          Loop {n.loop} ({n.loop_source}) · measured {n.measured_resistance_ohm?.toFixed(3)} + j
          {n.measured_reactance_ohm?.toFixed(3)} Ω · {n.candidate_count} matching location
          {n.candidate_count === 1 ? '' : 's'}
        </span>
      </div>
      <RelayLocation location={n} />
      <p className="notice amber-notice">
        {n.estimated_impedance_share === 1
          ? 'All line impedances in this model are estimated. '
          : n.estimated_impedance_share
            ? `${Math.round(n.estimated_impedance_share * 100)}% of line impedances in this model are estimated. `
            : ''}
        Candidates are every point where the modeled cumulative reactance from the substation equals the
        measured reactance; they are ordered by the implied fault resistance.
      </p>
      <VirtualTable
        rows={n.candidates || []}
        columns={[
          { key: 'distance', title: 'Distance', render: (c) => miles(c.distance_km) },
          { key: 'section', title: 'Section', render: (c) => c.section_id },
          { key: 'phases', title: 'Phases', width: '80px', render: (c) => c.phases },
          { key: 'rf', title: 'Fault resistance', render: (c) => c.fault_resistance_ohm.toFixed(3) + ' Ω' },
          {
            key: 'position',
            title: 'Latitude, longitude',
            render: (c) =>
              c.latitude !== null && c.longitude !== null
                ? `${c.latitude.toFixed(5)}, ${c.longitude.toFixed(5)}`
                : '—',
          },
        ]}
      />
      <details>
        <summary>Model and assumptions</summary>
        <p className="muted">
          {n.model} · {n.nominal_kv} kV · {n.measurement} · modeled {n.loop} reach{' '}
          {n.modeled_reach_ohm?.toFixed(3)} Ω · {n.modeled_mi?.toFixed(2)} mi of modeled conductor route
        </p>
        <ul>
          {n.assumptions?.map((text) => (
            <li key={text}>{text}</li>
          ))}
        </ul>
      </details>
    </div>
  )
}

export function FaultResults({ result, dataset }: { result: Experiment; dataset: Dataset }) {
  const analyses = result.fault_analyses
  if (!analyses) return null
  return (
    <Panel
      title="Fault distance analysis"
      subtitle="Single-ended reactance · distance from measurement terminal"
    >
      <div className="panel-pad">
        <p className="fault-source">
          Recording: <strong>{faultRecordName(dataset)}</strong>
        </p>
        <p className="muted">
          {result.interpretation} Centered filtering uses future samples; decision times show full-segment
          availability. Real-time latency is not reported.
        </p>
        {analyses.map((a) => (
          <article className="fault-result" key={a.segment}>
            <div className="inline-actions">
              <Badge tone={a.status === 'analyzed' ? 'green' : 'amber'}>Segment {a.segment + 1}</Badge>
              <span>
                {a.start.toFixed(4)}–{a.end.toFixed(4)} s
              </span>
            </div>
            {a.status !== 'analyzed' ? (
              <>
                <p>{a.reason}</p>
                {a.network_location ? <NetworkLocationView location={a.network_location} /> : null}
              </>
            ) : (
              <>
                <p>
                  Onset <strong>{a.onset?.toFixed(4)} s</strong> ({a.inception_source}) · fault type{' '}
                  <strong>{a.fault_type_heuristic}</strong> (heuristic)
                </p>
                {a.network_location ? <NetworkLocationView location={a.network_location} /> : null}
                {a.network_location ? <h4>Uniform-line estimate (settings above)</h4> : null}
                {a.distance?.status === 'estimated' ? (
                  <>
                    <div className="fault-distance-value">
                      <span>
                        {a.distance.uncompensated_ground_estimate
                          ? 'Uncompensated apparent distance'
                          : 'Apparent fault distance'}
                      </span>
                      <strong>
                        {a.distance.estimated_distance_km !== undefined
                          ? (a.distance.estimated_distance_km / KM_PER_MILE).toFixed(4)
                          : '—'}{' '}
                        <small>mi</small>
                      </strong>
                      <span>{a.distance.estimated_distance_km?.toFixed(4)} km</span>
                      <span>
                        Loop {a.distance.fault_loop} · {a.distance.distance_percent_of_line?.toFixed(2)}% of
                        line
                      </span>
                    </div>
                    {a.distance.uncompensated_ground_estimate ? (
                      <p className="notice amber-notice">
                        R0/X0 were not supplied; ground-current compensation was omitted.
                      </p>
                    ) : null}
                    {!a.distance.within_line ? (
                      <p className="notice amber-notice">
                        Estimate is outside the configured line. The value has not been clipped.
                      </p>
                    ) : null}
                    <p className="muted">
                      {a.distance.cycles_used} measured cycles · {miles(a.distance.cycle_min_km, 4)} to{' '}
                      {miles(a.distance.cycle_max_km, 4)}. This spread is cycle variation, not a confidence
                      interval.
                    </p>
                    {a.distance.absolute_error_km !== undefined ? (
                      <p>
                        Absolute error against supplied ground truth: {miles(a.distance.absolute_error_km, 4)}
                      </p>
                    ) : null}
                    <VirtualTable
                      rows={a.distance.cycles || []}
                      columns={[
                        { key: 'cycle', title: 'Cycle', render: (c) => c.cycle },
                        { key: 'time', title: 'After onset', render: (c) => c.start_after_fault_ms + ' ms' },
                        { key: 'z', title: 'Reactance', render: (c) => c.x_apparent_ohm.toFixed(4) + ' Ω' },
                        {
                          key: 'distance',
                          title: 'Distance',
                          render: (c) => miles(c.distance_km, 4),
                        },
                      ]}
                    />
                  </>
                ) : (
                  <p className="notice">
                    {a.distance?.apparent_distance_km !== undefined
                      ? `Distance withheld (${a.distance.status.replace(/_/g, ' ')}): ${a.distance.reason} ` +
                        `Apparent ${miles(a.distance.apparent_distance_km, 4)}.`
                      : `Distance unavailable: ${a.distance?.reason}`}
                  </p>
                )}
                <p className="muted">
                  Tensor: {a.tensor_shape?.join(' × ')} · {a.padding_left_samples} left and{' '}
                  {a.padding_right_samples} right padded samples. Padding is excluded from distance cycles.
                </p>
              </>
            )}
          </article>
        ))}
        <a className="button secondary" href={`/api/v1/experiments/${result.id}/export`}>
          <Download size={16} />
          Export fault results and tensors
        </a>
      </div>
    </Panel>
  )
}
