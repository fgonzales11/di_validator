import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowRight, Cpu, Download, FlaskConical, Pause, Play, Square } from 'lucide-react'
import type { Data, Layout } from 'plotly.js'
import { api } from '../api'
import { Chart } from '../components/Chart'
import { Badge, Empty, ErrorBox, Field, PageTitle, Panel } from '../components/ui'
import './meter-lab.css'

type Agent = 'pv' | 'fault'
type State =
  'preparing' | 'running' | 'paused' | 'draining' | 'completed' | 'cancelled' | 'failed' | 'interrupted'
type Json = Record<string, any>
type Descriptor = {
  id: Agent
  name: string
  forms: string[]
  configuration: Record<string, string>
  input: string
  timing: string
}
type Capabilities = {
  available: boolean
  reason?: string
  agents: Descriptor[]
  worker_alive: boolean
  metrology_available?: boolean
  seed_fixtures?: { id: string; name: string; sha256: string }[]
}
type CatalogEntry = {
  id: string
  agent: Agent
  name: string
  source: 'preset' | 'dataset' | 'upload'
  synthetic?: boolean
  description?: string
  warning?: string
  configuration?: Record<string, string>
}
type Telemetry = {
  state: State
  transmitted?: number
  received?: number
  processed?: number
  total_samples?: number
  queue?: number
  pending?: number
  rejected?: number
  stored_data?: number
  stored_events?: number
  sdk_attempts?: number
  diagnostics?: number
  scenario_time?: number
  actual_speed?: number
  last_activity?: number
  error?: string
  resources?: Json
  detection?: Json
  elapsed_seconds?: number
  persistent_bytes?: number
  stored_bytes?: number
  routine_data_bytes?: number
}
type Check = { category: string; name: string; verdict: string; detail: string }
type Run = {
  id: string
  name: string
  agent: Agent
  state: State
  created_at: string
  manifest: Json
  telemetry: Telemetry
  checks: { verdict: string; checks: Check[] }
  checkpoint_available: boolean
}
type Series = {
  voltage?: { channels: string[]; rows: (number | null)[][] } | null
  inputs: { channels: string[]; rows: (number | null)[][] }
  diagnostics: Json[]
  progress_time: number | null
  output_available: number
}
const terminal = (state?: string) => ['completed', 'cancelled', 'failed', 'interrupted'].includes(state || '')
const num = (n?: number) => (n ?? 0).toLocaleString(undefined, { maximumFractionDigits: 2 })
const KM_PER_MILE = 1.609344
// Miles first (relay LOCATION units), km alongside.
const miles = (km?: number | null) =>
  km == null ? '—' : `${(km / KM_PER_MILE).toFixed(3)} mi (${km.toFixed(3)} km)`
const withheld: Record<string, string> = {
  unavailable: 'Unavailable',
  out_of_range: 'Withheld · beyond line',
  behind_relay: 'Withheld · behind relay',
  inconsistent: 'Withheld · inconsistent cycles',
}
const get = <T,>(path: string) => api<T>('/meter-lab' + path)
const post = <T,>(path: string, body: unknown) => api<T>('/meter-lab' + path, body)
const phaseColors = ['#087d71', '#d48a35', '#637bb6']

// Host-side candidates on the recorded feeder's network model; the ARM agent reports one
// equivalent-line distance, while a branched feeder can cross the measured reactance several times.
function NetworkPlacement({ feeder, rows }: { feeder: Json; rows: Json[] }) {
  const located = rows.filter((r) => r.network_location)
  const relay = Number(feeder.wavewin?.location)
  return (
    <Panel
      title="Feeder network placement"
      subtitle={`Host-side, not ARM output · ${feeder.model || 'no network model'} · loop from ${feeder.fault_loop_source}`}
    >
      {feeder.warning && <p className="ml-hint">{feeder.warning}</p>}
      <p className="ml-hint">
        Relay LOCATION:{' '}
        {Number.isFinite(relay) ? `${relay.toFixed(2)} mi (${(relay * KM_PER_MILE).toFixed(2)} km)` : '—'} ·
        relay event {feeder.wavewin?.event_type || '—'}. The relay and the agent are both single-ended
        estimates; their difference shows agreement, not accuracy.
      </p>
      {located.length ? (
        <div className="ml-table-scroll">
          <table>
            <thead>
              <tr>
                {['Segment', 'Candidate', 'Distance', 'Section', 'Fault R (Ω)'].map((x) => (
                  <th key={x}>{x}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {located.flatMap((r) => {
                const n = r.network_location
                if (n.status !== 'located')
                  return [
                    <tr key={r.segment}>
                      <td>{r.segment}</td>
                      <td colSpan={4}>Not located · {n.reason}</td>
                    </tr>,
                  ]
                return n.candidates.slice(0, 3).map((c: Json, i: number) => (
                  <tr key={`${r.segment}-${i}`}>
                    <td>{i ? '' : r.segment}</td>
                    <td>
                      {i + 1} of {n.candidate_count}
                    </td>
                    <td>{miles(c.distance_km)}</td>
                    <td>{c.section_id}</td>
                    <td>{c.fault_resistance_ohm.toFixed(3)}</td>
                  </tr>
                ))
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty title="No placement yet">Candidates appear once analysis results are available.</Empty>
      )}
    </Panel>
  )
}

function Flow({
  run,
  selected,
  onSelect,
}: {
  run?: Run
  selected: number
  onSelect: (index: number) => void
}) {
  const t = run?.telemetry || ({} as Telemetry),
    total = t.total_samples || 0
  const count = [
    t.transmitted,
    t.received,
    t.processed,
    t.sdk_attempts,
    (t.stored_data || 0) + (t.stored_events || 0),
    t.diagnostics,
  ]
  const labels = [
    'Scenario',
    'Replay transport',
    'HW 4.2 ARM agent',
    'DataServer',
    'Stored outcomes',
    'Analysis diagnostics',
  ]
  if (run?.manifest.mode === 'metrology') labels[1] = 'DataServer → callback'
  const status = (i: number) => {
    if (t.state === 'failed' && i === 2) return 'failed'
    if (t.state === 'paused' && i < 3) return 'paused'
    if (terminal(t.state) && (count[i] || 0) > 0) return 'completed'
    if ((count[i] || 0) > 0 || (i === 2 && (t.received || 0) > 0)) return 'active'
    return 'waiting'
  }
  return (
    <div className="ml-flow">
      <svg viewBox="0 0 1060 224" role="img" aria-labelledby="ml-flow-title ml-flow-description">
        <title id="ml-flow-title">Observed Meter Lab data flow</title>
        <desc id="ml-flow-description">
          {labels.map((l, i) => `${l}: ${status(i)}, ${count[i] || 0} observed.`).join(' ')}
        </desc>
        <defs>
          <marker id="ml-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L6,3 z" fill="currentColor" />
          </marker>
        </defs>
        {[0, 1, 2].map((i) => (
          <path
            key={i}
            className={'ml-edge ' + status(i + 1)}
            d={`M${200 + i * 275},58 L${275 + i * 275},58`}
            markerEnd="url(#ml-arrow)"
          />
        ))}
        <path className={'ml-edge ' + status(4)} d="M937,98 L937,132" markerEnd="url(#ml-arrow)" />
        <path className={'ml-edge ' + status(5)} d="M662,98 L662,132" markerEnd="url(#ml-arrow)" />
        {labels.map((label, i) => {
          const x = i < 4 ? i * 275 : i === 4 ? 825 : 550,
            y = i < 4 ? 18 : 132
          return (
            <g
              key={label}
              transform={`translate(${x},${y})`}
              className={'ml-node ' + status(i) + (selected === i ? ' selected' : '')}
              role="button"
              tabIndex={0}
              aria-label={`${label}, ${status(i)}, ${count[i] || 0} observed. Show details`}
              onClick={() => onSelect(i)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  onSelect(i)
                }
              }}
            >
              <rect width="224" height="80" rx="12" />
              <circle cx="18" cy="23" r="4" />
              <text x="30" y="27" className="ml-node-label">
                {label}
              </text>
              <text x="18" y="53">
                {num(count[i])} {i < 3 ? 'samples' : 'records'}
              </text>
              <text x="18" y="69" className="ml-node-state">
                {status(i)}
              </text>
            </g>
          )
        })}
        <text x="18" y="178" className="ml-flow-note">
          {total
            ? `${num(t.processed)} / ${num(total)} samples processed`
            : 'Choose an agent and scenario to start'}
        </text>
        <text x="18" y="199" className="ml-flow-note">
          {run?.manifest.mode === 'metrology'
            ? 'Actual UTC • SDK one-second callback path'
            : 'Recording clock • bounded stream • ARM acknowledgements'}
        </text>
      </svg>
      <div className="ml-flow-text">
        {labels.map((l, i) => (
          <button key={l} onClick={() => onSelect(i)} aria-pressed={selected === i}>
            {l}: <strong>{status(i)}</strong>
          </button>
        ))}
      </div>
    </div>
  )
}

function StreamScope({ run }: { run: Run }) {
  const duration = run.agent === 'pv' ? 60 : 0.2
  const end = run.telemetry.scenario_time
  const received = run.telemetry.received ?? run.telemetry.processed ?? 0
  const hasSamples = received > 0 && end != null
  const { data: series, error } = useQuery({
    queryKey: ['meter-lab', 'scope', run.id, end, received],
    queryFn: () => get<Series>(`/runs/${run.id}/series?start=${end! - duration}&end=${end}&max_points=1500`),
    enabled: hasSamples,
    placeholderData: (previous) => previous,
    staleTime: 1000,
    gcTime: 10000,
  })
  const traces = useMemo<Data[]>(() => {
    const rows = (series?.inputs.rows ?? []).filter((row) => row[0] != null && row[0] <= end!)
    return (series?.inputs.channels ?? []).map((name, i) => ({
      type: 'scatter',
      mode: rows.length === 1 ? 'lines+markers' : 'lines',
      name,
      x: rows.map((row) => row[0]),
      y: rows.map((row) => row[i + 1]),
      yaxis: run.agent === 'fault' && i >= 3 ? 'y2' : 'y',
      connectgaps: false,
      line: { color: phaseColors[i % 3], width: 1.5, dash: i >= 3 ? 'dot' : 'solid' },
    }))
  }, [series, end, run.agent])
  const layout = useMemo<Partial<Layout>>(
    () => ({
      hovermode: 'x unified',
      margin: { l: 60, r: run.agent === 'fault' ? 60 : 20, t: 35, b: 55 },
      legend: { orientation: 'h', y: 1.15 },
      xaxis: {
        title: { text: run.agent === 'fault' ? 'Recording time (seconds)' : 'UTC (Unix seconds)' },
        range: [end! - duration, end!],
        gridcolor: '#dfe9e2',
        zeroline: false,
      },
      yaxis: {
        title: { text: run.agent === 'fault' ? 'Current (A)' : 'Power (kW / kvar)' },
        gridcolor: '#dfe9e2',
        zerolinecolor: '#b9cdc0',
        autorange: true,
      },
      ...(run.agent === 'fault'
        ? {
            yaxis2: {
              title: { text: 'Voltage (V)' },
              overlaying: 'y',
              side: 'right',
              showgrid: false,
              autorange: true,
            },
          }
        : {}),
    }),
    [duration, end, run.agent],
  )
  return (
    <section className="ml-scope" aria-label="Incoming data scope">
      <div className="ml-scope-heading">
        <div>
          <h3>Incoming data scope</h3>
          <p className="ml-hint">
            Latest {run.agent === 'pv' ? '60 seconds' : '200 milliseconds'} · follows received input
          </p>
        </div>
        <Badge tone={run.state === 'running' ? 'success' : 'neutral'}>
          {run.state === 'running' ? 'Live' : run.state === 'paused' ? 'Paused' : run.state}
        </Badge>
      </div>
      <ErrorBox error={error} />
      {hasSamples && series?.inputs.rows.length ? (
        <Chart data={traces} layout={layout} height={280} title="Incoming data stream" />
      ) : (
        <Empty title={hasSamples ? 'Loading input samples' : 'Waiting for input samples'}>
          The scope displays incoming samples as the run processes them.
        </Empty>
      )}
    </section>
  )
}

function RunCharts({
  run,
  series,
  compare,
  onWindow,
}: {
  run: Run
  series?: Series
  compare?: { run: Run; series: Series }
  onWindow: (range: [number, number] | null) => void
}) {
  const [range, setRange] = useState<[number, number] | null>(null)
  useEffect(() => {
    setRange(null)
  }, [run.id])
  const layout = useMemo<Partial<Layout>>(
    () => ({
      hovermode: 'x unified',
      uirevision: run.id,
      xaxis: {
        title: {
          text:
            run.manifest.mode === 'metrology'
              ? 'Actual UTC (Unix seconds)'
              : run.agent === 'pv'
                ? 'Recording UTC (Unix seconds)'
                : 'Recording time (seconds)',
        },
        ...(range ? { range } : {}),
        gridcolor: '#edf1ee',
        showspikes: true,
      },
      shapes: [
        ...(series?.progress_time != null
          ? [
              {
                type: 'line' as const,
                x0: series.progress_time,
                x1: series.progress_time,
                y0: 0,
                y1: 1,
                yref: 'paper' as const,
                line: { color: '#d29a3d', width: 1, dash: 'dot' as const },
              },
            ]
          : []),
        ...(run.agent === 'fault'
          ? (series?.diagnostics || [])
              .filter((d) => d.onset != null)
              .map((d) => ({
                type: 'line' as const,
                x0: d.onset,
                x1: d.onset,
                y0: 0,
                y1: 1,
                yref: 'paper' as const,
                line: { color: '#b14a65', width: 2 },
              }))
          : []),
      ],
    }),
    [range, run.agent, run.id, run.manifest.mode, series?.progress_time, series?.diagnostics],
  )
  const input = useMemo(
    () =>
      (series?.inputs.channels || []).map(
        (name, i) =>
          ({
            type: 'scattergl',
            mode: 'lines',
            name,
            x: series!.inputs.rows.map((r) => r[0]),
            y: series!.inputs.rows.map((r) => r[i + 1]),
            connectgaps: false,
            line: { color: phaseColors[i % 3], width: 1 },
          }) as Data,
      ),
    [series?.inputs],
  )
  const output = useMemo(() => {
    const all: { run: Run; series: Series }[] = series ? [{ run, series }, ...(compare ? [compare] : [])] : []
    return all.flatMap((item, j) => {
      const rows = item.series.diagnostics,
        suffix = j ? ' · comparison' : ''
      if (run.agent === 'fault')
        return [
          {
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Distance mi' + suffix,
            x: rows.map((r) => r.onset ?? r.start),
            y: rows.map((r) =>
              r.distance?.estimated_distance_km == null
                ? null
                : r.distance.estimated_distance_km / KM_PER_MILE,
            ),
            connectgaps: false,
          } as Data,
        ]
      return ['generation_kw', 'low_kw', 'high_kw'].map(
        (key, i) =>
          ({
            type: 'scatter',
            mode: 'lines',
            name: ['Estimated PV kW', 'Uncalibrated lower band', 'Uncalibrated upper band'][i] + suffix,
            x: rows.map((r) => r.start),
            y: rows.map((r) => r[key] ?? null),
            connectgaps: false,
            line: { width: i ? 1 : 2, dash: j ? 'dot' : 'solid' },
            ...(i === 2 ? { fill: 'tonexty', fillcolor: 'rgba(8,125,113,.08)' } : {}),
          }) as Data,
      )
    })
  }, [run, series, compare])
  const energy = useMemo(
    () =>
      ['import_kwh', 'export_kwh', 'generation_kwh'].map(
        (key, i) =>
          ({
            type: 'scatter',
            mode: 'lines',
            name: ['Measured import kWh', 'Measured export kWh', 'Estimated PV kWh'][i],
            x: (series?.diagnostics || []).map((r) => r.start),
            y: (series?.diagnostics || []).map((r) => r[key] ?? null),
            connectgaps: false,
          }) as Data,
      ),
    [series?.diagnostics],
  )
  const coverage = useMemo(
    () =>
      ['p_seconds', 'estimated_seconds'].map(
        (key, i) =>
          ({
            type: 'scatter',
            mode: 'lines',
            name: ['Measured coverage %', 'Estimated-energy coverage %'][i],
            x: (series?.diagnostics || []).map((r) => r.start),
            y: (series?.diagnostics || []).map((r) => (100 * (r[key] || 0)) / 900),
          }) as Data,
      ),
    [series?.diagnostics],
  )
  const chart = (data: Data[], title: string) => (
    <Chart
      data={data}
      title={title}
      height={280}
      layout={layout}
      onRange={(a, b) => {
        setRange([a, b])
        onWindow([a, b])
      }}
      cursorGroup={run.id}
    />
  )
  return (
    <div className="ml-chart-grid">
      <Panel
        title={run.agent === 'pv' ? 'Signed aggregate power' : 'Current waveforms'}
        subtitle={
          run.agent === 'pv'
            ? 'Positive import / negative export. Inputs in kW and kvar.'
            : 'IA, IB, IC · amperes · synchronized native samples'
        }
        actions={
          <button
            className="button subtle"
            onClick={() => {
              setRange(null)
              onWindow(null)
            }}
          >
            Reset zoom
          </button>
        }
      >
        {chart(run.agent === 'pv' ? input : input.slice(0, 3), 'Scenario input power/current')}
      </Panel>
      {run.agent === 'fault' && (
        <Panel title="Voltage waveforms" subtitle="UA, UB, UC · volts · gaps preserved">
          {chart(input.slice(3), 'Scenario input voltage')}
        </Panel>
      )}
      {run.agent === 'pv' && series?.voltage && (
        <Panel title="Voltage diagnostics" subtitle="Available phase voltages from the scenario · volts">
          {chart(
            series.voltage.channels.map(
              (name, i) =>
                ({
                  type: 'scattergl',
                  mode: 'lines',
                  name,
                  x: series.voltage!.rows.map((r) => r[0]),
                  y: series.voltage!.rows.map((r) => r[i + 1]),
                  connectgaps: false,
                }) as Data,
            ),
            'Available phase voltages',
          )}
        </Panel>
      )}
      <Panel
        title={run.agent === 'pv' ? 'Estimated PV generation' : 'Fault distance'}
        subtitle="ARM analysis diagnostics · recording time"
      >
        {series?.diagnostics.length ? (
          chart(output, 'ARM analysis output')
        ) : (
          <Empty title="Waiting for analysis">
            {run.agent === 'pv'
              ? 'Results arrive after 15-minute intervals. PV estimates require sufficient learning and model quality.'
              : 'Fault results arrive after the complete waveform segment has been processed.'}
          </Empty>
        )}
      </Panel>
      {run.agent === 'pv' && (
        <>
          <Panel
            title="Interval energy"
            subtitle="Measured import/export and estimated generation remain separate"
          >
            {chart(energy, 'Measured and estimated interval energy')}
          </Panel>
          <Panel title="Energy coverage" subtitle="Missing energy is not extrapolated">
            {chart(coverage, 'Measured and estimated energy coverage')}
          </Panel>
        </>
      )}
      <p className="ml-chart-caption">
        Amber dotted line: processed replay progress.{' '}
        {run.agent === 'fault'
          ? 'Rose line: detected or manual inception.'
          : 'Unavailable generation remains a gap. Uncertainty is uncalibrated for daytime PV accuracy.'}{' '}
        Zoom and cursor position are shared across charts.
      </p>
    </div>
  )
}

export default function MeterLab() {
  const client = useQueryClient(),
    hosted = import.meta.env.VITE_HOSTED === 'true'
  const [agent, setAgent] = useState<Agent>('pv'),
    [selected, setSelected] = useState('pv_export'),
    [source, setSource] = useState<'preset' | 'dataset' | 'upload'>('preset')
  const [meter, setMeter] = useState('GENX_PP'),
    [speed, setSpeed] = useState('fastest'),
    [mode, setMode] = useState('replay'),
    [seedRun, setSeedRun] = useState('')
  const [parameters, setParameters] = useState<Json>({
    seed: 42,
    magnitude: 1,
    load_scale: 1,
    pv_scale: 1,
    cloud_variation: 0,
    noise: 0,
    polarity: 1,
    onset_seconds: 0.1,
    fault_type: 'AG',
    distance_km: 10,
    frequency: 50,
    outage_seconds: 0,
  })
  const [configuration, setConfiguration] = useState<Record<string, string>>({}),
    [upload, setUpload] = useState<Json | null>(null),
    [mapping, setMapping] = useState<Json>({
      timestamp: 'timestamp',
      timestamp_unit: 'utc_seconds',
      channels: {},
      units: {},
      validity: {},
      asset_level: 'meter',
      aggregate_power: false,
      sample_rate: 1,
      grid_frequency: 50,
    })
  const [chartWindow, setChartWindow] = useState<[number, number] | null>(null)
  const [preview, setPreview] = useState<Json | null>(null),
    [scenario, setScenario] = useState<Json | null>(null),
    [runId, setRunId] = useState(() => new URLSearchParams(location.search).get('run') || '')
  const [stage, setStage] = useState(2),
    [tab, setTab] = useState('charts'),
    [error, setError] = useState<unknown>(),
    [busy, setBusy] = useState(false),
    [compareId, setCompareId] = useState('')
  const { data: cap, error: capError } = useQuery({
    queryKey: ['meter-lab', 'capabilities'],
    queryFn: () => get<Capabilities>('/capabilities'),
    enabled: !hosted,
    refetchInterval: 5000,
  })
  const { data: catalog } = useQuery({
    queryKey: ['meter-lab', 'scenarios'],
    queryFn: () => get<{ catalog: CatalogEntry[]; saved: Json[] }>('/scenarios'),
    enabled: !hosted,
  })
  const { data: runs = [] } = useQuery({
    queryKey: ['meter-lab', 'runs'],
    queryFn: () => get<Run[]>('/runs'),
    enabled: !hosted,
    refetchInterval: 5000,
  })
  const { data: run, error: runError } = useQuery({
    queryKey: ['meter-lab', 'run', runId],
    queryFn: () => get<Run>('/runs/' + runId),
    enabled: !!runId && !hosted,
    refetchInterval: (q) => (terminal(q.state.data?.state) ? 5000 : 1000),
  })
  const { data: series } = useQuery({
    queryKey: [
      'meter-lab',
      'series',
      runId,
      run?.telemetry.diagnostics,
      run?.telemetry.processed,
      chartWindow,
    ],
    queryFn: () =>
      get<Series>(
        '/runs/' + runId + '/series' + (chartWindow ? `?start=${chartWindow[0]}&end=${chartWindow[1]}` : ''),
      ),
    enabled: !!run && !hosted,
    staleTime: 1000,
    gcTime: 10000,
  })
  const { data: stored } = useQuery({
    queryKey: ['meter-lab', 'outcomes', runId, run?.telemetry.stored_data, run?.telemetry.stored_events],
    queryFn: () => get<{ rows: Json[]; total: number }>('/runs/' + runId + '/outcomes?limit=500'),
    enabled: !!run && !hosted,
  })
  const { data: logs } = useQuery({
    queryKey: ['meter-lab', 'logs', runId],
    queryFn: () => get<{ name: string; text: string }[]>('/runs/' + runId + '/logs'),
    enabled: !!run && tab === 'logs',
    refetchInterval: terminal(run?.state) ? false : 2000,
  })
  const { data: comparison, error: compareError } = useQuery({
    queryKey: ['meter-lab', 'compare', runId, compareId, chartWindow],
    queryFn: () =>
      get<{ right: Run; right_series: Series; configuration_equal: boolean }>(
        `/compare?left=${runId}&right=${compareId}` +
          (chartWindow ? `&start=${chartWindow[0]}&end=${chartWindow[1]}` : ''),
      ),
    enabled: !!runId && !!compareId && runId !== compareId,
  })
  const descriptors = cap?.agents || [],
    descriptor = descriptors.find((d) => d.id === agent),
    entry = catalog?.catalog.find((c) => c.id === selected && c.source === source && c.agent === agent)
  const active = runs.find((r) => !terminal(r.state)),
    frozen = !!run && !terminal(run.state),
    ready = cap?.available && !active && !frozen
  const loadedRun = useRef('')
  useEffect(() => {
    setChartWindow(null)
  }, [runId])
  useEffect(() => {
    if (runId) {
      const url = new URL(location.href)
      url.searchParams.set('run', runId)
      history.replaceState(null, '', url)
    }
  }, [runId])
  useEffect(() => {
    if (entry?.configuration && !runId) setConfiguration(entry.configuration)
  }, [entry, runId])
  useEffect(() => {
    if (!run || loadedRun.current === run.id) return
    const saved = catalog?.saved.find((s) => s.id === run.manifest.scenario_id)
    if (!saved) return
    loadedRun.current = run.id
    setAgent(run.agent)
    setSelected(saved.request.source_id)
    setSource(saved.request.source)
    setParameters(saved.request.parameters)
    setMapping(saved.request.mapping)
    setConfiguration(run.manifest.configuration)
    setMeter(run.manifest.meter_form)
    setSpeed(run.manifest.speed)
    setMode(run.manifest.mode)
    setSeedRun(
      run.manifest.seed_fixture ? 'fixture:' + run.manifest.seed_fixture : run.manifest.seed_run_id || '',
    )
    setScenario(saved)
  }, [run, catalog])
  const change = (key: string, value: unknown) => {
    setParameters((p) => ({ ...p, [key]: value }))
    setScenario(null)
    setPreview(null)
  }
  const selectAgent = (value: Agent) => {
    setAgent(value)
    if (value === 'fault') setMeter('GENX_PP')
    setSelected(value === 'pv' ? 'pv_export' : '1A_val1')
    setSource('preset')
    setConfiguration({})
    setScenario(null)
    setPreview(null)
    setSeedRun('')
    setMode('replay')
  }
  const perform = async (fn: () => Promise<void>) => {
    setBusy(true)
    setError(undefined)
    try {
      await fn()
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }
  const request = () => ({
    agent,
    source,
    source_id: selected,
    parameters,
    configuration,
    mapping: {
      ...mapping,
      channels: Object.fromEntries(Object.entries(mapping.channels).filter(([, v]) => v)),
      units: Object.fromEntries(Object.entries(mapping.units).filter(([, v]) => v)),
      validity: Object.fromEntries(Object.entries(mapping.validity).filter(([, v]) => v)),
    },
  })
  const prepare = async () => {
    const s = await post<Json>('/scenarios', request())
    setScenario(s)
    setPreview(await get<Json>('/scenarios/' + s.id + '/preview'))
    return s
  }
  const start = () =>
    perform(async () => {
      const s = scenario || (await prepare())
      const r = await post<Run>('/runs', {
        scenario_id: s.id,
        meter_form: meter,
        speed,
        mode,
        seed_run_id: seedRun && !seedRun.startsWith('fixture:') ? seedRun : null,
        seed_fixture: seedRun.startsWith('fixture:') ? seedRun.slice(8) : null,
      })
      setRunId(r.id)
      setTab('charts')
      setCompareId('')
      await client.invalidateQueries({ queryKey: ['meter-lab', 'runs'] })
      await client.invalidateQueries({ queryKey: ['meter-lab', 'scenarios'] })
    })
  const control = (action: string) =>
    perform(async () => {
      await post('/runs/' + runId + '/control', { action })
      await client.invalidateQueries({ queryKey: ['meter-lab', 'run', runId] })
    })
  const doUpload = (files: FileList | null) =>
    perform(async () => {
      if (!files?.length) return
      const body = new FormData()
      Array.from(files).forEach((f) => body.append('files', f))
      const response = await fetch('/api/v1/meter-lab/uploads', { method: 'POST', body })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail)
      setUpload(result)
      setSource('upload')
      setSelected(result.id)
      setScenario(null)
      setPreview(null)
    })
  const newRun = () => {
    loadedRun.current = ''
    setRunId('')
    setScenario(null)
    setPreview(null)
    setCompareId('')
    history.replaceState(null, '', '/meter-lab')
  }
  if (hosted)
    return (
      <>
        <PageTitle
          eyebrow="METER LAB"
          title="A local meter test bench"
          description="Replay scenarios through HW 4.2 ARM agents."
        />
        <Empty title="Available on your local workstation">
          Meter Lab requires Windows and Ubuntu-22.04 WSL. Emulator controls are unavailable in hosted
          deployments.
        </Empty>
      </>
    )
  return (
    <div className="meter-lab">
      <PageTitle
        eyebrow="HW 4.2 · GEN5 RIVA"
        title="Meter Lab"
        description="Stream a scenario. Observe the ARM agent. Inspect the evidence."
      >
        <Badge tone={cap?.available ? 'success' : 'neutral'}>
          <Cpu size={13} />
          {cap?.available ? 'ARM lab ready' : 'Lab unavailable'}
        </Badge>
        {run && terminal(run.state) && (
          <button className="button secondary" onClick={newRun}>
            <FlaskConical size={16} />
            New run
          </button>
        )}
      </PageTitle>
      <ErrorBox error={error || capError || runError || compareError} />
      {cap && !cap.available && (
        <div className="ml-notice">
          <Cpu size={20} />
          <div>
            <strong>Local runtime unavailable</strong>
            <p>{cap.reason}</p>
            <code>python3 meter_emulator/lab_setup.py</code>
            <p>Run installation in Ubuntu-22.04, then start DI Validator with scripts/start.ps1.</p>
          </div>
        </div>
      )}
      <div className="ml-workbench">
        <Panel
          title="Configure a scenario"
          subtitle={
            frozen
              ? 'Setup is frozen for this run.'
              : 'Clean state is the default. Every run preserves its input and configuration.'
          }
          className="ml-setup"
        >
          <fieldset disabled={frozen || busy}>
            <Field label="Agent">
              <select value={agent} onChange={(e) => selectAgent(e.target.value as Agent)}>
                <option value="pv">PV Detection Agent</option>
                <option value="fault">Fault Location Agent</option>
              </select>
            </Field>
            <div className="ml-fields">
              <Field label="Meter form">
                <select value={meter} onChange={(e) => setMeter(e.target.value)}>
                  {(descriptor?.forms || ['GENX_PP']).map((f) => (
                    <option key={f} value={f}>
                      {f === 'GENX_SP' ? 'Singlephase' : 'Polyphase'}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Playback speed">
                <select
                  value={speed}
                  disabled={mode === 'metrology'}
                  onChange={(e) => setSpeed(e.target.value)}
                >
                  {['fastest', '0.1', '1', '10', '100', '1000'].map((s) => (
                    <option key={s} value={s}>
                      {s === 'fastest' ? 'Fastest' : s + '×'}
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            <Field label="Input path">
              <select
                value={mode}
                onChange={(e) => {
                  setMode(e.target.value)
                  if (e.target.value === 'metrology') setSpeed('1')
                }}
              >
                <option value="replay">Scenario replay · recording clock</option>
                {agent === 'pv' && (
                  <option value="metrology" disabled={!cap?.metrology_available}>
                    SDK metrology check · actual UTC
                  </option>
                )}
              </select>
            </Field>
            <Field label="Scenario source">
              <select
                value={source}
                onChange={(e) => {
                  setSource(e.target.value as typeof source)
                  setScenario(null)
                  setPreview(null)
                  const first = catalog?.catalog.find((c) => c.agent === agent && c.source === e.target.value)
                  if (first) setSelected(first.id)
                }}
              >
                <option value="preset">Editable preset</option>
                <option value="dataset">Existing compatible dataset</option>
                <option value="upload">Upload measurements</option>
              </select>
            </Field>
            {source !== 'upload' ? (
              <Field label="Scenario">
                <select
                  value={selected}
                  onChange={(e) => {
                    setSelected(e.target.value)
                    setScenario(null)
                    setPreview(null)
                  }}
                >
                  {catalog?.catalog
                    .filter((c) => c.agent === agent && c.source === source)
                    .map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name}
                      </option>
                    ))}
                </select>
              </Field>
            ) : (
              <>
                <Field label="CSV, Parquet, or COMTRADE pair">
                  <input
                    type="file"
                    multiple
                    accept=".csv,.parquet,.cfg,.dat"
                    onChange={(e) => doUpload(e.target.files)}
                  />
                </Field>
                {upload && (
                  <small>
                    {upload.name} · {num(upload.bytes)} bytes
                  </small>
                )}
              </>
            )}
            {entry?.description && <p className="ml-hint">{entry.description}</p>}
            {entry?.warning && <p className="ml-warning">{entry.warning}</p>}
            {source === 'upload' && upload?.format !== 'comtrade' && (
              <details open>
                <summary>Channel and unit mapping</summary>
                <Field label="Timestamp column">
                  <select
                    value={mapping.timestamp}
                    onChange={(e) => {
                      setMapping({ ...mapping, timestamp: e.target.value })
                      setScenario(null)
                    }}
                  >
                    {(upload?.columns || []).map((c: string) => (
                      <option key={c}>{c}</option>
                    ))}
                  </select>
                </Field>
                <Field label="Timestamp convention">
                  <select
                    value={mapping.timestamp_unit}
                    onChange={(e) => {
                      setMapping({ ...mapping, timestamp_unit: e.target.value })
                      setScenario(null)
                    }}
                  >
                    <option value="utc_seconds">UTC Unix seconds</option>
                    <option value="iso_utc">ISO UTC</option>
                    {agent === 'fault' && <option value="seconds">Recording-relative seconds</option>}
                  </select>
                </Field>
                {(agent === 'pv' ? ['P', 'Q', 'VA', 'VB', 'VC'] : ['IA', 'IB', 'IC', 'UA', 'UB', 'UC']).map(
                  (c) => (
                    <div className="ml-fields" key={c}>
                      <Field label={c + ' column' + (c.startsWith('V') ? ' (optional)' : '')}>
                        <select
                          value={mapping.channels[c] || ''}
                          onChange={(e) => {
                            setMapping({ ...mapping, channels: { ...mapping.channels, [c]: e.target.value } })
                            setScenario(null)
                          }}
                        >
                          <option value="">Select column</option>
                          {(upload?.columns || []).map((n: string) => (
                            <option key={n}>{n}</option>
                          ))}
                        </select>
                      </Field>
                      {agent === 'pv' && ['P', 'Q'].includes(c) && (
                        <Field label={c + ' validity column (optional)'}>
                          <select
                            value={mapping.validity[c] || ''}
                            onChange={(e) => {
                              setMapping({
                                ...mapping,
                                validity: { ...mapping.validity, [c]: e.target.value },
                              })
                              setScenario(null)
                            }}
                          >
                            <option value="">Use finite values</option>
                            {(upload?.columns || []).map((n: string) => (
                              <option key={n}>{n}</option>
                            ))}
                          </select>
                        </Field>
                      )}
                      <Field label={c + ' unit'}>
                        <select
                          value={mapping.units[c] || ''}
                          onChange={(e) => {
                            setMapping({ ...mapping, units: { ...mapping.units, [c]: e.target.value } })
                            setScenario(null)
                          }}
                        >
                          <option value="">Select unit</option>
                          {(c === 'P'
                            ? ['W', 'kW']
                            : c === 'Q'
                              ? ['var', 'kvar']
                              : c[0] === 'I'
                                ? ['A', 'kA']
                                : ['V', 'kV']
                          ).map((u) => (
                            <option key={u}>{u}</option>
                          ))}
                        </select>
                      </Field>
                    </div>
                  ),
                )}
                <Field label="Sampling frequency (Hz)">
                  <input
                    type="number"
                    value={mapping.sample_rate}
                    onChange={(e) => {
                      setMapping({ ...mapping, sample_rate: Number(e.target.value) })
                      setScenario(null)
                    }}
                  />
                </Field>
                {agent === 'pv' && (
                  <label className="ml-check">
                    <input
                      type="checkbox"
                      checked={mapping.aggregate_power}
                      onChange={(e) => {
                        setMapping({ ...mapping, aggregate_power: e.target.checked })
                        setScenario(null)
                      }}
                    />
                    These are genuine one-second aggregate measurements from one meter.
                  </label>
                )}
              </details>
            )}
            <details>
              <summary>Scenario parameters</summary>
              <div className="ml-fields">
                <Field label="Start offset (seconds)">
                  <input
                    type="number"
                    min="0"
                    value={parameters.start_offset_seconds || 0}
                    onChange={(e) => change('start_offset_seconds', Number(e.target.value))}
                  />
                </Field>
                <Field label="Duration (seconds, optional)">
                  <input
                    type="number"
                    min="1"
                    value={parameters.duration_seconds ?? ''}
                    onChange={(e) =>
                      change('duration_seconds', e.target.value === '' ? null : Number(e.target.value))
                    }
                  />
                </Field>
              </div>
              <div className="ml-fields">
                {[
                  ['seed', 'Random seed'],
                  ['magnitude', 'Magnitude multiplier'],
                  ['noise', 'Noise fraction'],
                  ...(agent === 'pv'
                    ? [
                        ['load_scale', 'Load multiplier'],
                        ['pv_scale', 'PV multiplier'],
                        ['cloud_variation', 'Cloud variation'],
                      ]
                    : []),
                ].map(([key, label]) => (
                  <Field key={key} label={label}>
                    <input
                      type="number"
                      step="any"
                      value={parameters[key]}
                      onChange={(e) => change(key, Number(e.target.value))}
                    />
                  </Field>
                ))}
              </div>
              {selected === 'synthetic_fault' && (
                <>
                  <Field label="Fault loop">
                    <select
                      value={parameters.fault_type}
                      onChange={(e) => change('fault_type', e.target.value)}
                    >
                      {['AG', 'BG', 'CG', 'AB', 'BC', 'CA', 'POS', 'none'].map((l) => (
                        <option key={l}>{l}</option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Fault distance (km on configured line)">
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      value={parameters.distance_km}
                      onChange={(e) => change('distance_km', Number(e.target.value))}
                    />
                  </Field>
                  <Field label="Fault onset (seconds)">
                    <input
                      type="number"
                      step="0.01"
                      value={parameters.onset_seconds}
                      onChange={(e) => change('onset_seconds', Number(e.target.value))}
                    />
                  </Field>
                  <Field label="Grid frequency">
                    <select
                      value={parameters.frequency}
                      onChange={(e) => change('frequency', Number(e.target.value))}
                    >
                      <option value={50}>50 Hz</option>
                      <option value={60}>60 Hz</option>
                    </select>
                  </Field>
                </>
              )}
              <div className="ml-fields">
                <Field label="Outage start (s from start)">
                  <input
                    type="number"
                    min="0"
                    value={parameters.outage_start ?? ''}
                    onChange={(e) =>
                      change('outage_start', e.target.value === '' ? null : Number(e.target.value))
                    }
                  />
                </Field>
                <Field label="Outage duration (s)">
                  <input
                    type="number"
                    min="0"
                    value={parameters.outage_seconds}
                    onChange={(e) => change('outage_seconds', Number(e.target.value))}
                  />
                </Field>
              </div>
              {agent === 'pv' && (
                <Field label="Input polarity">
                  <select
                    value={parameters.polarity}
                    onChange={(e) => change('polarity', Number(e.target.value))}
                  >
                    <option value={1}>Normal</option>
                    <option value={-1}>Reverse recorded real power</option>
                  </select>
                </Field>
              )}
            </details>
            <details>
              <summary>Agent configuration</summary>
              <p className="ml-hint">
                Precision-sensitive values are sent as decimal strings. Invalid updates are rejected.
              </p>
              {Object.entries({ ...descriptor?.configuration, ...configuration }).map(([key, value]) => (
                <Field key={key} label={key.replaceAll('_', ' ')}>
                  <input
                    value={value}
                    onChange={(e) => {
                      setConfiguration((c) => ({ ...c, [key]: e.target.value }))
                      setScenario(null)
                      setPreview(null)
                    }}
                  />
                </Field>
              ))}
            </details>
            <Field label="Initial state">
              <select value={seedRun} onChange={(e) => setSeedRun(e.target.value)}>
                {agent === 'pv' &&
                  (cap?.seed_fixtures || []).map((f) => (
                    <option
                      key={f.id}
                      value={'fixture:' + f.id}
                      disabled={
                        selected !== 'warm_tail' && (parameters.start_offset_seconds || 0) < 32 * 86400
                      }
                    >
                      {f.name} · use warm tail
                    </option>
                  ))}
                <option value="">Clean state · no learned history</option>
                {agent === 'pv' &&
                  runs
                    .filter((r) => r.agent === 'pv' && r.checkpoint_available)
                    .map((r) => (
                      <option key={r.id} value={r.id}>
                        {r.name} · {r.id.slice(0, 8)} checkpoint
                      </option>
                    ))}
              </select>
            </Field>
            <div className="ml-actions">
              <button
                className="button secondary"
                disabled={!selected}
                onClick={() =>
                  perform(async () => {
                    await prepare()
                  })
                }
              >
                Validate & preview
              </button>
              <button className="button primary" disabled={!ready || !selected} onClick={start}>
                <Play size={16} />
                {busy ? 'Preparing…' : 'Start run'}
              </button>
            </div>
          </fieldset>
          <p className="ml-hint">
            {descriptor?.input}. {descriptor?.timing}. Physical-meter acquisition and resource qualification
            are outside this lab.
          </p>
        </Panel>
        <div className="ml-observe">
          <Panel
            title={run ? run.name : 'Observe the data flow'}
            subtitle={
              run
                ? `${run.id.slice(0, 12)} · ${new Date(run.created_at).toLocaleString()} · ${run.manifest.meter_form}`
                : 'Counters advance only when the runtime observes activity.'
            }
            actions={
              run ? (
                <Badge
                  tone={run.state === 'failed' ? 'danger' : run.state === 'completed' ? 'success' : 'neutral'}
                >
                  {run.state}
                </Badge>
              ) : undefined
            }
          >
            <Flow run={run} selected={stage} onSelect={setStage} />
            <div className="ml-stage-detail" aria-live="polite">
              {stage === 0 ? (
                <>
                  <strong>Scenario</strong>
                  <span>
                    {num(run?.manifest.total_samples)} frozen samples ·{' '}
                    {run?.manifest.input_sha256?.slice(0, 16) || 'No input frozen'}
                  </span>
                </>
              ) : stage === 1 ? (
                <>
                  <strong>Replay transport</strong>
                  <span>
                    {num(run?.telemetry.transmitted)} transmitted · {num(run?.telemetry.received)}{' '}
                    acknowledged · queue {num(run?.telemetry.queue)}
                  </span>
                </>
              ) : stage === 2 ? (
                <>
                  <strong>ARM processing</strong>
                  <span>
                    {num(run?.telemetry.processed)} processed · {num(run?.telemetry.diagnostics)} analysis
                    records
                  </span>
                </>
              ) : stage === 3 || stage === 4 ? (
                <>
                  <strong>DataServer delivery</strong>
                  <span>
                    {num(run?.telemetry.stored_data)} data rows · {num(run?.telemetry.stored_events)} events ·{' '}
                    {num(run?.telemetry.pending)} pending · {num(run?.telemetry.rejected)} rejected
                  </span>
                </>
              ) : (
                <>
                  <strong>Analysis diagnostics</strong>
                  <span>Unquantized ARM output is saved independently of delivery quotas.</span>
                </>
              )}
            </div>
            {run && (
              <>
                <div className="ml-stats">
                  <div>
                    <small>PROCESSED</small>
                    <strong>
                      {num(run.telemetry.processed)} <span>/ {num(run.manifest.total_samples)}</span>
                    </strong>
                  </div>
                  <div>
                    <small>ACHIEVED SPEED</small>
                    <strong>
                      {(run.telemetry.actual_speed ?? 0).toLocaleString(undefined, {
                        maximumSignificantDigits: 3,
                      })}
                      ×
                    </strong>
                  </div>
                  <div>
                    <small>STORED OUTCOMES</small>
                    <strong>
                      {num((run.telemetry.stored_data || 0) + (run.telemetry.stored_events || 0))}
                    </strong>
                  </div>
                </div>
                <progress
                  className="ml-progress"
                  max={run.manifest.total_samples || 1}
                  value={run.telemetry.processed || 0}
                />
                <div className="ml-run-actions">
                  {!terminal(run.state) ? (
                    <>
                      <button
                        className="button secondary"
                        disabled={
                          run.manifest.mode === 'metrology' ||
                          !['running', 'paused'].includes(run.state) ||
                          busy
                        }
                        onClick={() => control(run.state === 'paused' ? 'resume' : 'pause')}
                      >
                        {run.state === 'paused' ? <Play size={15} /> : <Pause size={15} />}{' '}
                        {run.state === 'paused' ? 'Resume' : 'Pause'}
                      </button>
                      <button className="button secondary" disabled={busy} onClick={() => control('stop')}>
                        <Square size={15} />
                        Stop
                      </button>
                      <small>Navigation and refresh keep this run alive.</small>
                    </>
                  ) : (
                    <a className="button secondary" href={`/api/v1/meter-lab/runs/${run.id}/export`}>
                      <Download size={15} />
                      Download evidence
                    </a>
                  )}
                </div>
                <StreamScope key={run.id} run={run} />
                {run.telemetry.error && <ErrorBox error={run.telemetry.error} />}
              </>
            )}
          </Panel>
          {!run && preview && (
            <Panel
              title="Input preview"
              subtitle={`${num(preview.samples)} samples · full-resolution input is retained`}
            >
              <Chart
                data={(preview.channels as string[]).map(
                  (name, i) =>
                    ({
                      type: 'scattergl',
                      mode: 'lines',
                      name,
                      x: preview.rows.map((r: number[]) => r[0]),
                      y: preview.rows.map((r: number[]) => r[i + 1]),
                      connectgaps: false,
                    }) as Data,
                )}
                height={320}
                title="Scenario input preview"
              />
            </Panel>
          )}
          {!run && !preview && (
            <Empty title="Ready for a controlled experiment">
              Select an agent and a scenario, inspect the input preview, then start a run. Both ARM
              diagnostics and actual stored outcomes will appear here.
            </Empty>
          )}
        </div>
      </div>
      {run && (
        <>
          <div className="ml-tabs" role="tablist" aria-label="Run evidence">
            {['charts', 'results', 'checks', 'resources', 'logs'].map((t) => (
              <button role="tab" aria-selected={tab === t} key={t} onClick={() => setTab(t)}>
                {t}
              </button>
            ))}
          </div>
          {tab === 'charts' && (
            <RunCharts
              run={run}
              series={series}
              onWindow={setChartWindow}
              compare={comparison ? { run: comparison.right, series: comparison.right_series } : undefined}
            />
          )}
          {tab === 'results' && (
            <div className="ml-result-grid">
              <Panel
                title="ARM analysis diagnostics"
                subtitle="These records come from the selected ARM agent's numerical core."
              >
                <div className="ml-table-scroll">
                  <table>
                    <thead>
                      <tr>
                        {(run.agent === 'pv'
                          ? ['UTC interval', 'State', 'PV kW', 'Reason', 'Import / export kWh']
                          : [
                              'Segment / onset',
                              'Classification',
                              'Loop',
                              'Distance',
                              'Measured R / X (Ω)',
                              'Status',
                            ]
                        ).map((x) => (
                          <th key={x}>{x}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(series?.diagnostics || []).slice(-300).map((r, i) => (
                        <tr key={i}>
                          {run.agent === 'pv' ? (
                            <>
                              <td>{new Date(r.start * 1000).toISOString()}</td>
                              <td>{r.state}</td>
                              <td>{r.generation_kw == null ? 'Unavailable' : num(r.generation_kw)}</td>
                              <td>{r.reason}</td>
                              <td>
                                {num(r.import_kwh)} / {num(r.export_kwh)}
                              </td>
                            </>
                          ) : (
                            <>
                              <td>
                                {r.segment} / {r.onset ?? '—'} s
                              </td>
                              <td>{r.fault_type_heuristic || '—'}</td>
                              <td>{r.distance?.fault_loop || '—'}</td>
                              <td>
                                {r.distance?.estimated_distance_km != null
                                  ? miles(r.distance.estimated_distance_km)
                                  : r.distance?.apparent_distance_km != null
                                    ? `Apparent ${miles(r.distance.apparent_distance_km)}`
                                    : '—'}
                              </td>
                              <td>
                                {r.distance?.x_apparent_ohm == null
                                  ? '—'
                                  : `${r.distance.r_apparent_ohm.toFixed(3)} / ${r.distance.x_apparent_ohm.toFixed(3)}`}
                              </td>
                              <td title={r.distance?.reason || r.reason || ''}>
                                {withheld[r.distance?.status] || r.distance?.status || r.status}
                                {r.distance?.reason && <div className="ml-hint">{r.distance.reason}</div>}
                              </td>
                            </>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="ml-hint">
                  Showing the latest 300 records; export contains the complete evidence.
                  {run.agent === 'fault' &&
                    ` Line: ${run.manifest.feeder?.line_source || 'generic example settings (not feeder-referenced)'}.`}
                </p>
              </Panel>
              {run.agent === 'fault' && run.manifest.feeder && (
                <NetworkPlacement feeder={run.manifest.feeder} rows={series?.diagnostics || []} />
              )}
              <Panel
                title="DataServer-stored outcomes"
                subtitle={`${stored?.total || 0} actual rows · SDK timestamps use actual UTC`}
              >
                <div className="ml-outcomes">
                  {stored?.rows.map((r, i) => (
                    <details key={i}>
                      <summary>
                        <Badge>{r.table === 'AgentEvents' ? 'Event' : 'Stored data'}</Badge> {r.Id} ·{' '}
                        {r.bytes} bytes · {new Date(r.TimeStamp * 1000).toISOString()}
                      </summary>
                      <pre>{JSON.stringify(r.decoded || r.payload, null, 2)}</pre>
                    </details>
                  ))}
                  {!stored?.total && (
                    <Empty title="No stored outcomes yet">
                      Routine PV output is batched hourly. Fault output follows segment completion. Policy
                      rejections remain visible in delivery counters.
                    </Empty>
                  )}
                </div>
              </Panel>
            </div>
          )}
          {tab === 'checks' && (
            <Panel
              title="Independent verification"
              subtitle="Numerical parity, scenario expectations, and delivery are assessed separately."
            >
              <div className="ml-check-list">
                {run.checks.checks.map((c, i) => (
                  <div key={i}>
                    <Badge
                      tone={c.verdict === 'pass' ? 'success' : c.verdict === 'fail' ? 'danger' : 'neutral'}
                    >
                      {c.verdict}
                    </Badge>
                    <div>
                      <small>{c.category}</small>
                      <strong>{c.name}</strong>
                      <p>{c.detail}</p>
                    </div>
                  </div>
                ))}
                {!run.checks.checks.length && (
                  <Empty title="Checks pending">
                    Evaluation begins after the run finishes and pending writes are drained.
                  </Empty>
                )}
              </div>
            </Panel>
          )}
          {tab === 'resources' && (
            <Panel
              title="Resource observations"
              subtitle="QEMU measurements do not qualify resources on a physical meter."
            >
              <div className="ml-stats">
                <div>
                  <small>PEAK RSS / AGENT POLICY</small>
                  <strong>{num(run.telemetry.resources?.peak_rss_kb)} / 2,048 KB</strong>
                </div>
                <div>
                  <small>OBSERVED CPU / AGENT POLICY</small>
                  <strong>{num(run.telemetry.resources?.max_observed_cpu_percent)} / 2%</strong>
                </div>
                <div>
                  <small>SHARED CONTAINER PEAK / CEILING</small>
                  <strong>{num(run.telemetry.resources?.container_peak_kb)} / 65,000 KB</strong>
                </div>
              </div>
              {(run.telemetry.resources?.ram_overrun || run.telemetry.resources?.cpu_overrun) && (
                <p className="ml-warning">
                  Measured agent policy overrun. The lab has not increased the inherited allocation.
                </p>
              )}
              <pre>{JSON.stringify(run.telemetry.resources || {}, null, 2)}</pre>
              <p>
                Persistent agent state: {num(run.telemetry.persistent_bytes)} bytes. Stored output:{' '}
                {num(run.telemetry.stored_bytes)} bytes, including {num(run.telemetry.routine_data_bytes)}{' '}
                bytes of routine data.
              </p>
              <p className="ml-hint">
                Agent flash policy: 2,048 KB. Delivery budgets use actual elapsed time, so accelerated runs
                may reach their quota before scenario time ends.
              </p>
            </Panel>
          )}
          {tab === 'logs' && (
            <Panel
              title="Run logs"
              subtitle="Bounded tails from the isolated supervisor, ARM agent, and DataServer."
            >
              {logs?.map((log) => (
                <details key={log.name}>
                  <summary>{log.name}</summary>
                  <pre>{log.text}</pre>
                </details>
              ))}
            </Panel>
          )}
        </>
      )}
      <Panel
        title="Saved runs"
        subtitle="Immutable inputs, build identity, actual outcomes, and checks are preserved for every run."
        actions={
          active && active.id !== runId ? (
            <button className="button secondary" onClick={() => setRunId(active.id)}>
              Reconnect to active run <ArrowRight size={15} />
            </button>
          ) : undefined
        }
      >
        {run && terminal(run.state) && (
          <div className="ml-compare">
            <Field label="Compare with a second run">
              <select value={compareId} onChange={(e) => setCompareId(e.target.value)}>
                <option value="">No comparison</option>
                {runs
                  .filter((r) => r.id !== runId && r.agent === run.agent && terminal(r.state))
                  .map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.name} · {r.id.slice(0, 8)} · {r.state}
                    </option>
                  ))}
              </select>
            </Field>
            {comparison && (
              <p>
                Configuration {comparison.configuration_equal ? 'matches' : 'differs'}. Output curves are
                overlaid in Charts; source timestamps are preserved.
              </p>
            )}
          </div>
        )}
        <div className="ml-table-scroll">
          <table>
            <thead>
              <tr>
                <th>Scenario</th>
                <th>Agent</th>
                <th>Started</th>
                <th>Run state</th>
                <th>Checks</th>
                <th>Processed</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id} className={r.id === runId ? 'ml-current' : ''}>
                  <td>
                    <strong>{r.name}</strong>
                    <small className="ml-id">{r.id.slice(0, 12)}</small>
                  </td>
                  <td>{r.agent === 'pv' ? 'PV detection' : 'Fault location'}</td>
                  <td>{new Date(r.created_at).toLocaleString()}</td>
                  <td>
                    <Badge>{r.state}</Badge>
                  </td>
                  <td>
                    <Badge
                      tone={
                        r.checks.verdict === 'fail'
                          ? 'danger'
                          : r.checks.verdict === 'pass'
                            ? 'success'
                            : 'neutral'
                      }
                    >
                      {r.checks.verdict}
                    </Badge>
                  </td>
                  <td>{num(r.telemetry.processed)}</td>
                  <td>
                    <button
                      className="button subtle"
                      onClick={() => {
                        setRunId(r.id)
                        setAgent(r.agent)
                        setCompareId('')
                      }}
                    >
                      Open
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!runs.length && <Empty title="No runs yet">Your first experiment will be saved here.</Empty>}
      </Panel>
    </div>
  )
}
