import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ArrowRight,
  ChevronRight,
  FlaskConical,
  Pause,
  Play,
  Plus,
  Save,
  Search,
  SkipBack,
  Tag,
  ZoomIn,
  ZoomOut,
} from 'lucide-react'
import type { Data, Layout } from 'plotly.js'
import { api } from '../api'
import { number, percent } from '../format'
import { Badge, Empty, ErrorBox, ExportLink, Field, PageTitle, Panel } from '../components/ui'
import { Chart } from '../components/Chart'
import { Modal } from '../components/Modal'
import { VirtualTable } from '../components/VirtualTable'
import type { Annotation, Dataset, Experiment, Job, WindowData } from '../types'
import ImportDialog from './ImportDialog'
import {
  ComtradeDialog,
  FaultResults,
  FaultSettings,
  WavewinDialog,
  defaultFaultParameters,
} from './FaultTools'
import type { FaultParameters } from './FaultTools'
import { faultNotebookRecord, findFaultNotebookDataset } from '../faultReference'

const kinds: Record<string, string> = {
  review: 'Reviewed interval',
  load_on: 'Load on',
  load_off: 'Load off',
  voltage_dip: 'Voltage dip',
  voltage_rise: 'Voltage rise',
  interruption: 'Interruption',
  fault: 'Fault inception',
}
export default function EventLab() {
  const [params, setParams] = useSearchParams(),
    client = useQueryClient()
  const chosen = params.get('dataset') || ''
  const [creating, setCreating] = useState(false),
    [importing, setImporting] = useState(false),
    [annotating, setAnnotating] = useState(false),
    [error, setError] = useState<unknown>(),
    [queued, setQueued] = useState(''),
    [jobId, setJobId] = useState('')
  const job = useQuery({
    queryKey: ['event-job', jobId],
    queryFn: () => api<Job>('/jobs/' + jobId),
    enabled: !!jobId,
    refetchInterval: 1500,
  })
  useEffect(() => {
    if (job.data?.status === 'completed' && job.data.result) setRunId(job.data.result.id)
  }, [job.data?.status, job.data?.result?.id])
  const [range, setRange] = useState<[number, number]>([0, 10]),
    [cursor, setCursor] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(1),
    [search, setSearch] = useState(''),
    [runId, setRunId] = useState('')
  const [comtradeOpen, setComtradeOpen] = useState(false),
    [wavewinOpen, setWavewinOpen] = useState(false),
    [faultParameters, setFaultParameters] = useState<FaultParameters>(defaultFaultParameters)
  const [algorithm, setAlgorithm] = useState('baseline-events'),
    [customParameters, setCustomParameters] = useState('{}')
  const { data: algorithms = [] } = useQuery({
    queryKey: ['algorithms'],
    queryFn: () => api<{ id: string; name: string; family: string; status: string }[]>('/algorithms'),
  })
  const { data: presets = [] } = useQuery({
    queryKey: ['presets'],
    queryFn: () => api<{ id: string; name: string; kind: string; config: Record<string, any> }[]>('/presets'),
  })
  const [threshold, setThreshold] = useState(1),
    [dip, setDip] = useState(0.9),
    [rise, setRise] = useState(1.1),
    [interruption, setInterruption] = useState(0.1),
    [minimum, setMinimum] = useState(0.02),
    [tolerance, setTolerance] = useState(0.1),
    [partition, setPartition] = useState('evaluation')
  const { data: datasets = [] } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => api<Dataset[]>('/datasets'),
    refetchInterval: 3000,
  })
  const recordings = datasets.filter((d) => d.format === 'recording'),
    notebookDataset = findFaultNotebookDataset(recordings)
  const dataset = recordings.find((d) => d.id === chosen) || notebookDataset || recordings[0]
  const { data: runs = [] } = useQuery({
    queryKey: ['experiments'],
    queryFn: () => api<Experiment[]>('/experiments'),
    refetchInterval: 3000,
  })
  const currentRuns = runs.filter((r) => r.kind === 'events' && r.dataset_ids.includes(dataset?.id || '')),
    result = currentRuns.find((r) => r.id === runId) || currentRuns[0]
  const { data: annotations = [] } = useQuery({
    queryKey: ['annotations', dataset?.id],
    queryFn: () => api<Annotation[]>('/annotations?dataset_id=' + dataset!.id),
    enabled: !!dataset,
  })
  const windowQuery = useQuery({
    queryKey: ['window', dataset?.id, range[0], range[1]],
    queryFn: () => api<WindowData>(`/recordings/${dataset!.id}/window?start=${range[0]}&end=${range[1]}`),
    enabled: !!dataset,
  })
  const end = dataset?.duration_seconds || 0
  useEffect(() => {
    setRange([0, Math.min(10, end)])
    setCursor(0)
    setPlaying(false)
    setRunId('')
    setJobId('')
    setQueued('')
    setFaultParameters(defaultFaultParameters)
    if (dataset?.comtrade) setAlgorithm('fault-distance')
    else setAlgorithm((previous) => (previous === 'fault-distance' ? 'baseline-events' : previous))
  }, [dataset?.id, end])
  function chooseRecording(id: string) {
    setJobId('')
    setQueued('')
    setParams({ dataset: id })
    setRunId('')
    setFaultParameters(defaultFaultParameters)
    if (recordings.find((d) => d.id === id)?.comtrade) setAlgorithm('fault-distance')
  }
  useEffect(() => {
    if (!playing) return
    const timer = setInterval(() => setCursor((t) => Math.min(end, t + speed * 0.2)), 200)
    return () => clearInterval(timer)
  }, [playing, speed, end])
  useEffect(() => {
    if (cursor >= end && playing) setPlaying(false)
    if (playing && cursor > range[1])
      setRange([Math.max(0, cursor - 0.5), Math.min(end, cursor + Math.max(1, range[1] - range[0]))])
  }, [cursor, end, playing, range])
  const traces = useMemo<Data[]>(
    () =>
      (windowQuery.data?.traces.map((t, i) => ({
        type: 'scatter',
        mode: 'lines',
        name: t.channel + ' · ' + t.unit,
        x: t.x,
        y: t.y,
        connectgaps: false,
        yaxis: i ? 'y' + (i + 1) : 'y',
        line: { width: 1.2 },
        hovertemplate: '%{x:.4f}s · %{y:.3f} ' + t.unit + '<extra>' + t.channel + '</extra>',
      })) as Data[]) || [],
    [windowQuery.data],
  )
  const layout = useMemo<Partial<Layout>>(() => {
    const n = traces.length || 1,
      axes: Record<string, unknown> = {}
    for (let i = 0; i < n; i++) {
      axes['yaxis' + (i ? i + 1 : '')] = {
        domain: [1 - (i + 1) / n + 0.055, 1 - i / n - 0.015],
        title: { text: dataset?.channels[i]?.unit },
        gridcolor: '#edf1ee',
        zeroline: false,
      }
    }
    const shapes: Partial<Layout['shapes']> =
      result?.events
        .filter((e) => e.start <= range[1] && e.end >= range[0])
        .map((e) => ({
          type: 'rect',
          xref: 'x',
          yref: 'paper',
          x0: e.start,
          x1: Math.max(e.end, e.start + Math.min(0.03, (range[1] - range[0]) * 0.008)),
          y0: 0,
          y1: 1,
          fillcolor: e.kind.startsWith('load') || e.kind === 'fault' ? '#2fa38b' : '#e4a957',
          opacity: 0.13,
          line: { width: 0 },
        })) || []
    shapes.push({
      type: 'line',
      xref: 'x',
      yref: 'paper',
      x0: cursor,
      x1: cursor,
      y0: 0,
      y1: 1,
      line: { color: '#205b54', width: 1.5, dash: 'dot' },
    })
    return {
      ...axes,
      showlegend: true,
      legend: { orientation: 'h', y: 1.06 },
      margin: { l: 55, r: 15, t: 30, b: 40 },
      xaxis: {
        anchor: 'free',
        position: 0,
        title: { text: 'Seconds from recording start' },
        range: [range[0], range[1]],
        gridcolor: '#edf1ee',
      },
      shapes,
      dragmode: 'zoom',
      uirevision: dataset?.id + '-' + range.join('-'),
    } as Partial<Layout>
  }, [traces.length, dataset, range, cursor, result])
  function zoom(a: number, b: number) {
    setPlaying(false)
    setRange([Math.max(0, a), Math.min(end, Math.max(a + 0.001, b))])
  }
  async function run() {
    setError(undefined)
    try {
      const response = await api<Job>('/events/run', {
        dataset_id: dataset!.id,
        name: dataset!.name + (algorithm === 'fault-distance' ? ' - fault distance' : ' - event benchmark'),
        algorithm,
        parameters: algorithm === 'fault-distance' ? faultParameters : JSON.parse(customParameters),
        step_threshold: threshold,
        dip_ratio: dip,
        rise_ratio: rise,
        interruption_ratio: interruption,
        minimum_duration: minimum,
        match_tolerance: tolerance,
        partition,
      })
      setJobId(response.id)
      setQueued('Detection queued. Results will appear when processing completes.')
      void client.invalidateQueries()
    } catch (e) {
      setError(e)
    }
  }
  const filtered =
    result?.events.filter((e) => (kinds[e.kind] || e.kind).toLowerCase().includes(search.toLowerCase())) || []
  return (
    <>
      <PageTitle
        eyebrow="SUB-SECOND RESEARCH"
        title="Event lab"
        description="Find the transition. Inspect the waveform. Measure the detection."
      >
        <button className="button secondary" onClick={() => setComtradeOpen(true)}>
          COMTRADE recordings
        </button>
        <button className="button secondary" onClick={() => setWavewinOpen(true)}>
          Wavewin event logs
        </button>
        <button className="button secondary" onClick={() => setCreating(true)}>
          <FlaskConical size={16} />
          Synthetic recording
        </button>
        <button className="button" onClick={() => setImporting(true)}>
          <Plus size={16} />
          Import recording
        </button>
      </PageTitle>
      <ErrorBox
        error={
          error ||
          windowQuery.error ||
          (['failed', 'cancelled'].includes(job.data?.status || '') ? job.data?.message : undefined)
        }
      />
      {dataset ? (
        <>
          <div className="event-toolbar">
            <Field label="Recording">
              <select value={dataset.id} onChange={(e) => chooseRecording(e.target.value)}>
                {recordings.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </Field>
            <div className="recording-facts">
              <Badge tone={dataset.synthetic ? 'amber' : 'green'}>
                {dataset.comtrade
                  ? 'RTDS simulation'
                  : dataset.synthetic
                    ? 'Synthetic fixture'
                    : 'Measured data'}
              </Badge>
              <span>{number(dataset.sample_rate)} Hz</span>
              <span>{dataset.channels.length} channels</span>
              <span>{number(dataset.duration_seconds)} s</span>
            </div>
          </div>
          {dataset.comtrade && notebookDataset ? (
            <div className="notebook-storage-note">
              <span>
                The saved fault_distance.ipynb example uses <strong>{faultNotebookRecord}</strong>.
              </span>
              <button className="text-button" onClick={() => chooseRecording(notebookDataset.id)}>
                Use notebook example
              </button>
            </div>
          ) : null}
          <div className="event-layout">
            <div>
              <Panel
                title="Signal inspection"
                subtitle={`${dataset.channels.map((c) => c.kind.replace('_', ' ')).join(' · ')}`}
                actions={
                  <div className="inline-actions">
                    <button
                      className="icon-button"
                      aria-label="Zoom in"
                      onClick={() => zoom(cursor, Math.min(end, cursor + 2))}
                    >
                      <ZoomIn size={17} />
                    </button>
                    <button
                      className="icon-button"
                      aria-label="Show entire recording"
                      onClick={() => zoom(0, end)}
                    >
                      <ZoomOut size={17} />
                    </button>
                    <button className="button secondary small" onClick={() => setAnnotating(true)}>
                      <Tag size={14} />
                      Annotate
                    </button>
                  </div>
                }
              >
                <Chart
                  data={traces}
                  layout={layout}
                  height={510}
                  onRange={zoom}
                  title="Synchronized recording channels"
                />
                <div className="chart-caption">
                  <span>{windowQuery.isFetching ? 'Loading samples…' : windowQuery.data?.aggregation}</span>
                  <span>
                    {dataset.clock_verified === false
                      ? 'Elapsed time / reported clock ' +
                        dataset.comtrade?.reported_start +
                        ' / timezone unverified'
                      : 'Origin: ' + dataset.start}
                  </span>
                </div>
                <div className="replay-controls">
                  <button
                    className="icon-button"
                    aria-label="Restart replay"
                    onClick={() => {
                      setCursor(0)
                      setRange([0, Math.min(10, end)])
                    }}
                  >
                    <SkipBack size={17} />
                  </button>
                  <button
                    className="play-button"
                    aria-label={playing ? 'Pause replay' : 'Start replay'}
                    onClick={() => setPlaying(!playing)}
                  >
                    {playing ? <Pause size={16} /> : <Play size={16} />}
                  </button>
                  <span className="replay-clock">{cursor.toFixed(2)} s</span>
                  <input
                    aria-label="Replay position"
                    type="range"
                    min="0"
                    max={end}
                    step=".01"
                    value={cursor}
                    onChange={(e) => {
                      const t = +e.target.value
                      setCursor(t)
                      if (t < range[0] || t > range[1]) zoom(Math.max(0, t - 1), Math.min(end, t + 5))
                    }}
                  />
                  <select aria-label="Replay speed" value={speed} onChange={(e) => setSpeed(+e.target.value)}>
                    {[0.25, 1, 5, 20].map((v) => (
                      <option value={v} key={v}>
                        {v}×
                      </option>
                    ))}
                  </select>
                </div>
                <div className="window-inputs">
                  <Field label="Window start (s)">
                    <input
                      type="number"
                      min="0"
                      step={1 / dataset.sample_rate}
                      value={Number(range[0].toFixed(9))}
                      onChange={(e) => zoom(+e.target.value, range[1])}
                    />
                  </Field>
                  <Field label="Window end (s)">
                    <input
                      type="number"
                      min={range[0] + 0.001}
                      step={1 / dataset.sample_rate}
                      max={Number(end.toFixed(9))}
                      value={Number(range[1].toFixed(9))}
                      onChange={(e) => zoom(range[0], +e.target.value)}
                    />
                  </Field>
                  <span>Zoom into a few seconds to inspect native samples.</span>
                </div>
              </Panel>
              <Panel
                title="Detected events"
                subtitle="Select an event to inspect its native samples"
                actions={
                  <div className="search">
                    <Search size={15} />
                    <input
                      value={search}
                      placeholder="Filter events…"
                      aria-label="Filter events"
                      onChange={(e) => setSearch(e.target.value)}
                    />
                  </div>
                }
              >
                {result ? (
                  <>
                    <Field label="Detection run">
                      <select value={result.id} onChange={(e) => setRunId(e.target.value)}>
                        {currentRuns.map((r) => (
                          <option key={r.id} value={r.id}>
                            {r.name} · {r.id.slice(0, 6)}
                          </option>
                        ))}
                      </select>
                    </Field>
                    <VirtualTable
                      rows={filtered}
                      onRow={(e) => {
                        setCursor(e.start)
                        zoom(Math.max(0, e.start - 0.5), Math.min(end, e.start + 1.5))
                      }}
                      columns={[
                        { key: 'kind', title: 'Event', render: (e) => kinds[e.kind] },
                        { key: 'start', title: 'Onset', render: (e) => e.start.toFixed(4) + ' s' },
                        {
                          key: 'duration',
                          title: 'Duration',
                          render: (e) => (e.end - e.start).toFixed(4) + ' s',
                        },
                        {
                          key: 'emitted',
                          title: 'Emitted at',
                          render: (e) => e.emitted_at.toFixed(4) + ' s',
                        },
                      ]}
                    />
                  </>
                ) : (
                  <Empty title="No detector results yet">
                    Run the baseline detectors to inspect candidate transitions.
                  </Empty>
                )}
              </Panel>
              {result ? <FaultResults result={result} dataset={dataset} /> : null}
              <Panel
                title="Annotation evidence"
                subtitle={`${annotations.length} annotations · review coverage controls evaluation`}
              >
                <div className="annotation-list">
                  {annotations.slice(0, 20).map((a) => (
                    <div className="annotation-item" key={a.id}>
                      <button
                        onClick={() => {
                          setCursor(a.start)
                          zoom(a.start, Math.min(end, Math.max(a.start + 2, a.end)))
                        }}
                      >
                        <Badge tone={a.kind === 'review' ? 'neutral' : 'green'}>{kinds[a.kind]}</Badge>
                        <span>
                          {a.start.toFixed(2)}–{a.end.toFixed(2)} s
                        </span>
                        <small>{a.partition}</small>
                        <ChevronRight size={14} />
                      </button>
                      <button
                        className="annotation-withdraw"
                        aria-label={'Withdraw ' + kinds[a.kind] + ' annotation'}
                        onClick={async () => {
                          try {
                            await api('/annotations/' + a.id + '/withdraw', {})
                            void client.invalidateQueries({ queryKey: ['annotations'] })
                          } catch (e) {
                            setError(e)
                          }
                        }}
                      >
                        Withdraw
                      </button>
                    </div>
                  ))}
                </div>
              </Panel>
            </div>
            <aside className="event-settings">
              <Panel
                title="Detector settings"
                subtitle={
                  algorithm === 'fault-distance'
                    ? 'Offline waveform analysis - native samples'
                    : 'Causal processing - state retained across chunks'
                }
              >
                <div className="panel-pad">
                  <Field label="Detector adapter">
                    <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
                      <option
                        value="baseline-events"
                        disabled={
                          new Set(dataset.channels.map((c) => c.kind)).size !== dataset.channels.length
                        }
                      >
                        Baseline load + voltage events
                      </option>
                      {algorithms
                        .filter(
                          (a) => a.family === 'Events' && !['load-step', 'voltage-events'].includes(a.id),
                        )
                        .map((a) => (
                          <option
                            key={a.id}
                            value={a.id}
                            disabled={
                              a.id === 'fault-distance' &&
                              (dataset.channels.filter((c) => c.kind === 'current').length < 3 ||
                                dataset.channels.filter((c) => c.kind === 'voltage').length < 3)
                            }
                          >
                            {a.name}
                          </option>
                        ))}
                    </select>
                  </Field>
                  {algorithm !== 'baseline-events' && algorithm !== 'fault-distance' ? (
                    <Field label="Adapter parameters (JSON)">
                      <textarea
                        value={customParameters}
                        onChange={(e) => setCustomParameters(e.target.value)}
                      />
                    </Field>
                  ) : null}
                  <Field label="Saved detector configuration">
                    <select
                      value=""
                      onChange={(e) => {
                        const c = presets.find((p) => p.id === e.target.value)?.config
                        if (c) {
                          setAlgorithm(c.algorithm || 'baseline-events')
                          setCustomParameters(JSON.stringify(c.parameters || {}))
                          setFaultParameters({ ...defaultFaultParameters, ...c.parameters })
                          setThreshold(c.step_threshold)
                          setDip(c.dip_ratio)
                          setRise(c.rise_ratio)
                          setInterruption(c.interruption_ratio)
                          setMinimum(c.minimum_duration)
                          setTolerance(c.match_tolerance)
                        }
                      }}
                    >
                      <option value="">Load preset?</option>
                      {presets
                        .filter((p) => p.kind === 'events')
                        .map((p) => (
                          <option value={p.id} key={p.id}>
                            {p.name}
                          </option>
                        ))}
                    </select>
                  </Field>
                  {algorithm === 'fault-distance' ? (
                    <FaultSettings dataset={dataset} value={faultParameters} onChange={setFaultParameters} />
                  ) : (
                    <>
                      <Field label="Load step threshold" hint="kW when power is available; otherwise A RMS.">
                        <input
                          type="number"
                          min=".01"
                          step=".1"
                          value={threshold}
                          onChange={(e) => setThreshold(+e.target.value)}
                        />
                      </Field>
                      <div className="form-grid">
                        <Field label="Voltage dip ratio">
                          <input
                            type="number"
                            min=".01"
                            max=".99"
                            step=".01"
                            value={dip}
                            onChange={(e) => setDip(+e.target.value)}
                          />
                        </Field>
                        <Field label="Voltage rise ratio">
                          <input
                            type="number"
                            min="1.01"
                            step=".01"
                            value={rise}
                            onChange={(e) => setRise(+e.target.value)}
                          />
                        </Field>
                      </div>
                      <Field label="Interruption ratio">
                        <input
                          type="number"
                          min=".001"
                          max=".99"
                          step=".01"
                          value={interruption}
                          onChange={(e) => setInterruption(+e.target.value)}
                        />
                      </Field>
                      <Field label="Minimum duration (s)">
                        <input
                          type="number"
                          min=".001"
                          step=".001"
                          value={minimum}
                          onChange={(e) => setMinimum(+e.target.value)}
                        />
                      </Field>
                    </>
                  )}
                  <Field label="Matching tolerance (s)">
                    <input
                      type="number"
                      min=".001"
                      step=".01"
                      value={tolerance}
                      onChange={(e) => setTolerance(+e.target.value)}
                    />
                  </Field>
                  <Field label="Annotation partition">
                    <select value={partition} onChange={(e) => setPartition(e.target.value)}>
                      <option value="evaluation">Evaluation</option>
                      <option value="development">Development</option>
                    </select>
                  </Field>
                  <button className="button full" onClick={run}>
                    <Play size={15} />
                    Run detectors
                  </button>
                  <button
                    className="button secondary full"
                    style={{ marginTop: 8 }}
                    onClick={async () => {
                      try {
                        await api('/presets', {
                          kind: 'events',
                          name: dataset.name + ' ? detector settings',
                          config: {
                            dataset_id: dataset.id,
                            algorithm,
                            parameters:
                              algorithm === 'fault-distance' ? faultParameters : JSON.parse(customParameters),
                            step_threshold: threshold,
                            dip_ratio: dip,
                            rise_ratio: rise,
                            interruption_ratio: interruption,
                            minimum_duration: minimum,
                            match_tolerance: tolerance,
                            partition,
                          },
                        })
                        setJobId('')
                        setQueued('Detector configuration saved.')
                        void client.invalidateQueries({ queryKey: ['presets'] })
                      } catch (e) {
                        setError(e)
                      }
                    }}
                  >
                    <Save size={14} />
                    Save configuration
                  </button>
                  {job.data ? (
                    <div role="status">
                      <p className={job.data.status === 'completed' ? 'success' : 'muted'}>
                        {job.data.status === 'completed'
                          ? 'Detection completed. Results are ready.'
                          : job.data.message}
                      </p>
                      {['running', 'queued'].includes(job.data.status) ? (
                        <progress value={job.data.progress} max={1} />
                      ) : null}
                    </div>
                  ) : queued ? (
                    <p className="success">{queued}</p>
                  ) : null}
                </div>
              </Panel>
              {result ? (
                <Panel title="Event evaluation" subtitle={result.interpretation}>
                  <div className="panel-pad">
                    <div className="event-metrics">
                      <div>
                        <span>Precision</span>
                        <strong>{percent(result.metrics.precision)}</strong>
                      </div>
                      <div>
                        <span>Recall</span>
                        <strong>{percent(result.metrics.recall)}</strong>
                      </div>
                      <div>
                        <span>F1</span>
                        <strong>{percent(result.metrics.f1)}</strong>
                      </div>
                      <div>
                        <span>False alarms / h</span>
                        <strong>{result.metrics.false_alarms_per_hour?.toFixed(2) ?? '—'}</strong>
                      </div>
                      <div>
                        <span>Onset error</span>
                        <strong>
                          {result.metrics.onset_mae === null
                            ? '—'
                            : ((result.metrics.onset_mae || 0) * 1000).toFixed(1) + ' ms'}
                        </strong>
                      </div>
                      <div>
                        <span>Mean latency</span>
                        <strong>
                          {result.metrics.mean_latency === null
                            ? '—'
                            : ((result.metrics.mean_latency || 0) * 1000).toFixed(1) + ' ms'}
                        </strong>
                      </div>
                    </div>
                    <p className="muted">
                      {((result.metrics.reviewed_hours || 0) * 3600).toFixed(1)} reviewed seconds. Unreviewed
                      detections are excluded.
                    </p>
                    <ExportLink id={result.id} />
                  </div>
                </Panel>
              ) : null}
              <div className="principle-card">
                <span className="eyebrow">NATIVE RESOLUTION</span>
                <h3>The plot is a view. The samples are the evidence.</h3>
                <p>
                  Overview envelopes preserve extrema. Detectors process original samples, with explicit
                  channel derivations where needed.
                </p>
              </div>
            </aside>
          </div>
        </>
      ) : (
        <Panel
          title="Your signal investigation starts here"
          subtitle="1 kHz recordings · synchronized channels · annotated evidence"
        >
          <Empty
            title="Explore a recording at sample resolution"
            action={
              <button className="button" onClick={() => setCreating(true)}>
                <FlaskConical size={16} />
                Create synthetic recording
              </button>
            }
          >
            Start with a reproducible fixture containing load steps, voltage dips, rises, interruptions,
            noise, and a data gap.
          </Empty>
        </Panel>
      )}
      {comtradeOpen ? (
        <ComtradeDialog onClose={() => setComtradeOpen(false)} onOpen={chooseRecording} />
      ) : null}
      {wavewinOpen ? <WavewinDialog onClose={() => setWavewinOpen(false)} onOpen={chooseRecording} /> : null}
      {creating ? <SyntheticDialog onClose={() => setCreating(false)} /> : null}
      {importing ? <ImportDialog onClose={() => setImporting(false)} /> : null}
      {annotating && dataset ? (
        <AnnotationDialog
          dataset={dataset}
          start={cursor}
          end={Math.min(end, Math.max(cursor, range[1]))}
          onClose={() => setAnnotating(false)}
        />
      ) : null}
    </>
  )
}
function SyntheticDialog({ onClose }: { onClose: () => void }) {
  const [duration, setDuration] = useState(120),
    [seed, setSeed] = useState(42),
    [error, setError] = useState<unknown>(),
    [busy, setBusy] = useState(false),
    client = useQueryClient()
  return (
    <Modal title="Create a synthetic recording" onClose={onClose}>
      <div className="notice amber-notice">
        Synthetic fixtures test software and detector behavior. They do not establish real-world detection
        performance.
      </div>
      <div className="form-grid">
        <Field label="Duration">
          <select value={duration} onChange={(e) => setDuration(+e.target.value)}>
            <option value="120">2 minutes · quick investigation</option>
            <option value="3600">1 hour · longer replay</option>
            <option value="86400">24 hours · storage benchmark</option>
          </select>
        </Field>
        <Field label="Random seed">
          <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} />
        </Field>
      </div>
      <p className="muted">
        1,000 samples/second · voltage, current, power, voltage RMS. Ground-truth events and reviewed
        intervals are included.
      </p>
      <ErrorBox error={error} />
      <div className="form-actions">
        <button
          className="button"
          disabled={busy}
          onClick={async () => {
            setBusy(true)
            try {
              await api('/synthetic', {
                duration_seconds: duration,
                sample_rate: 1000,
                seed,
                name: duration === 86400 ? '24-hour storage benchmark' : 'Switching & voltage disturbances',
              })
              void client.invalidateQueries()
              onClose()
            } catch (e) {
              setError(e)
              setBusy(false)
            }
          }}
        >
          {busy ? 'Queueing…' : 'Generate recording'}
          <ArrowRight size={15} />
        </button>
      </div>
    </Modal>
  )
}
function AnnotationDialog({
  dataset,
  start,
  end,
  onClose,
}: {
  dataset: Dataset
  start: number
  end: number
  onClose: () => void
}) {
  const [kind, setKind] = useState('review'),
    [a, setA] = useState(start),
    [b, setB] = useState(end),
    [note, setNote] = useState(''),
    [partition, setPartition] = useState('evaluation'),
    [error, setError] = useState<unknown>(),
    client = useQueryClient()
  return (
    <Modal title="Annotate recording evidence" onClose={onClose}>
      <form
        onSubmit={async (e) => {
          e.preventDefault()
          try {
            await api('/annotations', {
              dataset_id: dataset.id,
              asset_id: dataset.asset_id,
              kind,
              start: a,
              end: kind.startsWith('load') || kind === 'fault' ? a : b,
              note,
              partition,
            })
            void client.invalidateQueries({ queryKey: ['annotations'] })
            onClose()
          } catch (e) {
            setError(e)
          }
        }}
      >
        <Field label="Annotation type">
          <select value={kind} onChange={(e) => setKind(e.target.value)}>
            {Object.entries(kinds).map(([key, name]) => (
              <option key={key} value={key}>
                {name}
              </option>
            ))}
          </select>
        </Field>
        <div className="form-grid">
          <Field label="Start (seconds)">
            <input
              type="number"
              min="0"
              step=".001"
              required
              value={a}
              onChange={(e) => setA(+e.target.value)}
            />
          </Field>
          <Field label="End (seconds)">
            <input
              type="number"
              min={a}
              step=".001"
              required
              disabled={kind.startsWith('load') || kind === 'fault'}
              value={kind.startsWith('load') || kind === 'fault' ? a : b}
              onChange={(e) => setB(+e.target.value)}
            />
          </Field>
        </div>
        <Field label="Partition">
          <select value={partition} onChange={(e) => setPartition(e.target.value)}>
            <option value="evaluation">Evaluation</option>
            <option value="development">Development</option>
          </select>
        </Field>
        <Field label="Evidence note">
          <textarea value={note} onChange={(e) => setNote(e.target.value)} />
        </Field>
        <p className="muted">
          Keep all recordings from this asset in one partition. Mark reviewed coverage separately from
          individual event labels.
        </p>
        <ErrorBox error={error} />
        <div className="form-actions">
          <button className="button" type="submit">
            Save annotation
          </button>
        </div>
      </form>
    </Modal>
  )
}
