import { useMemo, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { BookOpen, Download, Play, Save, Settings2, TrendingUp } from 'lucide-react'
import type { Data } from 'plotly.js'
import { api } from '../api'
import { number } from '../format'
import { useStored } from '../hooks/useStored'
import { Badge, Empty, ErrorBox, Field, PageTitle, Panel } from '../components/ui'
import { Chart } from '../components/Chart'
import { Modal } from '../components/Modal'
import { VirtualTable } from '../components/VirtualTable'
import type { Asset, Dataset, Job } from '../types'

type Provider = {
  id: string
  name: string
  mode: string
  group: string
  ready: boolean
  version: string
  reason: string
  defaults: Record<string, string | number>
}
type Source = { name: string; format: string; registered: boolean }
type Config = {
  name: string
  dataset_id?: string
  asset_id: string
  channel: string
  source_file?: string
  timestamp_column: string
  unit: string
  start?: string
  end?: string
  horizon: number
  validation_windows: number
  context_length: number
  max_history: number
  season_length: number
  lags: number
  models: string[]
  parameters: Record<string, Record<string, string | number>>
  missing_policy: string
  max_gap_steps: number
  min_coverage: number
  seed: number
  timeout_seconds: number
  reuse_run_id?: string
  reuse_model?: string
}
type Stats = { mae: number; rmse: number; smape: number; wape: number | null; mase: number | null; n: number }
type Model = {
  model: string
  name: string
  status: string
  selected?: boolean
  error?: string
  validation?: Stats
  holdout?: Stats
  interval_method?: string
}
type Run = {
  id: string
  name: string
  created_at: string
  mode: string
  config: Config
  source: {
    name: string
    channel: string
    unit: string
    timezone: string
    resolution_seconds: number
    missing: number
    synthetic: boolean | null
  }
  selected_model: string
  models: Model[]
  population_id: string
  selection_policy: string
  preprocessing?: {
    phase: string
    window: number
    filled: number
    gaps_filled: number
    observed_coverage: number
  }[]
}
type Prediction = {
  timestamp: string
  actual: number | null
  prediction: number
  lower: number | null
  upper: number | null
  phase: string
  window: number
}
const initial: Config = {
  name: 'Forecast benchmark',
  asset_id: '',
  channel: '',
  timestamp_column: 'Timestamp',
  unit: 'unverified',
  horizon: 24,
  validation_windows: 3,
  context_length: 2048,
  max_history: 10000,
  season_length: 24,
  lags: 24,
  models: ['seasonal_naive', 'random_forest', 'xgboost', 'lightgbm', 'ensemble'],
  parameters: {},
  missing_policy: 'reject',
  max_gap_steps: 3,
  min_coverage: 0.9,
  seed: 42,
  timeout_seconds: 600,
}
const metric = (n: number | null | undefined) =>
  n == null ? 'Unavailable' : n.toLocaleString(undefined, { maximumFractionDigits: 4 })

export default function Forecasting() {
  const client = useQueryClient()
  const [tab, setTab] = useState('configure'),
    [chosen, setChosen] = useStored('forecast-source', ''),
    [form, setForm] = useStored<Config>('forecast-form-v1', initial)
  const [jobId, setJobId] = useStored('forecast-job', ''),
    [selected, setSelected] = useState<Run>(),
    [settings, setSettings] = useState<Provider>(),
    [error, setError] = useState<unknown>(),
    [busy, setBusy] = useState(false)
  const { data: models = [] } = useQuery({
    queryKey: ['forecast-models'],
    queryFn: () => api<Provider[]>('/forecasting/models'),
    refetchInterval: 5000,
  })
  const { data: datasets = [] } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => api<Dataset[]>('/datasets'),
  })
  const { data: sources = [] } = useQuery({ queryKey: ['sources'], queryFn: () => api<Source[]>('/sources') })
  const { data: runs = [] } = useQuery({
    queryKey: ['forecast-runs'],
    queryFn: () => api<Run[]>('/forecasting/runs'),
    refetchInterval: 3000,
  })
  const { data: presets = [] } = useQuery({
    queryKey: ['presets'],
    queryFn: () => api<{ id: string; name: string; kind: string; config: Config }[]>('/presets'),
  })
  const { data: job } = useQuery({
    queryKey: ['forecast-job', jobId],
    queryFn: () => api<Job>('/jobs/' + jobId),
    enabled: !!jobId,
    refetchInterval: 1500,
  })
  const raw = chosen.startsWith('source:') ? chosen.slice(7) : undefined
  const dataset = raw ? undefined : datasets.find((d) => d.id === chosen) || datasets[0]
  const { data: assets = [] } = useQuery({
    queryKey: ['forecast-assets', dataset?.id],
    queryFn: async () => {
      const [assets, quality] = await Promise.all([
        api<Asset[]>('/assets?dataset_id=' + dataset!.id + '&limit=10000'),
        api<{ assets: { asset_id: string; coverage: number }[] }>('/datasets/' + dataset!.id + '/quality'),
      ])
      const coverage = new Map(quality.assets.map((a) => [a.asset_id, a.coverage]))
      return assets.sort((a, b) => (coverage.get(b.id) || 0) - (coverage.get(a.id) || 0))
    },
    enabled: !!dataset,
  })
  const { data: columns } = useQuery({
    queryKey: ['forecast-columns', raw],
    queryFn: () =>
      api<{ columns: string[]; numeric: string[] }>(
        '/forecasting/source-columns?name=' + encodeURIComponent(raw!),
      ),
    enabled: !!raw,
  })
  const channels = raw ? columns?.numeric || [] : dataset?.channels.map((c) => c.name) || []
  const channel = channels.includes(form.channel)
    ? form.channel
    : raw && channels.includes('Load_kW')
      ? 'Load_kW'
      : channels[0] || ''
  const asset = assets.find((a) => a.id === form.asset_id)?.id || assets[0]?.id || ''
  const active = job && ['queued', 'running'].includes(job.status)
  function change<K extends keyof Config>(key: K, value: Config[K]) {
    setForm({ ...form, [key]: value })
  }
  function configuration(): Config {
    return {
      ...form,
      dataset_id: dataset?.id,
      source_file: raw,
      channel,
      asset_id: raw ? '' : asset,
      start: form.start || undefined,
      end: form.end || undefined,
      parameters: Object.fromEntries(
        Object.entries(form.parameters).filter(([k]) => form.models.includes(k)),
      ),
    }
  }
  function load(config: Config) {
    setForm(config)
    setChosen(config.source_file ? 'source:' + config.source_file : config.dataset_id || '')
    setTab('configure')
    setSelected(undefined)
  }
  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setJobId('')
    setError(undefined)
    try {
      const result = await api<Job>('/forecasting/runs', configuration())
      setJobId(result.id)
      void client.invalidateQueries({ queryKey: ['jobs'] })
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }
  async function setup(model: Provider) {
    setError(undefined)
    try {
      const result = await api<Job>('/forecasting/setup', { model: model.id })
      setJobId(result.id)
    } catch (e) {
      setError(e)
    }
  }
  return (
    <>
      <PageTitle
        eyebrow="TIME-SERIES FORECASTING"
        title="Forecasting workbench"
        description="Build models, test future windows, and inspect every forecast against its source."
      >
        <a
          className="button secondary"
          href="/notebooks/lab/index.html?path=04%20-%20Forecasting.ipynb"
          target="_blank"
          rel="noreferrer"
        >
          <BookOpen size={16} />
          Notebook example
        </a>
      </PageTitle>
      <div className="forecast-tabs" role="tablist" aria-label="Forecasting workspace">
        {[
          ['configure', 'Configure'],
          ['runs', 'Saved runs'],
          ['models', 'Model library'],
        ].map(([key, name]) => (
          <button
            key={key}
            role="tab"
            aria-selected={tab === key}
            className={tab === key ? 'active' : ''}
            onClick={() => setTab(key)}
          >
            {name}
          </button>
        ))}
      </div>
      <ErrorBox error={error} />
      {job ? (
        <div className="forecast-job" role="status">
          <div>
            <strong>
              {job.kind === 'forecast_setup' ? 'Model setup' : job.result?.name || 'Forecast job'}
            </strong>
            <Badge tone={job.status === 'failed' ? 'red' : job.status === 'completed' ? 'green' : 'neutral'}>
              {job.status}
            </Badge>
          </div>
          <p>{job.message}</p>
          {active ? (
            <>
              <progress max={1} value={job.progress} />
              <button
                className="text-button"
                onClick={async () => {
                  await api('/jobs/' + job.id + '/cancel', {})
                  void client.invalidateQueries({ queryKey: ['forecast-job'] })
                }}
              >
                Cancel job
              </button>
            </>
          ) : job.status === 'completed' && job.kind === 'forecast' ? (
            <button
              className="button secondary small"
              onClick={async () => setSelected(await api<Run>('/forecasting/runs/' + job.result!.id))}
            >
              Open completed forecast
            </button>
          ) : null}
          {job.status === 'failed' ? (
            <details>
              <summary>Failure details</summary>
              <pre className="code">{job.error}</pre>
            </details>
          ) : null}
        </div>
      ) : null}
      {tab === 'configure' ? (
        <form onSubmit={submit} className="forecast-layout">
          <Panel title="Series and forecast window" subtitle="One asset and one measurement per experiment">
            <div className="forecast-form">
              <Field label="Forecast name">
                <input value={form.name} onChange={(e) => change('name', e.target.value)} required />
              </Field>
              <Field label="Forecast data source">
                <select
                  value={raw ? 'source:' + raw : dataset?.id || ''}
                  onChange={(e) => {
                    setChosen(e.target.value)
                    setForm({ ...form, asset_id: '', channel: '', start: '', end: '' })
                  }}
                >
                  <optgroup label="Indexed datasets">
                    {datasets.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name}
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label="Unmapped source tables">
                    {sources
                      .filter((s) => s.format === 'mapping_required')
                      .map((s) => (
                        <option key={s.name} value={'source:' + s.name}>
                          {s.name}
                        </option>
                      ))}
                  </optgroup>
                </select>
              </Field>
              {raw ? (
                <div className="notice">
                  Reported source clock; timezone and provenance are unverified. Labels are excluded from
                  forecast targets.
                </div>
              ) : (
                <Field label="Forecast asset">
                  <select value={asset} onChange={(e) => change('asset_id', e.target.value)}>
                    {assets.map((a) => (
                      <option key={a.id} value={a.id}>
                        {a.source_id}
                      </option>
                    ))}
                  </select>
                </Field>
              )}
              <Field label="Forecast measurement">
                <select value={channel} onChange={(e) => change('channel', e.target.value)}>
                  {channels.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </Field>
              {raw ? (
                <div className="form-grid">
                  <Field label="Timestamp column">
                    <select
                      value={form.timestamp_column}
                      onChange={(e) => change('timestamp_column', e.target.value)}
                    >
                      {columns?.columns.map((c) => (
                        <option key={c}>{c}</option>
                      ))}
                    </select>
                  </Field>
                  <Field label="Measurement unit" hint="Keep unverified unless the source declares it.">
                    <input value={form.unit} onChange={(e) => change('unit', e.target.value)} />
                  </Field>
                </div>
              ) : (
                <p className="muted">
                  {dataset?.channels.find((c) => c.name === channel)?.unit} ·{' '}
                  {dataset?.format === 'recording'
                    ? `${dataset.sample_rate} Hz`
                    : `${dataset?.interval_seconds} seconds/interval`}{' '}
                  · {dataset?.timezone}
                </p>
              )}
              <div className="form-grid">
                {(['start', 'end'] as const).map((key) => (
                  <Field
                    key={key}
                    label={key === 'start' ? 'Start timestamp (optional)' : 'End timestamp (optional)'}
                  >
                    <input
                      value={form[key] || ''}
                      placeholder="ISO timestamp"
                      onChange={(e) => change(key, e.target.value)}
                    />
                  </Field>
                ))}
              </div>
              <div className="form-grid">
                {(
                  [
                    ['horizon', 'Forecast steps', 1, 512],
                    ['validation_windows', 'Validation windows', 2, 8],
                    ['context_length', 'Training context (steps)', 32, 200000],
                    ['max_history', 'Maximum history (steps)', 100, 200000],
                    ['season_length', 'Season length (steps)', 1, 10000],
                    ['lags', 'Lag features', 1, 1024],
                  ] as const
                ).map(([key, label, min, max]) => (
                  <Field key={key} label={label}>
                    <input
                      type="number"
                      min={min}
                      max={max}
                      required
                      value={form[key]}
                      onChange={(e) => change(key, +e.target.value)}
                    />
                  </Field>
                ))}
              </div>
              <div className="forecast-interpolation">
                <label>
                  <input
                    type="checkbox"
                    checked={form.missing_policy === 'interpolate'}
                    onChange={(e) => change('missing_policy', e.target.checked ? 'interpolate' : 'reject')}
                    aria-describedby="interpolation-help"
                  />
                  Interpolate missing data
                </label>
                <p id="interpolation-help" className="muted">
                  Fill short gaps using linear interpolation between observed values inside each training
                  window. Leading, trailing, or longer gaps cannot be filled. Evaluation targets remain
                  unchanged.
                </p>
              </div>
              {form.missing_policy !== 'interpolate' ? (
                <Field label="Missing data">
                  <select
                    value={form.missing_policy}
                    onChange={(e) => change('missing_policy', e.target.value)}
                  >
                    <option value="reject">Reject gaps and missing readings</option>
                    <option value="forward_fill">Bounded forward fill - training history only</option>
                  </select>
                </Field>
              ) : null}
              {form.missing_policy !== 'reject' ? (
                <Field
                  label={
                    form.missing_policy === 'interpolate'
                      ? 'Maximum interpolated gap (steps)'
                      : 'Maximum filled gap (steps)'
                  }
                  hint="Maximum consecutive missing samples. Observed coverage must still meet the configured minimum."
                >
                  <input
                    type="number"
                    min="1"
                    max="100"
                    required
                    value={form.max_gap_steps}
                    onChange={(e) => change('max_gap_steps', +e.target.value)}
                  />
                </Field>
              ) : null}
              <details>
                <summary>Reproducibility and runtime limits</summary>
                <div className="form-grid">
                  <Field label="Random seed">
                    <input
                      type="number"
                      value={form.seed}
                      onChange={(e) => change('seed', +e.target.value)}
                    />
                  </Field>
                  <Field label="Minimum coverage">
                    <input
                      type="number"
                      min="0.5"
                      max="1"
                      step=".01"
                      value={form.min_coverage}
                      onChange={(e) => change('min_coverage', +e.target.value)}
                    />
                  </Field>
                  <Field label="Neural process timeout (seconds)">
                    <input
                      type="number"
                      min="10"
                      max="86400"
                      value={form.timeout_seconds}
                      onChange={(e) => change('timeout_seconds', +e.target.value)}
                    />
                  </Field>
                </div>
              </details>
            </div>
          </Panel>
          <div className="forecast-main">
            <Panel
              title={form.reuse_run_id ? 'Use a saved fitted model' : 'Models to compare'}
              subtitle="Parameters and model versions are saved with each run"
            >
              <div className="forecast-form">
                {form.reuse_run_id ? (
                  <div className="notice">
                    Using {form.reuse_model} from run {form.reuse_run_id.slice(0, 8)}. New observations must
                    match its asset, channel, units, and resolution.
                    <button
                      type="button"
                      className="text-button"
                      onClick={() =>
                        setForm({
                          ...form,
                          reuse_run_id: undefined,
                          reuse_model: undefined,
                          models: initial.models,
                        })
                      }
                    >
                      Train a new comparison
                    </button>
                  </div>
                ) : null}
                <div className="forecast-model-grid">
                  {models.map((model) => (
                    <div
                      className={'forecast-model ' + (form.models.includes(model.id) ? 'selected' : '')}
                      key={model.id}
                    >
                      <label>
                        <input
                          type="checkbox"
                          checked={form.models.includes(model.id)}
                          disabled={!model.ready || !!form.reuse_run_id}
                          onChange={(e) =>
                            change(
                              'models',
                              e.target.checked
                                ? [...form.models, model.id]
                                : form.models.filter((m) => m !== model.id),
                            )
                          }
                        />
                        <strong>{model.name}</strong>
                      </label>
                      <div>
                        <span>{model.mode}</span>
                        <Badge tone={model.ready ? 'green' : 'amber'}>
                          {model.ready ? 'Runtime ready' : 'Setup needed'}
                        </Badge>
                      </div>
                      <div className="forecast-model-actions">
                        {Object.keys(model.defaults).length ? (
                          <button type="button" className="text-button" onClick={() => setSettings(model)}>
                            <Settings2 size={13} />
                            Parameters
                          </button>
                        ) : null}
                        {!model.ready ? (
                          <button type="button" className="text-button" onClick={() => setup(model)}>
                            Install runtime
                          </button>
                        ) : null}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="notice">
                  Validation windows move forward in time. The final window is held out from model selection.
                  Future forecasts are refitted afterward. Ensemble members use equal, fixed weights.
                </div>
                <div className="form-actions">
                  <select
                    aria-label="Forecast preset"
                    value=""
                    onChange={(e) => {
                      const p = presets.find((p) => p.id === e.target.value)
                      if (p) load(p.config)
                    }}
                  >
                    <option value="">Load preset</option>
                    {presets
                      .filter((p) => p.kind === 'forecast')
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
                        await api('/presets', { name: form.name, kind: 'forecast', config: configuration() })
                        void client.invalidateQueries({ queryKey: ['presets'] })
                      } catch (e) {
                        setError(e)
                      }
                    }}
                  >
                    <Save size={15} />
                    Save preset
                  </button>
                  <button
                    className="button"
                    disabled={busy || !!active || !channel || !form.models.length}
                    type="submit"
                  >
                    <Play size={15} />
                    {form.reuse_run_id ? 'Predict with saved model' : 'Run forecast benchmark'}
                  </button>
                </div>
              </div>
            </Panel>
            <div className="forecast-note">
              <TrendingUp size={21} />
              <div>
                <h3>Forecasts remain connected to their evidence.</h3>
                <p>
                  Review MAE, RMSE, sMAPE, WAPE, and MASE with the held-out observations. Export predictions,
                  fitted models, splits, configuration, and an HTML report.
                </p>
              </div>
            </div>
          </div>
        </form>
      ) : tab === 'runs' ? (
        <Panel
          title="Saved forecasting runs"
          subtitle="Open a run to inspect its forecasts and reuse a fitted model"
        >
          {runs.length ? (
            <VirtualTable
              rows={runs}
              onRow={setSelected}
              columns={[
                { key: 'name', title: 'Run', width: 'minmax(230px,2fr)', render: (r) => r.name },
                {
                  key: 'source',
                  title: 'Measurement',
                  width: 'minmax(190px,1fr)',
                  render: (r) => r.source.channel,
                },
                { key: 'model', title: 'Selected model', render: (r) => r.selected_model },
                { key: 'mode', title: 'Mode', render: (r) => r.mode },
                {
                  key: 'mae',
                  title: 'Holdout MAE',
                  render: (r) => metric(r.models.find((m) => m.selected)?.holdout?.mae),
                },
              ]}
            />
          ) : (
            <Empty title="Create your first forecasting run">
              Configure a series and compare models over future windows.
            </Empty>
          )}
        </Panel>
      ) : (
        <div className="algorithm-grid">
          {models.map((m) => (
            <article className="algorithm-card" key={m.id}>
              <div className="algorithm-top">
                <TrendingUp size={22} />
                <Badge tone={m.ready ? 'green' : 'amber'}>{m.ready ? 'Runtime ready' : 'Setup needed'}</Badge>
              </div>
              <span className="eyebrow">{m.mode}</span>
              <h2>{m.name}</h2>
              <p className="muted">
                {m.version ? 'Package version ' + m.version : 'Included adapter · optional runtime'}
              </p>
              {m.reason ? <p>{m.reason}</p> : null}
              <p>Regularly sampled, numeric measurements. Uses local computation.</p>
              {!m.ready ? (
                <button className="button secondary" onClick={() => setup(m)}>
                  Install runtime
                </button>
              ) : null}
            </article>
          ))}
        </div>
      )}
      {settings ? (
        <Modal title={settings.name + ' parameters'} onClose={() => setSettings(undefined)}>
          <div className="form-grid">
            {Object.entries({ ...settings.defaults, ...form.parameters[settings.id] }).map(([key, value]) => (
              <Field key={key} label={key}>
                <input
                  type={typeof value === 'number' ? 'number' : 'text'}
                  step="any"
                  value={value}
                  onChange={(e) =>
                    change('parameters', {
                      ...form.parameters,
                      [settings.id]: {
                        ...form.parameters[settings.id],
                        [key]: typeof value === 'number' ? +e.target.value : e.target.value,
                      },
                    })
                  }
                />
              </Field>
            ))}
          </div>
          <p className="muted">
            All supported models train locally. AutoGluon is restricted to tree models; no Hugging Face models
            or checkpoints are used.
          </p>
        </Modal>
      ) : null}
      {selected ? (
        <ForecastResult
          run={selected}
          onClose={() => setSelected(undefined)}
          onReuse={(model) =>
            load({
              ...selected.config,
              name: selected.name + ' · saved inference',
              reuse_run_id: selected.id,
              reuse_model: model,
              models: [model],
            })
          }
        />
      ) : null}
    </>
  )
}

function ForecastResult({
  run,
  onClose,
  onReuse,
}: {
  run: Run
  onClose: () => void
  onReuse: (model: string) => void
}) {
  const [model, setModel] = useState(run.selected_model),
    [phase, setPhase] = useState(run.mode === 'inference' ? 'future' : 'holdout')
  const { data: rows = [] } = useQuery({
    queryKey: ['forecast-predictions', run.id, model, phase],
    queryFn: () => api<Prediction[]>(`/forecasting/runs/${run.id}/predictions?model=${model}&phase=${phase}`),
  })
  const data = useMemo<Data[]>(() => {
    const x = rows.map((r) => r.timestamp),
      traces: Data[] = []
    if (rows.some((r) => r.lower !== null)) {
      traces.push({
        x,
        y: rows.map((r) => r.lower),
        type: 'scatter',
        mode: 'lines',
        line: { width: 0 },
        showlegend: false,
        hoverinfo: 'skip',
      })
      traces.push({
        x,
        y: rows.map((r) => r.upper),
        type: 'scatter',
        mode: 'lines',
        line: { width: 0 },
        fill: 'tonexty',
        fillcolor: 'rgba(8,125,113,.12)',
        name: 'Validation residual band',
      })
    }
    traces.push({
      x,
      y: rows.map((r) => r.prediction),
      type: 'scatter',
      mode: 'lines',
      name: 'Forecast',
      line: { color: '#087d71', width: 2 },
    })
    if (phase !== 'future')
      traces.push({
        x,
        y: rows.map((r) => r.actual),
        type: 'scatter',
        mode: 'lines',
        name: 'Observed',
        line: { color: '#db8a38', width: 2 },
      })
    return traces
  }, [rows, phase])
  const layout = useMemo(
    () => ({
      yaxis: { title: { text: run.source.unit } },
      xaxis: {
        title: {
          text:
            run.source.timezone === 'unverified'
              ? 'Reported source clock · timezone unverified'
              : 'Timestamp · ' + run.source.timezone,
        },
      },
    }),
    [run],
  )
  return (
    <Modal title={run.name} onClose={onClose} wide>
      <div className="forecast-result-top">
        <Badge tone={run.source.synthetic === true ? 'amber' : 'neutral'}>
          {run.source.synthetic === true
            ? 'Synthetic'
            : run.source.synthetic === null
              ? 'Source provenance unverified'
              : 'Measured'}
        </Badge>
        <a className="button secondary" href={`/api/v1/forecasting/runs/${run.id}/export`}>
          <Download size={15} />
          Export forecast bundle
        </a>
      </div>
      <p>{run.selection_policy}</p>
      <div className="form-grid">
        <Field label="Forecast result model">
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {run.models
              .filter((m) => m.status === 'completed')
              .map((m) => (
                <option key={m.model} value={m.model}>
                  {m.name}
                  {m.selected ? ' · selected' : ''}
                </option>
              ))}
          </select>
        </Field>
        <Field label="Forecast view">
          <select value={phase} onChange={(e) => setPhase(e.target.value)}>
            {run.mode === 'backtest' ? (
              <>
                <option value="holdout">Final holdout</option>
                <option value="validation">Validation windows</option>
              </>
            ) : null}
            <option value="future">Future forecast</option>
          </select>
        </Field>
      </div>
      <Chart data={data} layout={layout} title="Forecast and observed measurements" height={350} />
      <p className="muted">
        {run.source.resolution_seconds} seconds/step · {number(run.source.missing)} missing source
        observations · bands use validation residuals and do not guarantee coverage.
      </p>
      {run.config.missing_policy === 'interpolate' ? (
        <div className="notice" aria-label="Interpolation audit">
          <strong>Interpolation applied to training history</strong>
          <p>Estimated values are excluded from observed-target metrics. Original readings are preserved.</p>
          <ul>
            {run.preprocessing?.map((p) => (
              <li key={p.phase + p.window}>
                {p.phase === 'validation'
                  ? `Validation ${p.window + 1}`
                  : p.phase === 'holdout'
                    ? 'Holdout training'
                    : p.phase === 'future'
                      ? 'Future forecast training'
                      : 'Inference history'}
                : {p.filled} values in {p.gaps_filled} gaps interpolated;{' '}
                {(100 * p.observed_coverage).toFixed(1)}% originally observed.
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <VirtualTable
        rows={run.models}
        columns={[
          {
            key: 'model',
            title: 'Model',
            width: 'minmax(200px,1.5fr)',
            render: (m) => (
              <>
                {m.name}
                {m.selected ? ' ★' : ''}
              </>
            ),
          },
          { key: 'validation', title: 'Validation MAE', render: (m) => metric(m.validation?.mae) },
          { key: 'mae', title: 'Holdout MAE', render: (m) => metric(m.holdout?.mae) },
          { key: 'rmse', title: 'RMSE', render: (m) => metric(m.holdout?.rmse) },
          { key: 'smape', title: 'sMAPE (%)', render: (m) => metric(m.holdout?.smape) },
          { key: 'wape', title: 'WAPE (%)', render: (m) => metric(m.holdout?.wape) },
          { key: 'mase', title: 'MASE', render: (m) => metric(m.holdout?.mase) },
        ]}
      />
      {run.models
        .filter((m) => m.status === 'failed')
        .map((m) => (
          <div className="error" key={m.model}>
            {m.name}: {m.error}
          </div>
        ))}
      {run.mode === 'backtest' && model !== 'ensemble' ? (
        <div className="form-actions">
          <button className="button" onClick={() => onReuse(model)}>
            Use saved fitted model
          </button>
        </div>
      ) : null}
      <details>
        <summary>Configuration and provenance</summary>
        <pre className="code">
          {JSON.stringify(
            {
              source: run.source,
              configuration: run.config,
              preprocessing: run.preprocessing,
              population_id: run.population_id,
            },
            null,
            2,
          )}
        </pre>
      </details>
    </Modal>
  )
}
