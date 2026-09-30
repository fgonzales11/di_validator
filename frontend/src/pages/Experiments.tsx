import { useMemo, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ArrowRight,
  ArrowUpRight,
  FlaskConical,
  GitCompareArrows,
  Plus,
  Save,
  SlidersHorizontal,
  Trophy,
} from 'lucide-react'
import type { Data, Layout } from 'plotly.js'
import { api } from '../api'
import { date, number, percent } from '../format'
import { Badge, Empty, ErrorBox, ExportLink, Field, PageTitle, Panel } from '../components/ui'
import { Chart } from '../components/Chart'
import { Modal } from '../components/Modal'
import { VirtualTable } from '../components/VirtualTable'
import type { Dataset, Experiment } from '../types'

type Config = {
  name: string
  dataset_ids: string[]
  asset_ids: string[]
  target: string
  label_policy: string
  start: string | null
  end: string | null
  min_coverage: number
  seed: number
  test_size: number
  folds: number
  models: string[]
  parameters: Record<string, Record<string, number>>
  threshold_policy: string
  feature_groups?: string[]
}
const names: Record<string, string> = {
  logistic: 'Logistic regression',
  svm: 'RBF SVM',
  forest: 'Random forest',
  boosting: 'Gradient boosting',
}
const initial: Config = {
  name: 'PV · transformer benchmark',
  dataset_ids: [],
  asset_ids: [],
  target: 'PV',
  label_policy: 'registry',
  start: null,
  end: null,
  min_coverage: 0.9,
  seed: 42,
  test_size: 0.2,
  folds: 5,
  models: ['logistic', 'svm', 'forest', 'boosting'],
  parameters: {},
  threshold_policy: 'default',
}
export default function Experiments() {
  const [creating, setCreating] = useState(false),
    [selected, setSelected] = useState<string>(),
    [compare, setCompare] = useState<string[]>([]),
    [comparing, setComparing] = useState(false)
  const { data: experiments = [], error } = useQuery({
    queryKey: ['experiments'],
    queryFn: () => api<Experiment[]>('/experiments'),
    refetchInterval: 3000,
  })
  const classification = experiments.filter((e) => e.kind === 'classification')
  const active = experiments.find((e) => e.id === selected)
  return (
    <>
      <PageTitle
        eyebrow="REPRODUCIBLE RESEARCH"
        title="Experiment workbench"
        description="Define a question, freeze the evidence, and evaluate the result."
      >
        <button className="button secondary" disabled={compare.length < 2} onClick={() => setComparing(true)}>
          <GitCompareArrows size={16} />
          Compare {compare.length ? `(${compare.length})` : ''}
        </button>
        <button className="button" onClick={() => setCreating(true)}>
          <Plus size={17} />
          New experiment
        </button>
      </PageTitle>
      <ErrorBox error={error} />
      <div className="experiment-intro">
        <div className="intro-icon">
          <FlaskConical size={25} />
        </div>
        <div>
          <strong>Evidence travels with every result.</strong>
          <p>Data versions, label sources, training splits, and parameters are saved together.</p>
        </div>
        <Badge tone="green">Reproducible by design</Badge>
      </div>
      {classification.length ? (
        <Panel
          title="Classification runs"
          subtitle="Select a run to inspect holdout performance and individual predictions"
        >
          <div className="experiment-list">
            {classification.map((e) => (
              <article className="experiment-row" key={e.id}>
                <input
                  aria-label={'Compare ' + e.name}
                  type="checkbox"
                  checked={compare.includes(e.id)}
                  onChange={(ev) =>
                    setCompare(ev.target.checked ? [...compare, e.id] : compare.filter((id) => id !== e.id))
                  }
                />
                <div className="experiment-row-main">
                  <button className="dataset-name" onClick={() => setSelected(e.id)}>
                    {e.name}
                  </button>
                  <span>
                    {date(e.created_at)} · {number(e.counts.labeled)} labeled assets · {e.counts.test} held
                    out
                  </span>
                </div>
                <Badge tone={e.label_policy === 'confirmed' ? 'green' : 'amber'}>
                  {e.label_policy === 'heuristic'
                    ? 'Heuristic agreement'
                    : e.label_policy === 'provisional'
                      ? 'Exploratory'
                      : e.label_policy === 'registry'
                        ? 'Registry evidence'
                        : 'Confirmed labels'}
                </Badge>
                <div className="run-score">
                  <small>HOLDOUT AP</small>
                  <strong>{percent(e.models.find((m) => m.selected)?.test.average_precision)}</strong>
                </div>
                <button
                  className="icon-button"
                  aria-label={'Inspect ' + e.name}
                  onClick={() => setSelected(e.id)}
                >
                  <ArrowUpRight size={18} />
                </button>
              </article>
            ))}
          </div>
        </Panel>
      ) : (
        <Panel title="Your experiments" subtitle="Classification benchmarks with an untouched holdout">
          <Empty
            title="Build your first benchmark"
            action={
              <button className="button" onClick={() => setCreating(true)}>
                <Plus size={16} />
                Configure experiment
              </button>
            }
          >
            Compare EV or PV classifiers, inspect false positives, and export a reproducible result.
          </Empty>
        </Panel>
      )}
      <div className="three-notes">
        <div>
          <span className="step-number">1</span>
          <h3>Define the cohort</h3>
          <p>Select assets, a feature window, and the evidence you will evaluate against.</p>
        </div>
        <div>
          <span className="step-number">2</span>
          <h3>Train without leakage</h3>
          <p>Related assets stay together. Preprocessing and model selection use training folds.</p>
        </div>
        <div>
          <span className="step-number">3</span>
          <h3>Inspect the holdout</h3>
          <p>Follow each prediction back to its profile. Compare like-for-like populations.</p>
        </div>
      </div>
      {creating ? <CreateExperiment onClose={() => setCreating(false)} /> : null}
      {active ? <Results experiment={active} onClose={() => setSelected(undefined)} /> : null}
      {comparing ? (
        <Comparison
          experiments={classification.filter((e) => compare.includes(e.id))}
          onClose={() => setComparing(false)}
        />
      ) : null}
    </>
  )
}
function CreateExperiment({ onClose }: { onClose: () => void }) {
  const [form, setForm] = useState<Config>(initial),
    [error, setError] = useState<unknown>(),
    [busy, setBusy] = useState(false),
    [presetMessage, setPresetMessage] = useState('')
  const client = useQueryClient()
  const { data: all = [] } = useQuery({ queryKey: ['datasets'], queryFn: () => api<Dataset[]>('/datasets') })
  const { data: presets = [] } = useQuery({
    queryKey: ['presets'],
    queryFn: () => api<{ id: string; kind: string; name: string; config: Config }[]>('/presets'),
  })
  const { data: registered = [] } = useQuery({
    queryKey: ['algorithms'],
    queryFn: () => api<{ id: string; name: string; family: string; status: string }[]>('/algorithms'),
  })
  const modelNames = {
    ...names,
    ...Object.fromEntries(
      registered
        .filter((a) => a.family === 'Classification' && a.status === 'ready' && a.id !== 'baseline')
        .map((a) => [a.id, a.name]),
    ),
  }
  const datasets = all.filter((d) => d.format === 'wide_ami' && d.interval_seconds === 3600)
  function set<K extends keyof Config>(k: K, v: Config[K]) {
    setForm((f) => ({ ...f, [k]: v }))
  }
  function parameter(model: string, key: string, value: number) {
    setForm((f) => ({
      ...f,
      parameters: { ...f.parameters, [model]: { ...f.parameters[model], [key]: value } },
    }))
  }
  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(undefined)
    try {
      await api('/experiments', form)
      void client.invalidateQueries()
      onClose()
    } catch (e) {
      setError(e)
      setBusy(false)
    }
  }
  return (
    <Modal title="Configure classification experiment" onClose={onClose} wide>
      <form onSubmit={submit}>
        <ErrorBox error={error} />
        <div className="form-grid">
          <Field label="Experiment name">
            <input required value={form.name} onChange={(e) => set('name', e.target.value)} />
          </Field>
          <Field label="Target">
            <select
              value={form.target}
              onChange={(e) =>
                setForm((f) => ({
                  ...f,
                  target: e.target.value,
                  label_policy: e.target.value === 'EV' ? 'confirmed' : 'registry',
                  name: e.target.value + ' · classifier benchmark',
                }))
              }
            >
              <option>PV</option>
              <option>EV</option>
            </select>
          </Field>
        </div>
        <Field label="Saved configuration">
          <select
            value=""
            onChange={(e) => {
              const p = presets.find((p) => p.id === e.target.value)
              if (p) setForm(p.config)
            }}
          >
            <option value="">Choose a preset…</option>
            {presets
              .filter((p) => p.kind === 'classification')
              .map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
          </select>
        </Field>
        <div className="section-title">
          <h3>1. Data & evidence</h3>
          <button
            type="button"
            className="text-button"
            onClick={() =>
              set(
                'dataset_ids',
                datasets.map((d) => d.id),
              )
            }
          >
            Select all datasets
          </button>
        </div>
        <div className="dataset-checks">
          {datasets.map((d) => (
            <label key={d.id}>
              <input
                type="checkbox"
                checked={form.dataset_ids.includes(d.id)}
                onChange={(e) =>
                  set(
                    'dataset_ids',
                    e.target.checked
                      ? [...form.dataset_ids, d.id]
                      : form.dataset_ids.filter((id) => id !== d.id),
                  )
                }
              />
              <span>
                {d.name}
                <small>
                  {number(d.assets)} {d.asset_level} profiles
                </small>
              </span>
            </label>
          ))}
        </div>
        {!datasets.length ? (
          <p className="muted">Import hourly interval data to enable classification.</p>
        ) : null}
        <div className="form-grid">
          <Field label="Feature window start" hint="Blank uses common data coverage.">
            <input
              type="date"
              value={form.start || ''}
              onChange={(e) => set('start', e.target.value || null)}
            />
          </Field>
          <Field
            label="Feature window through"
            hint="PV registry windows are capped before the registry snapshot."
          >
            <input
              type="date"
              value={form.end?.slice(0, 10) || ''}
              onChange={(e) => set('end', e.target.value ? e.target.value + 'T23:00:00' : null)}
            />
          </Field>
        </div>
        <Field label="Label evidence policy">
          <select value={form.label_policy} onChange={(e) => set('label_policy', e.target.value)}>
            <option value="confirmed">Confirmed labels only</option>
            {form.target === 'PV' ? (
              <>
                <option value="registry">PV registry · nonmatches remain unknown</option>
                <option value="provisional">Exploratory · provisional registry negatives</option>
              </>
            ) : (
              <option value="heuristic">Exploratory · EV heuristic agreement</option>
            )}
          </select>
        </Field>
        <div
          className={
            'notice ' + (['provisional', 'heuristic'].includes(form.label_policy) ? 'amber-notice' : '')
          }
        >
          {form.label_policy === 'provisional'
            ? 'Registry nonmatches will be treated as provisional negatives. Metrics describe this assumption, not confirmed absence.'
            : form.label_policy === 'heuristic'
              ? 'This run measures agreement with unvalidated screening rules. It does not establish field detection accuracy.'
              : 'Unknown labels are excluded. Both classes need at least five labeled assets and enough independent groups.'}
        </div>
        <h3>2. Models & validation</h3>
        <div className="model-checks">
          <label>
            <input type="checkbox" checked disabled />
            Majority baseline <small>Always included</small>
          </label>
          {Object.entries(modelNames).map(([key, name]) => (
            <label key={key}>
              <input
                type="checkbox"
                checked={form.models.includes(key)}
                onChange={(e) =>
                  set(
                    'models',
                    e.target.checked ? [...form.models, key] : form.models.filter((m) => m !== key),
                  )
                }
              />
              {name}
            </label>
          ))}
        </div>
        <div className="form-grid">
          <Field label="Minimum profile coverage">
            <input
              type="number"
              min=".01"
              max="1"
              step=".01"
              value={form.min_coverage}
              onChange={(e) => set('min_coverage', +e.target.value)}
            />
          </Field>
          <Field label="Holdout fraction">
            <input
              type="number"
              min=".06"
              max=".49"
              step=".01"
              value={form.test_size}
              onChange={(e) => set('test_size', +e.target.value)}
            />
          </Field>
          <Field label="Training CV folds">
            <input
              type="number"
              min="2"
              max="10"
              value={form.folds}
              onChange={(e) => set('folds', +e.target.value)}
            />
          </Field>
          <Field label="Random seed">
            <input type="number" value={form.seed} onChange={(e) => set('seed', +e.target.value)} />
          </Field>
          <Field label="Decision threshold">
            <select value={form.threshold_policy} onChange={(e) => set('threshold_policy', e.target.value)}>
              <option value="default">Fixed model default</option>
              <option value="training_f1">Select using training out-of-fold F1</option>
            </select>
          </Field>
        </div>
        <details>
          <summary>Feature groups</summary>
          <div className="model-checks">
            {['load', 'calendar', 'ramps', 'shape'].map((group) => (
              <label key={group}>
                <input
                  type="checkbox"
                  checked={(form.feature_groups || ['load', 'calendar', 'ramps', 'shape']).includes(group)}
                  onChange={(e) => {
                    const current = form.feature_groups || ['load', 'calendar', 'ramps', 'shape']
                    set(
                      'feature_groups',
                      e.target.checked ? [...current, group] : current.filter((g) => g !== group),
                    )
                  }}
                />
                {group}
              </label>
            ))}
          </div>
        </details>
        <details>
          <summary>
            Model parameters <SlidersHorizontal size={15} />
          </summary>
          <div className="form-grid">
            {form.models.includes('logistic') ? (
              <Field label="Logistic regression · C">
                <input
                  type="number"
                  min=".001"
                  step="any"
                  value={form.parameters.logistic?.C ?? 1}
                  onChange={(e) => parameter('logistic', 'C', +e.target.value)}
                />
              </Field>
            ) : null}
            {form.models.includes('svm') ? (
              <Field label="SVM · C">
                <input
                  type="number"
                  min=".001"
                  step="any"
                  value={form.parameters.svm?.C ?? 1}
                  onChange={(e) => parameter('svm', 'C', +e.target.value)}
                />
              </Field>
            ) : null}
            {form.models.includes('forest') ? (
              <>
                <Field label="Random forest · trees">
                  <input
                    type="number"
                    min="10"
                    max="2000"
                    value={form.parameters.forest?.n_estimators ?? 250}
                    onChange={(e) => parameter('forest', 'n_estimators', +e.target.value)}
                  />
                </Field>
                <Field label="Random forest · maximum depth">
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={form.parameters.forest?.max_depth ?? 8}
                    onChange={(e) => parameter('forest', 'max_depth', +e.target.value)}
                  />
                </Field>
              </>
            ) : null}
            {form.models.includes('boosting') ? (
              <Field label="Gradient boosting · iterations">
                <input
                  type="number"
                  min="10"
                  max="1000"
                  value={form.parameters.boosting?.max_iter ?? 150}
                  onChange={(e) => parameter('boosting', 'max_iter', +e.target.value)}
                />
              </Field>
            ) : null}
          </div>
        </details>
        {presetMessage ? <p className="success">{presetMessage}</p> : null}
        <div className="form-actions">
          <button
            type="button"
            className="button secondary"
            onClick={async () => {
              try {
                await api('/presets', { kind: 'classification', name: form.name, config: form })
                setPresetMessage('Configuration saved.')
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
            type="submit"
            disabled={busy || !form.dataset_ids.length || !form.models.length}
          >
            {busy ? 'Queueing…' : 'Run experiment'}
            <ArrowRight size={16} />
          </button>
        </div>
      </form>
    </Modal>
  )
}
type Prediction = {
  asset_id: string
  model: string
  label: number
  prediction: number
  score: number
  score_type: string
  evidence: string
  dataset_id: string
}
function Results({ experiment: e, onClose }: { experiment: Experiment; onClose: () => void }) {
  const [model, setModel] = useState(e.selected_model),
    [errors, setErrors] = useState(false),
    [tab, setTab] = useState('metrics')
  const result = e.models.find((m) => m.model === model)!
  const { data: predictions = [] } = useQuery({
    queryKey: ['predictions', e.id, model, errors],
    queryFn: () =>
      api<Prediction[]>(
        '/experiments/' +
          e.id +
          '/predictions?' +
          new URLSearchParams({ model, errors_only: String(errors) }),
      ),
  })
  const roc = useMemo<Data[]>(
    () => [
      { type: 'scatter', mode: 'lines', x: result.curves.fpr, y: result.curves.tpr, name: 'ROC' },
      {
        type: 'scatter',
        mode: 'lines',
        x: [0, 1],
        y: [0, 1],
        line: { dash: 'dot', color: '#bec8c1' },
        name: 'Chance',
      },
    ],
    [result],
  )
  const pr = useMemo<Data[]>(
    () => [
      {
        type: 'scatter',
        mode: 'lines',
        x: result.curves.recall,
        y: result.curves.precision,
        name: 'Precision–recall',
      },
    ],
    [result],
  )
  const rocLayout = useMemo<Partial<Layout>>(
    () => ({
      xaxis: { title: { text: 'False positive rate' } },
      yaxis: { title: { text: 'True positive rate' }, range: [0, 1.05] },
    }),
    [],
  )
  const prLayout = useMemo<Partial<Layout>>(
    () => ({
      xaxis: { title: { text: 'Recall' } },
      yaxis: { title: { text: 'Precision' }, range: [0, 1.05] },
    }),
    [],
  )
  return (
    <Modal title={e.name} onClose={onClose} wide>
      <div className="results-head">
        <Badge tone={e.label_policy === 'confirmed' ? 'green' : 'amber'}>{e.interpretation}</Badge>
        <ExportLink id={e.id} />
      </div>
      <p className="muted">
        {date(e.effective_start)} — {date(e.effective_end)} · {e.counts.train} training / {e.counts.test}{' '}
        holdout assets · {e.counts.unknown} unknown labels excluded
      </p>
      {!e.beats_baseline ? (
        <div className="notice amber-notice">
          The selected model did not exceed the majority baseline's training-CV average precision.
        </div>
      ) : null}
      <div className="chart-tabs">
        <button className={tab === 'metrics' ? 'active' : ''} onClick={() => setTab('metrics')}>
          Model comparison
        </button>
        <button className={tab === 'predictions' ? 'active' : ''} onClick={() => setTab('predictions')}>
          Prediction review
        </button>
        <button className={tab === 'manifest' ? 'active' : ''} onClick={() => setTab('manifest')}>
          Provenance
        </button>
      </div>
      {tab === 'metrics' ? (
        <>
          <VirtualTable
            rows={e.models}
            onRow={(r) => setModel(r.model)}
            columns={[
              {
                key: 'model',
                title: 'Model',
                width: 'minmax(210px,2fr)',
                render: (r) => (
                  <span>
                    {r.name}
                    {r.selected ? <Trophy size={13} className="teal" /> : null}
                  </span>
                ),
              },
              { key: 'cv', title: 'Training CV AP', render: (r) => percent(r.cv.average_precision) },
              { key: 'ap', title: 'Holdout AP', render: (r) => percent(r.test.average_precision) },
              { key: 'f1', title: 'Holdout F1', render: (r) => percent(r.test.f1) },
              {
                key: 'balanced',
                title: 'Balanced accuracy',
                render: (r) => percent(r.test.balanced_accuracy),
              },
            ]}
          />
          <div className="section-title">
            <h3>{result.name}</h3>
            <Badge>{result.selected ? 'Selected on training CV' : 'Comparison model'}</Badge>
          </div>
          <div className="mini-stats">
            {['precision', 'recall', 'roc_auc'].map((key) => (
              <div key={key}>
                <span>{key.replace('_', ' ')}</span>
                <strong>{percent(result.test[key])}</strong>
              </div>
            ))}
            <div>
              <span>Accuracy · 95% interval</span>
              <strong>{percent(result.test.accuracy)}</strong>
              <small>
                {percent(result.test.accuracy_low)} – {percent(result.test.accuracy_high)}
              </small>
            </div>
          </div>
          <div className="two-charts">
            <Chart data={roc} layout={rocLayout} height={260} title="ROC curve" />
            <Chart data={pr} layout={prLayout} height={260} title="Precision-recall curve" />
          </div>
          <div className="confusion">
            <h3>Confusion matrix</h3>
            <table>
              <thead>
                <tr>
                  <th>Actual / predicted</th>
                  <th>Absent / negative</th>
                  <th>Present / positive</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th>Absent / negative</th>
                  <td>{result.confusion[0][0]}</td>
                  <td>{result.confusion[0][1]}</td>
                </tr>
                <tr>
                  <th>Present / positive</th>
                  <td>{result.confusion[1][0]}</td>
                  <td>{result.confusion[1][1]}</td>
                </tr>
              </tbody>
            </table>
            <p className="muted">
              Class meaning follows the evidence policy. Sampling intervals do not capture label error.
            </p>
          </div>
        </>
      ) : tab === 'predictions' ? (
        <>
          <div className="prediction-controls">
            <Field label="Model">
              <select value={model} onChange={(ev) => setModel(ev.target.value)}>
                {e.models.map((m) => (
                  <option key={m.model} value={m.model}>
                    {m.name}
                  </option>
                ))}
              </select>
            </Field>
            <label className="toggle">
              <input type="checkbox" checked={errors} onChange={(ev) => setErrors(ev.target.checked)} />
              Errors only
            </label>
          </div>
          <VirtualTable
            rows={predictions}
            columns={[
              {
                key: 'asset',
                title: 'Asset profile',
                render: (r) => (
                  <a
                    className="text-link"
                    href={'/explorer?' + new URLSearchParams({ dataset: r.dataset_id, asset: r.asset_id })}
                  >
                    {r.asset_id.split(':').at(-1)}
                    <ArrowUpRight size={12} />
                  </a>
                ),
              },
              { key: 'truth', title: 'Label / prediction', render: (r) => r.label + ' / ' + r.prediction },
              { key: 'score', title: 'Score', render: (r) => r.score.toFixed(3) },
              {
                key: 'type',
                title: 'Score type',
                width: 'minmax(220px,2fr)',
                render: (r) => r.score_type.replaceAll('_', ' '),
              },
              {
                key: 'evidence',
                title: 'Label evidence',
                width: 'minmax(220px,2fr)',
                render: (r) => r.evidence,
              },
            ]}
          />
        </>
      ) : (
        <pre className="code">
          {JSON.stringify(
            {
              configuration: e.config,
              split_id: e.split_id,
              evidence_id: e.evidence_id,
              dataset_ids: e.dataset_ids,
            },
            null,
            2,
          )}
        </pre>
      )}
    </Modal>
  )
}
function Comparison({ experiments, onClose }: { experiments: Experiment[]; onClose: () => void }) {
  const comparable =
    new Set(
      experiments.map(
        (e) =>
          e.split_id +
          e.evidence_id +
          e.dataset_ids.slice().sort().join() +
          e.effective_start +
          e.effective_end,
      ),
    ).size === 1
  return (
    <Modal title="Compare experiments" onClose={onClose} wide>
      <div className={'notice ' + (!comparable ? 'amber-notice' : '')}>
        {comparable
          ? 'These runs share the same datasets, label evidence, feature windows, and evaluation population.'
          : 'These runs differ in datasets, evidence, windows, or split membership. Their metrics are not a like-for-like ranking.'}
      </div>
      <VirtualTable
        rows={experiments}
        columns={[
          { key: 'name', title: 'Experiment', width: 'minmax(220px,2fr)', render: (e) => e.name },
          {
            key: 'model',
            title: 'Selected model',
            width: 'minmax(160px,1fr)',
            render: (e) => e.models.find((m) => m.selected)?.name,
          },
          {
            key: 'ap',
            title: 'Holdout AP',
            render: (e) => percent(e.models.find((m) => m.selected)?.test.average_precision),
          },
          {
            key: 'f1',
            title: 'Holdout F1',
            render: (e) => percent(e.models.find((m) => m.selected)?.test.f1),
          },
          { key: 'test', title: 'Holdout assets', render: (e) => e.counts.test },
        ]}
      />
    </Modal>
  )
}
