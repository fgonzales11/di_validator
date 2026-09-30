import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { ArrowRight, CalendarDays, CheckCheck, Search, SlidersHorizontal, Tag } from 'lucide-react'
import type { Data, Layout } from 'plotly.js'
import { api, upload } from '../api'
import { date, number } from '../format'
import { useStored } from '../hooks/useStored'
import { Badge, Empty, ErrorBox, Field, PageTitle, Panel } from '../components/ui'
import { Chart } from '../components/Chart'
import { Modal } from '../components/Modal'
import type { Asset, Dataset, WindowData } from '../types'

const views = [
  ['series', 'Time series'],
  ['daily', 'Daily profile'],
  ['seasonal', 'Seasonal'],
  ['heatmap', 'Calendar heatmap'],
  ['distribution', 'Distribution'],
]
export default function Explorer() {
  const [params, setParams] = useSearchParams()
  const [chosenDataset, setDataset] = useStored('explorer.dataset', params.get('dataset') || '')
  const [selected, setSelected] = useState<string[]>(params.get('asset') ? [params.get('asset')!] : [])
  const [view, setView] = useStored('explorer.view', 'series'),
    [power, setPower] = useState(false),
    [search, setSearch] = useState(''),
    [start, setStart] = useState(''),
    [end, setEnd] = useState(''),
    [labelOpen, setLabelOpen] = useState(false)
  const { data: all = [] } = useQuery({ queryKey: ['datasets'], queryFn: () => api<Dataset[]>('/datasets') })
  const datasets = all.filter((d) => d.format === 'wide_ami')
  const dataset = datasets.find((d) => d.id === (params.get('dataset') || chosenDataset)) || datasets[0]
  const assets = useQuery({
    queryKey: ['assets', dataset?.id, search],
    queryFn: () =>
      api<Asset[]>('/assets?' + new URLSearchParams({ dataset_id: dataset!.id, search, limit: '300' })),
    enabled: !!dataset,
  })
  const effective = selected.length ? selected : assets.data?.length ? [assets.data[0].id] : []
  const selectedKey = effective.join(',')
  const query = useQuery({
    queryKey: ['series', dataset?.id, selectedKey, start, end, view, power],
    queryFn: () =>
      api<WindowData>(
        '/series?' +
          new URLSearchParams({
            dataset_id: dataset!.id,
            asset_ids: selectedKey,
            view,
            power: String(power),
            ...(start ? { start } : {}),
            ...(end ? { end: end + 'T23:59:59' } : {}),
          }),
      ),
    enabled: !!dataset && !!selectedKey,
  })
  const plot = useMemo<Data[]>(
    () =>
      (query.data?.traces.map((t) =>
        view === 'heatmap'
          ? {
              type: 'heatmap',
              x: t.x,
              y: t.y,
              z: t.z,
              colorscale: [
                [0, '#f3f5ef'],
                [0.3, '#bddbcb'],
                [0.7, '#3d9e89'],
                [1, '#075b55'],
              ],
              colorbar: { title: { text: query.data.unit } },
              hoverongaps: false,
            }
          : view === 'distribution'
            ? { type: 'bar', x: t.x, y: t.y, name: t.asset_id?.split(':').at(-1) }
            : {
                type: 'scatter',
                mode: view === 'series' ? 'lines' : 'lines+markers',
                x: t.x,
                y: t.y,
                connectgaps: false,
                name: t.asset_id?.split(':').at(-1),
                line: { width: 1.6 },
              },
      ) as Data[]) || [],
    [query.data, view],
  )
  const layout = useMemo<Partial<Layout>>(
    () => ({
      xaxis: {
        title: {
          text:
            view === 'daily' || view === 'heatmap'
              ? 'Local hour'
              : view === 'seasonal'
                ? 'Month'
                : view === 'distribution'
                  ? query.data?.unit
                  : 'Time · ' + (dataset?.timezone || ''),
        },
        gridcolor: '#edf1ee',
        zeroline: false,
      },
      yaxis: {
        title: {
          text: view === 'distribution' ? 'Observations' : view === 'heatmap' ? 'Day' : query.data?.unit,
        },
        gridcolor: '#edf1ee',
        zeroline: false,
      },
      barmode: 'overlay',
      dragmode: 'zoom',
    }),
    [view, query.data?.unit, dataset?.timezone],
  )
  if (!dataset)
    return (
      <>
        <PageTitle
          eyebrow="SIGNAL EXPLORATION"
          title="Measurement explorer"
          description="Find patterns in your measurements before building a model."
        />
        <Empty
          title="Import an interval dataset first"
          action={
            <a className="button" href="/">
              Open datasets
              <ArrowRight size={16} />
            </a>
          }
        >
          The explorer supports transformer and meter profiles at their declared sampling interval.
        </Empty>
      </>
    )
  return (
    <>
      <PageTitle
        eyebrow="SIGNAL EXPLORATION"
        title="Measurement explorer"
        description="From a year of behavior to a single interval. Follow the signal."
      >
        <button className="button secondary" disabled={!effective.length} onClick={() => setLabelOpen(true)}>
          <Tag size={16} />
          Review labels
        </button>
      </PageTitle>
      <div className="explorer-layout">
        <aside className="filter-panel">
          <div className="filter-title">
            <SlidersHorizontal size={17} />
            <strong>Selection</strong>
          </div>
          <Field label="Dataset">
            <select
              value={dataset.id}
              onChange={(e) => {
                setParams({ dataset: e.target.value })
                setDataset(e.target.value)
                setSelected([])
              }}
            >
              {datasets.map((d) => (
                <option value={d.id} key={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
          </Field>
          <div className="filter-pair">
            <Field label="From">
              <input type="date" value={start} onChange={(e) => setStart(e.target.value)} />
            </Field>
            <Field label="Through">
              <input type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
            </Field>
          </div>
          <div className="filter-divider" />
          <div className="filter-subheading">
            Assets <Badge>{number(dataset.assets)}</Badge>
          </div>
          <div className="search">
            <Search size={15} />
            <input
              placeholder="Find an asset…"
              aria-label="Find an asset"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <small className="muted">Compare up to 8 profiles</small>
          <div className="asset-list">
            {assets.data?.map((asset) => (
              <label key={asset.id}>
                <input
                  type="checkbox"
                  checked={effective.includes(asset.id)}
                  disabled={effective.length >= 8 && !effective.includes(asset.id)}
                  onChange={(e) =>
                    setSelected(
                      e.target.checked ? [...effective, asset.id] : effective.filter((a) => a !== asset.id),
                    )
                  }
                />
                <span>{asset.source_id}</span>
                <small>{asset.level === 'transformer' ? 'TX' : 'M'}</small>
              </label>
            ))}
          </div>
          <div className="filter-divider" />
          <label className="toggle">
            <input type="checkbox" checked={power} onChange={(e) => setPower(e.target.checked)} />
            <span>Show average power (kW)</span>
          </label>
          <div className="filter-note">
            <CheckCheck size={17} />
            <p>Negative net energy is retained. Data gaps are never filled with zeros.</p>
          </div>
        </aside>
        <div className="explorer-main">
          <div className="selection-banner">
            <div>
              <Badge tone="green">{dataset.asset_level}</Badge>
              <strong>{dataset.circuit}</strong>
            </div>
            <span>
              <CalendarDays size={14} />
              {date(dataset.start)} — {date(dataset.end)}
            </span>
          </div>
          <Panel
            title="Load behavior"
            subtitle={`${effective.length} selected ${effective.length === 1 ? 'asset' : 'assets'} · ${dataset.interval_seconds / 3600}h intervals`}
            actions={<Badge>{power ? 'kW' : dataset.channels[0].unit}</Badge>}
          >
            <div className="chart-tabs">
              {views.map(([key, label]) => (
                <button className={view === key ? 'active' : ''} key={key} onClick={() => setView(key)}>
                  {label}
                </button>
              ))}
            </div>
            <ErrorBox error={query.error} />
            {query.isFetching ? <div className="chart-loading">Updating measurements…</div> : null}
            <Chart data={plot} layout={layout} height={450} title={views.find((v) => v[0] === view)?.[1]} />
            <div className="chart-caption">
              <span>
                {query.data?.aggregation} · {dataset.timezone}
              </span>
              <span>Drag to zoom · Double-click to reset · Toolbar to export</span>
            </div>
          </Panel>
          <div className="insight-grid">
            <div>
              <span className="eyebrow">SOURCE TRACEABILITY</span>
              <h3>Every point has a provenance.</h3>
              <p>
                Dataset version <code>{dataset.id.slice(0, 12)}</code>
                <br />
                Original values and excluded timestamps remain available in the dataset audit.
              </p>
            </div>
            <div>
              <span className="eyebrow">INTERVAL CONVENTION</span>
              <h3>
                {dataset.interval_position === 'unknown'
                  ? 'Reported clock labels retained.'
                  : 'Timestamp marks interval ' + dataset.interval_position + '.'}
              </h3>
              <p>
                {dataset.interval_position === 'unknown'
                  ? 'Confirm start versus end before aligning this source with another dataset.'
                  : 'Average power uses the declared interval duration.'}
              </p>
            </div>
          </div>
        </div>
      </div>
      {labelOpen ? <LabelReview assetId={effective[0]} onClose={() => setLabelOpen(false)} /> : null}
    </>
  )
}
function LabelReview({ assetId, onClose }: { assetId: string; onClose: () => void }) {
  const [target, setTarget] = useState('PV'),
    [value, setValue] = useState('1'),
    [evidence, setEvidence] = useState(''),
    [start, setStart] = useState('2025-01-01'),
    [end, setEnd] = useState('2025-12-31'),
    [error, setError] = useState<unknown>(),
    [message, setMessage] = useState('')
  const client = useQueryClient()
  const { data: labels = [] } = useQuery({
    queryKey: ['labels'],
    queryFn: () =>
      api<
        {
          id: string
          asset_id: string
          target: string
          value: number | null
          evidence: string
          created_at: string
        }[]
      >('/labels'),
  })
  async function save(e: React.FormEvent) {
    e.preventDefault()
    try {
      await api('/labels', {
        asset_id: assetId,
        target,
        value: value === 'unknown' ? null : +value,
        evidence,
        valid_from: start,
        valid_to: end + 'T23:59:59',
        verification: 'confirmed',
      })
      setMessage('Label revision saved. Completed experiments retain their original evidence.')
      void client.invalidateQueries({ queryKey: ['labels'] })
    } catch (e) {
      setError(e)
    }
  }
  return (
    <Modal title="Label evidence" onClose={onClose}>
      <p className="muted">
        Asset <strong>{assetId.split(':').at(-1)}</strong> · {assetId.split(':')[0]} level
      </p>
      <form onSubmit={save}>
        <div className="form-grid">
          <Field label="Target">
            <select value={target} onChange={(e) => setTarget(e.target.value)}>
              <option>PV</option>
              <option>EV</option>
            </select>
          </Field>
          <Field label="Confirmed evidence">
            <select value={value} onChange={(e) => setValue(e.target.value)}>
              <option value="1">Present</option>
              <option value="0">Absent</option>
              <option value="unknown">Unknown / withdraw evidence</option>
            </select>
          </Field>
          <Field label="Valid from">
            <input type="date" required value={start} onChange={(e) => setStart(e.target.value)} />
          </Field>
          <Field label="Valid through">
            <input type="date" required value={end} onChange={(e) => setEnd(e.target.value)} />
          </Field>
        </div>
        <Field label="Evidence source or review note">
          <textarea
            required
            value={evidence}
            onChange={(e) => setEvidence(e.target.value)}
            placeholder="Describe how this label was verified…"
          />
        </Field>
        <ErrorBox error={error} />
        {message ? <div className="success">{message}</div> : null}
        <button className="button" type="submit">
          Save label revision
        </button>
      </form>
      <details>
        <summary>Import confirmed labels from CSV</summary>
        <p className="muted">
          Columns: asset_id, target, value, evidence, valid_from, valid_to, verification. Leave value blank
          for unknown.
        </p>
        <input
          aria-label="Import label CSV"
          type="file"
          accept=".csv"
          onChange={async (e) => {
            const file = e.target.files?.[0]
            if (!file) return
            try {
              const u = await upload(file)
              await api('/labels/import', { path: u.path })
              setMessage('Labels imported.')
              void client.invalidateQueries()
            } catch (e) {
              setError(e)
            }
          }}
        />
      </details>
      <h3>Revision history</h3>
      {labels
        .filter((l) => l.asset_id === assetId)
        .map((l) => (
          <div key={l.id} className="label-history">
            <Badge>
              {l.target} {l.value === null ? 'unknown' : l.value === 1 ? 'present' : 'absent'}
            </Badge>
            <span>{l.evidence}</span>
            <small>{date(l.created_at)}</small>
          </div>
        ))}
    </Modal>
  )
}
