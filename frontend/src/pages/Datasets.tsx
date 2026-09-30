import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  ChevronRight,
  Database,
  FileSpreadsheet,
  FolderOpen,
  Layers3,
  Network,
  Plus,
  Search,
  ShieldCheck,
} from 'lucide-react'
import { api, upload } from '../api'
import { date, number, percent } from '../format'
import { Badge, Empty, ErrorBox, Field, PageTitle, Panel } from '../components/ui'
import { Modal } from '../components/Modal'
import { VirtualTable } from '../components/VirtualTable'
import type { Dataset } from '../types'
import ImportDialog from './ImportDialog'

type Quality = {
  asset_id: string
  source_id: string
  coverage: number
  observed: number
  expected: number
  constant: boolean
  negative_reads: number
  meter_coverage?: number
}
export default function Datasets() {
  const client = useQueryClient(),
    navigate = useNavigate()
  const [search, setSearch] = useState(''),
    [importing, setImporting] = useState(false),
    [selected, setSelected] = useState<Dataset>(),
    [inventory, setInventory] = useState(false),
    [error, setError] = useState<unknown>(),
    [busy, setBusy] = useState(false)
  const { data: datasets = [], error: loadError } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => api<Dataset[]>('/datasets'),
    refetchInterval: 3000,
  })
  const { data: sources = [] } = useQuery({
    queryKey: ['sources'],
    queryFn: () =>
      api<{ name: string; path: string; bytes: number; registered: boolean; format: string }[]>('/sources'),
    refetchInterval: 5000,
  })
  const { data: overview } = useQuery({
    queryKey: ['overview'],
    queryFn: () =>
      api<{ assets: number; readings: number; registries: { rows: number; cutoff: string }[] }>('/overview'),
    refetchInterval: 4000,
  })
  const { data: quality } = useQuery({
    queryKey: ['quality', selected?.id],
    queryFn: () =>
      api<{
        summary: Record<string, number>
        assets: Quality[]
        timestamp_audit: { source_row: number; timestamp: string; reason: string }[]
      }>('/datasets/' + selected!.id + '/quality'),
    enabled: !!selected,
  })
  const filtered = datasets.filter((d) =>
    (d.name + d.circuit + d.format).toLowerCase().includes(search.toLowerCase()),
  )
  async function register() {
    setBusy(true)
    setError(undefined)
    try {
      await api('/bootstrap', {})
      void client.invalidateQueries()
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }
  return (
    <>
      <PageTitle
        eyebrow="DATA FOUNDATION"
        title="Dataset workspace"
        description="Bring your measurements into focus. Explore, validate, and build from evidence."
      >
        <button className="button secondary" onClick={() => setInventory(true)}>
          <Network size={16} />
          Asset relationships
        </button>
        <button className="button" onClick={() => setImporting(true)}>
          <Plus size={17} />
          Import data
        </button>
      </PageTitle>
      <ErrorBox error={error || loadError} />
      <div className="stat-grid">
        <div className="stat">
          <div>
            <span>Datasets</span>
            <Database size={17} />
          </div>
          <strong>{number(datasets.length)}</strong>
          <small>Versioned and traceable</small>
        </div>
        <div className="stat">
          <div>
            <span>Assets</span>
            <Layers3 size={17} />
          </div>
          <strong>{number(overview?.assets)}</strong>
          <small>Meters, transformers & terminals</small>
        </div>
        <div className="stat">
          <div>
            <span>Measurements</span>
            <ActivityGlyph />
          </div>
          <strong>{number(overview?.readings)}</strong>
          <small>Original resolution preserved</small>
        </div>
        <div className="stat">
          <div>
            <span>PV registry</span>
            <ShieldCheck size={17} />
          </div>
          <strong>{overview?.registries.length ? number(overview.registries[0].rows) : '—'}</strong>
          <small>
            {overview?.registries.length
              ? 'Evidence as of ' + date(overview.registries[0].cutoff)
              : 'Ready to connect'}
          </small>
        </div>
      </div>
      <div className="data-layout">
        <div>
          <Panel
            title="Measurement library"
            subtitle="Your local source of truth"
            actions={
              <div className="search">
                <Search size={16} />
                <input
                  aria-label="Search datasets"
                  placeholder="Search datasets…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </div>
            }
          >
            <div className="library-tabs">
              <span className="active">
                All datasets <b>{datasets.length}</b>
              </span>
              <span>
                Hourly intervals <b>{datasets.filter((d) => d.format === 'wide_ami').length}</b>
              </span>
              <span>
                Recordings <b>{datasets.filter((d) => d.format === 'recording').length}</b>
              </span>
            </div>
            {filtered.length ? (
              <div className="dataset-list">
                {filtered.map((d) => (
                  <article className="dataset-row" key={d.id}>
                    <div className={'dataset-icon ' + (d.format === 'recording' ? 'amber' : '')}>
                      <FileSpreadsheet size={21} />
                    </div>
                    <div className="dataset-info">
                      <button className="dataset-name" onClick={() => setSelected(d)}>
                        {d.name}
                      </button>
                      <div className="dataset-meta">
                        <span>{d.asset_level}</span>
                        <span>
                          {d.format === 'recording'
                            ? number(d.sample_rate) + ' Hz'
                            : number(d.interval_seconds / 3600) + 'h intervals'}
                        </span>
                        <span>
                          {number(d.assets)} {d.assets === 1 ? 'asset' : 'assets'}
                        </span>
                      </div>
                      <div className="dataset-dates">
                        {d.clock_verified === false ? (
                          <>Reported {d.comtrade?.reported_start} / timezone unverified</>
                        ) : (
                          <>
                            {date(d.start)} - {date(d.end)}
                          </>
                        )}
                      </div>
                    </div>
                    <div className="dataset-status">
                      <Badge tone={d.synthetic ? 'amber' : 'green'}>
                        {d.synthetic ? 'Synthetic' : 'Indexed'}
                      </Badge>
                      <small>{number(d.readings)} readings</small>
                    </div>
                    <button
                      className="icon-button"
                      aria-label={'Explore ' + d.name}
                      onClick={() =>
                        navigate((d.format === 'recording' ? '/events' : '/explorer') + '?dataset=' + d.id)
                      }
                    >
                      <ArrowUpRight size={18} />
                    </button>
                  </article>
                ))}
              </div>
            ) : (
              <Empty
                title={search ? 'No matching datasets' : 'Your next discovery starts here'}
                action={
                  <button className="button" disabled={busy} onClick={register}>
                    <FolderOpen size={16} />
                    {busy ? 'Queueing imports…' : 'Register existing data'}
                  </button>
                }
              >
                Six transformer datasets and a PV registry are available in your data directory.
              </Empty>
            )}
            <div className="panel-foot">
              <span>
                <ShieldCheck size={14} />
                Original files are preserved
              </span>
              <span>{filtered.length} datasets</span>
            </div>
          </Panel>
          <div className="next-grid">
            <button className="workflow-card" onClick={() => navigate('/experiments')}>
              <span className="step-number">01</span>
              <div>
                <h3>Put an algorithm to the test</h3>
                <p>Train classifiers and inspect their evidence.</p>
              </div>
              <ArrowRight size={20} />
            </button>
            <button className="workflow-card" onClick={() => navigate('/events')}>
              <span className="step-number amber">02</span>
              <div>
                <h3>Look closer at an event</h3>
                <p>Replay signals at their native resolution.</p>
              </div>
              <ArrowRight size={20} />
            </button>
          </div>
        </div>
        <aside className="right-column">
          <Panel
            title="Available on disk"
            subtitle="data/ · local directory"
            actions={<FolderOpen size={18} />}
          >
            <div className="source-list">
              {sources.map((s) => (
                <div className="source" key={s.path}>
                  <FileSpreadsheet size={17} />
                  <div>
                    <strong title={s.name}>{s.name}</strong>
                    <small>
                      {(s.bytes / 1e6).toFixed(1)} MB ·{' '}
                      {s.format === 'label_registry'
                        ? 'Label registry'
                        : s.format === 'wide_ami'
                          ? 'Wide AMI'
                          : 'Source table - mapping needed'}
                    </small>
                  </div>
                  {s.registered ? <CheckCircle2 size={15} className="teal" /> : null}
                </div>
              ))}
            </div>
            <div className="panel-pad">
              <button className="button secondary full" disabled={busy} onClick={register}>
                Register available files
                <ArrowRight size={15} />
              </button>
              {sources.some((s) => s.format === 'mapping_required' && !s.registered) ? (
                <p className="muted">
                  Other source tables are available in <a href="/notebook">Notebook</a>. Use Import data to
                  declare their measurement mapping.
                </p>
              ) : null}
            </div>
          </Panel>
          <div className="principle-card">
            <span className="eyebrow">EVIDENCE BEFORE ACCURACY</span>
            <h3>Know what your labels mean.</h3>
            <p>
              Confirmed labels, registry evidence, and exploratory assumptions stay visible through every
              experiment.
            </p>
            <a href="/experiments">
              Review an experiment <ChevronRight size={15} />
            </a>
          </div>
        </aside>
      </div>
      {importing ? <ImportDialog onClose={() => setImporting(false)} /> : null}
      {inventory ? <Inventory datasets={datasets} onClose={() => setInventory(false)} /> : null}
      {selected ? (
        <Modal title={selected.name} onClose={() => setSelected(undefined)} wide>
          {selected.comtrade ? (
            <div className="notice amber-notice">
              <p>{selected.comtrade.timing_note}</p>
              <p>Reported clock: {selected.comtrade.reported_start}</p>
              <p>{selected.comtrade.instrument_note}</p>
            </div>
          ) : null}
          <div className="detail-grid">
            <Field label="Source checksum">
              <code>{selected.source_checksum.slice(0, 20)}…</code>
            </Field>
            <Field label="Time convention">
              <span>
                {selected.clock_verified === false ? 'Elapsed time; absolute clock unverified' : null}
              </span>
              <span hidden={selected.clock_verified === false}>
                {selected.timezone} · interval {selected.interval_position}
              </span>
            </Field>
            <Field label="Measurements">
              <span>{selected.channels.map((c) => c.kind + ' (' + c.unit + ')').join(', ')}</span>
            </Field>
            <Field label="Timestamp exclusions">
              <span>{selected.quality.excluded_timestamps || 0} rows retained in audit</span>
            </Field>
          </div>
          {quality?.assets.length ? (
            <VirtualTable
              rows={quality.assets}
              columns={[
                { key: 'source_id', title: 'Asset', render: (r) => r.source_id },
                { key: 'coverage', title: 'Coverage', render: (r) => percent(r.coverage) },
                {
                  key: 'observed',
                  title: 'Observed / expected',
                  render: (r) => number(r.observed) + ' / ' + number(r.expected),
                },
                { key: 'constant', title: 'Profile', render: (r) => (r.constant ? 'Constant' : 'Variable') },
                { key: 'negative', title: 'Negative readings', render: (r) => number(r.negative_reads) },
              ]}
            />
          ) : null}
          {quality?.timestamp_audit.length ? (
            <details>
              <summary>Unresolved timestamp audit ({quality.timestamp_audit.length})</summary>
              <pre className="code">
                {quality.timestamp_audit
                  .map((r) => `${r.source_row}: ${r.timestamp || ''} · ${r.reason}`)
                  .join('\n')}
              </pre>
            </details>
          ) : null}
          <div className="form-actions">
            <button
              className="button"
              onClick={() =>
                navigate(
                  (selected.format === 'recording' ? '/events' : '/explorer') + '?dataset=' + selected.id,
                )
              }
            >
              Open in explorer
              <ArrowRight size={16} />
            </button>
          </div>
        </Modal>
      ) : null}
    </>
  )
}
function ActivityGlyph() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.6">
      <path d="M1 10h3l2-6 4 11 3-8 2 3h2" />
    </svg>
  )
}
function Inventory({ datasets, onClose }: { datasets: Dataset[]; onClose: () => void }) {
  const [error, setError] = useState<unknown>(),
    [message, setMessage] = useState(''),
    [selected, setSelected] = useState<string[]>([]),
    [coverage, setCoverage] = useState(1)
  const client = useQueryClient()
  const { data: rows = [] } = useQuery({
    queryKey: ['relationships'],
    queryFn: () => api<unknown[]>('/relationships'),
  })
  async function importFile(file?: File) {
    if (!file) return
    try {
      const uploaded = await upload(file)
      await api('/relationships/import', { path: uploaded.path })
      setMessage('Relationship inventory imported.')
      void client.invalidateQueries()
    } catch (e) {
      setError(e)
    }
  }
  return (
    <Modal title="Meter-to-transformer inventory" onClose={onClose} wide>
      <p className="muted">
        Import the expected meter inventory, including meters without readings. Effective dates preserve
        topology changes and correct coverage.
      </p>
      <div className="notice">
        CSV columns: meter_id, transformer_id, valid_from, valid_to. Asset IDs use{' '}
        <code>meter:circuit:source_id</code> or <code>transformer:circuit:source_id</code>. Dates must not
        overlap for the same meter.
      </div>
      <Field label="Relationship CSV">
        <input type="file" accept=".csv" onChange={(e) => importFile(e.target.files?.[0])} />
      </Field>
      <p>{rows.length} effective-dated relationships</p>
      <h3>Aggregate meter readings</h3>
      <div className="check-list">
        {datasets
          .filter((d) => d.asset_level === 'meter' && d.format === 'wide_ami')
          .map((d) => (
            <label key={d.id}>
              <input
                type="checkbox"
                checked={selected.includes(d.id)}
                onChange={(e) =>
                  setSelected(e.target.checked ? [...selected, d.id] : selected.filter((id) => id !== d.id))
                }
              />
              {d.name}
            </label>
          ))}
      </div>
      <Field label="Minimum contributing-meter coverage">
        <input
          type="number"
          min="0.01"
          max="1"
          step=".01"
          value={coverage}
          onChange={(e) => setCoverage(+e.target.value)}
        />
      </Field>
      <ErrorBox error={error} />
      {message ? <div className="success">{message}</div> : null}
      <div className="form-actions">
        <button
          className="button"
          disabled={!selected.length}
          onClick={async () => {
            try {
              await api('/aggregate', { dataset_ids: selected, min_meter_coverage: coverage })
              setMessage('Aggregation queued.')
              void client.invalidateQueries()
            } catch (e) {
              setError(e)
            }
          }}
        >
          Create transformer dataset
        </button>
      </div>
    </Modal>
  )
}
