import { useEffect, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { BookOpen, Check, Code2, Copy, Download, ExternalLink, Maximize2, Minimize2 } from 'lucide-react'
import { api } from '../api'
import { number } from '../format'
import { useStored } from '../hooks/useStored'
import { Badge, ErrorBox, Field, PageTitle } from '../components/ui'
import type { Dataset } from '../types'
import { faultNotebookRecord, findFaultNotebookDataset } from '../faultReference'

type NotebookStatus = { ready: boolean; url: string; row_limit: number; storage: string; notebooks: string[] }
type Source = { name: string; bytes: number; format: string; registered: boolean }
type LiteWindow = Window & {
  jupyterapp?: {
    started: Promise<void>
    restored: Promise<void>
    commands: { execute: (id: string, args: Record<string, unknown>) => Promise<unknown> }
    shell: {
      collapseLeft: () => void
      expandLeft: () => void
      currentWidget?: { context?: { ready: Promise<void> }; sessionContext?: { ready: Promise<void> } }
    }
  }
}

export default function Notebook() {
  const { data: status, error } = useQuery({
    queryKey: ['notebook-status'],
    queryFn: () => api<NotebookStatus>('/notebooks/status'),
  })
  const { data: datasets = [] } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => api<Dataset[]>('/datasets'),
  })
  const { data: sources = [] } = useQuery({
    queryKey: ['sources'],
    queryFn: () => api<Source[]>('/sources'),
    refetchInterval: 10000,
  })
  const [chosen, setChosen] = useStored('notebook-dataset', '')
  const [help, setHelp] = useState(false),
    [expanded, setExpanded] = useState(false),
    [copied, setCopied] = useState(false)
  const [notebookPath, setNotebookPath] = useState('01 - Getting started.ipynb'),
    [opening, setOpening] = useState(false)
  const [loaded, setLoaded] = useState(0),
    [copyError, setCopyError] = useState<unknown>()
  const iframe = useRef<HTMLIFrameElement>(null)
  useEffect(() => {
    if (!loaded) return
    let disposed = false,
      timer: number | undefined,
      observer: ResizeObserver | undefined,
      narrow: boolean | undefined
    function connect() {
      const element = iframe.current,
        app = (element?.contentWindow as LiteWindow | null)?.jupyterapp
      const panel = app?.shell.currentWidget
      if (!panel?.context) {
        timer = window.setTimeout(connect, 250)
        return
      }
      // Opening the URL's document restores the file browser after app.restored.
      // Wait for that document before applying the initial compact layout.
      void Promise.all([app!.started, app!.restored, panel.context.ready, panel.sessionContext?.ready]).then(
        () => {
          if (disposed || !element) return
          if ((element.contentWindow as LiteWindow | null)?.jupyterapp !== app) {
            connect()
            return
          }
          observer = new ResizeObserver(([entry]) => {
            const width = entry.contentRect.width
            if (!width) return // A hidden application tab must not change its notebook layout.
            const compact = width < 650
            if (compact && narrow !== true) app!.shell.collapseLeft()
            if (!compact && narrow === true) app!.shell.expandLeft()
            narrow = compact
          })
          observer.observe(element)
        },
      )
    }
    connect()
    return () => {
      disposed = true
      window.clearTimeout(timer)
      observer?.disconnect()
    }
  }, [loaded])
  const rawSources = sources.filter((s) => s.format === 'mapping_required' && !s.registered)
  const faultDataset = findFaultNotebookDataset(datasets)
  const source = rawSources.find((s) => 'source:' + s.name === chosen)
  const dataset = source
    ? undefined
    : datasets.find((d) => d.id === chosen) ||
      (notebookPath === 'fault_distance.ipynb' ? faultDataset : undefined) ||
      datasets[0]
  const snippet = source
    ? `from di_sources import read_csv\n\nframe = await read_csv(${JSON.stringify(source.name)})\nframe.head()`
    : dataset
      ? `from di_data import DIClient\ndi = DIClient()\n\n${
          dataset.format === 'recording'
            ? `page = await di.readings(${JSON.stringify(dataset.id)}, start=0, end=2)`
            : `assets = await di.assets(${JSON.stringify(dataset.id)})\npage = await di.readings(\n    ${JSON.stringify(dataset.id)},\n    asset_ids=[assets[0]["id"]], limit=10000\n)`
        }\nframe = di.to_frame(page)\nframe.head()`
      : 'Import a dataset in Datasets to get started.'
  async function openNotebook(path: string) {
    setOpening(true)
    setCopyError(undefined)
    try {
      const app = (iframe.current?.contentWindow as LiteWindow | null)?.jupyterapp
      if (!app) throw new Error('Wait for the notebook workspace to finish loading.')
      await Promise.all([app.started, app.restored])
      await app.commands.execute('docmanager:open', { path })
      setNotebookPath(path)
    } catch (e) {
      setCopyError(e)
    } finally {
      setOpening(false)
    }
  }
  async function copy() {
    try {
      await navigator.clipboard.writeText(snippet)
      setCopied(true)
      setCopyError(undefined)
    } catch (e) {
      setCopyError(e)
    }
  }
  return (
    <div className={expanded ? 'notebook-page notebook-expanded' : 'notebook-page'}>
      <PageTitle
        eyebrow="CUSTOM ANALYSIS"
        title="Notebook"
        description="Write Python, explore native measurements, and test your own ideas."
      >
        <button className="button secondary" onClick={() => setHelp(!help)} aria-expanded={help}>
          <BookOpen size={16} />
          {help ? 'Hide guide' : 'Data guide'}
        </button>
        {status?.ready ? (
          <a
            className="button secondary"
            href={'/notebooks/lab/index.html?path=' + encodeURIComponent(notebookPath)}
            target="_blank"
            rel="noreferrer"
          >
            <ExternalLink size={16} />
            Open in new tab
          </a>
        ) : null}
        <button
          className="icon-button"
          onClick={() => setExpanded(!expanded)}
          aria-label={expanded ? 'Exit notebook full screen' : 'Expand notebook'}
        >
          {expanded ? <Minimize2 size={19} /> : <Maximize2 size={19} />}
        </button>
      </PageTitle>
      <ErrorBox error={error || copyError} />
      <div className="notebook-data-bar">
        <Field label="Open notebook">
          <select
            value={notebookPath}
            disabled={!loaded || opening}
            onChange={(e) => void openNotebook(e.target.value)}
          >
            {(status?.notebooks || [notebookPath]).map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Notebook data source">
          <select
            value={source ? 'source:' + source.name : dataset?.id || ''}
            onChange={(e) => {
              setChosen(e.target.value)
              setCopied(false)
            }}
          >
            <optgroup label="Indexed datasets">
              {datasets.map((d) => (
                <option value={d.id} key={d.id}>
                  {d.name}
                </option>
              ))}
            </optgroup>
            {rawSources.length ? (
              <optgroup label="Source files · mapping unverified">
                {rawSources.map((s) => (
                  <option value={'source:' + s.name} key={s.name}>
                    {s.name}
                  </option>
                ))}
              </optgroup>
            ) : null}
          </select>
        </Field>
        {dataset || source ? (
          <>
            <div className="notebook-data-facts">
              <Badge tone={source || dataset?.synthetic ? 'amber' : 'green'}>
                {source ? 'Source table' : dataset?.synthetic ? 'Synthetic' : 'Measured'}
              </Badge>
              <span>
                {source
                  ? `${(source.bytes / 1024).toFixed(0)} KiB · mapping unverified`
                  : dataset?.format === 'recording'
                    ? `${number(dataset.sample_rate)} Hz`
                    : `${number(dataset?.assets)} ${dataset?.asset_level} profiles`}
              </span>
            </div>
            <button className="button secondary" onClick={copy}>
              {copied ? <Check size={15} /> : <Copy size={15} />}Copy data code
            </button>
            {source && source.bytes <= 20 * 1024 * 1024 ? (
              <a
                className="text-link notebook-source-download"
                href={`/api/v1/notebooks/files/${encodeURIComponent(source.name)}`}
              >
                <Download size={14} />
                Original file
              </a>
            ) : dataset && !dataset.synthetic && /\.(csv|parquet)$/i.test(dataset.source || '') ? (
              <a
                className="text-link notebook-source-download"
                href={`/api/v1/notebooks/sources/${dataset.id}`}
              >
                <Download size={14} />
                Original file
              </a>
            ) : null}
          </>
        ) : (
          <span className="muted">Import data to connect it to your notebook.</span>
        )}
      </div>
      {notebookPath === 'fault_distance.ipynb' ? (
        <div className="notebook-storage-note">
          <BookOpen size={15} />
          <span>
            This notebook starts with <strong>RECORD_NAME = "{faultNotebookRecord}"</strong>. The data source
            selector prepares copyable code; it does not change the notebook’s recording.{' '}
            {faultDataset ? (
              <Link className="text-link" to={'/events?dataset=' + encodeURIComponent(faultDataset.id)}>
                Open {faultNotebookRecord} in Event Lab
              </Link>
            ) : null}
          </span>
        </div>
      ) : null}
      {help ? (
        <div className="notebook-guide">
          <div>
            <h2>From dataset to DataFrame</h2>
            <p>
              Copy this code into a notebook cell and run it with <kbd>Shift</kbd> + <kbd>Enter</kbd>. The
              data source selector prepares the code; it does not change running notebook variables.
            </p>
            {source ? (
              <p>
                Source CSV reads preserve every supplied column and leave timestamps as text. Units, timezone,
                asset level, and labels remain unverified. This reader accepts files up to 20 MiB; import
                larger files for paged access.
              </p>
            ) : (
              <p>
                Use <code>di.iter_frames()</code> to process larger selections in pages. Reads preserve native
                samples, missing values, units, and timestamp precision.
              </p>
            )}
            <p>
              Starter notebooks cover hourly profiles, event research, and{' '}
              <strong>03 - Power anomaly exploration.ipynb</strong>. Open{' '}
              <strong>fault_distance.ipynb</strong> above to run the imported COMTRADE pipeline. Choose its
              recording in the first code cell and verify the line parameters before interpreting distances.
            </p>
          </div>
          <pre className="code">{snippet}</pre>
        </div>
      ) : null}
      <div className="notebook-storage-note">
        <Code2 size={15} />
        <span>
          Python runs in your browser. Notebooks save in this browser’s storage; download an{' '}
          <code>.ipynb</code> copy to keep or share your work.
        </span>
      </div>
      {status?.ready ? (
        <div className="notebook-frame-wrap">
          {!loaded ? (
            <div className="notebook-loading" role="status">
              Loading JupyterLite workspace…
            </div>
          ) : null}
          <iframe
            ref={iframe}
            className="notebook-frame"
            title="JupyterLite notebook workspace"
            src={status.url}
            allow="clipboard-read; clipboard-write"
            onLoad={() => setLoaded((count) => count + 1)}
          />
        </div>
      ) : !error ? (
        <div className="notice" role="status">
          {status
            ? 'The notebook workspace has not been built yet. Restart with Start-DI-Validator.cmd to prepare it, or run uv run python -m scripts.build_notebooks.'
            : 'Checking notebook workspace…'}
        </div>
      ) : null}
    </div>
  )
}
