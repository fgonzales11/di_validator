import React, { lazy, Suspense, useEffect, useState } from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import {
  Activity,
  ArrowUpRight,
  Beaker,
  BookOpen,
  Boxes,
  Database,
  FlaskConical,
  Menu,
  Network,
  PanelLeftClose,
  Radio,
  ShieldCheck,
  TrendingUp,
} from 'lucide-react'
import { api } from './api'
import { JobDrawer } from './components/JobDrawer'
import type { Job } from './types'
import './style.css'

const Datasets = lazy(() => import('./pages/Datasets'))
const Explorer = lazy(() => import('./pages/Explorer'))
const Experiments = lazy(() => import('./pages/Experiments'))
const EventLab = lazy(() => import('./pages/EventLab'))
const MeterLab = lazy(() => import('./pages/MeterLab'))
const Algorithms = lazy(() => import('./pages/Algorithms'))
const Notebook = lazy(() => import('./pages/Notebook'))
const Forecasting = lazy(() => import('./pages/Forecasting'))
const queryClient = new QueryClient({ defaultOptions: { queries: { staleTime: 3000, retry: 1 } } })
const links = [
  { to: '/', label: 'Datasets', icon: Database },
  { to: '/explorer', label: 'Explorer', icon: Activity },
  { to: '/algorithms', label: 'Algorithms', icon: Boxes },
  { to: '/experiments', label: 'Experiments', icon: FlaskConical },
  { to: '/events', label: 'Event Lab', icon: Radio },
  { to: '/meter-lab', label: 'Meter Lab', icon: Network },
  { to: '/forecasting', label: 'Forecasting', icon: TrendingUp },
  { to: '/notebook', label: 'Notebook', icon: BookOpen },
]
const hosted = import.meta.env.VITE_HOSTED === 'true'
function App() {
  const [menu, setMenu] = useState(false),
    [jobsOpen, setJobsOpen] = useState(false)
  const location = useLocation()
  const [notebookOpened, setNotebookOpened] = useState(location.pathname === '/notebook')
  useEffect(() => {
    if (location.pathname === '/notebook') setNotebookOpened(true)
  }, [location.pathname])
  const { data: jobs = [] } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => api<Job[]>('/jobs'),
    refetchInterval: 2000,
  })
  const active = jobs.filter((j) => ['queued', 'running'].includes(j.status)),
    running = active.find((j) => j.status === 'running')
  const page = links.find((l) => l.to === location.pathname)?.label || 'Workspace'
  return (
    <div className="app">
      <a href="#main" className="skip">
        Skip to content
      </a>
      <aside className={'sidebar ' + (menu ? 'open' : '')}>
        <a href="/" className="brand">
          <span className="brand-mark">
            <Network size={23} />
          </span>
          <span>
            DI <strong>Validator</strong>
            <small>INTELLIGENCE WORKBENCH</small>
          </span>
        </a>
        <div className="workspace-label">
          WORKSPACE <span>01</span>
        </div>
        <nav aria-label="Main navigation">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end onClick={() => setMenu(false)}>
              <Icon size={19} />
              <span>{label}</span>
              {label === 'Event Lab' ? <span className="nav-tag">LAB</span> : null}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-note">
          <div className="note-icon">
            <ShieldCheck size={20} />
          </div>
          <strong>{hosted ? 'Private cloud workspace.' : 'Your data stays here.'}</strong>
          <p>
            {hosted ? 'Persistent cloud storage.' : 'Local processing.'}
            <br />
            Reproducible results.
          </p>
          <span className="local-tag">
            <i />
            {hosted ? 'AUTHORIZED ACCESS' : 'ON THIS DEVICE'}
          </span>
        </div>
        <div className="sidebar-bottom">
          <a href="/docs" target="_blank" rel="noreferrer">
            <BookOpen size={16} />
            API reference
            <ArrowUpRight size={13} />
          </a>
          <span>
            DI Validator <b>v0.1</b>
          </span>
        </div>
      </aside>
      {menu ? (
        <button className="scrim" onClick={() => setMenu(false)} aria-label="Close navigation" />
      ) : null}
      <div className="workspace">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            onClick={() => setMenu(!menu)}
            aria-label="Toggle navigation"
          >
            {menu ? <PanelLeftClose size={20} /> : <Menu size={20} />}
          </button>
          <div className="breadcrumbs">
            Workspace <span>/</span>
            <strong>{page}</strong>
          </div>
          <div className="spacer" />
          <span className="engine">
            <i />
            {hosted ? 'Cloud engine' : 'Local engine'}
          </span>
          <button className="activity-button" onClick={() => setJobsOpen(true)}>
            <Activity size={17} />
            Activity{active.length ? <span className="count">{active.length}</span> : null}
          </button>
          <div className="avatar" title={hosted ? 'Cloud workspace' : 'Local workspace'}>
            DI
          </div>
        </header>
        {running ? (
          <div className="running-strip">
            <span className="pulse" />
            {running.message}
            <progress max={1} value={running.progress} />
            <button onClick={() => setJobsOpen(true)}>View job</button>
          </div>
        ) : null}
        <main id="main">
          <Suspense fallback={<div className="loading">Loading workspace…</div>}>
            <Routes>
              <Route path="/" element={<Datasets />} />
              <Route path="/explorer" element={<Explorer />} />
              <Route path="/algorithms" element={<Algorithms />} />
              <Route path="/experiments" element={<Experiments />} />
              <Route path="/events" element={<EventLab />} />
              <Route path="/meter-lab" element={<MeterLab />} />
              <Route path="/forecasting" element={<Forecasting />} />
              <Route path="/notebook" element={null} />
            </Routes>
            {notebookOpened ? (
              <div hidden={location.pathname !== '/notebook'}>
                <Notebook />
              </div>
            ) : null}
          </Suspense>
        </main>
        <footer className="workspace-footer">
          <span>
            <Beaker size={13} />
            Built for investigation. Every result has a source.
          </span>
          <span>{hosted ? 'PRIVATE CLOUD WORKSPACE' : 'LOCAL WORKSPACE · NO CLOUD CONNECTION'}</span>
        </footer>
      </div>
      {jobsOpen ? <JobDrawer onClose={() => setJobsOpen(false)} /> : null}
    </div>
  )
}
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
)
