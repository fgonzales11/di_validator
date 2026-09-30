import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Activity, Check, ChevronDown, LoaderCircle, RefreshCw, X } from 'lucide-react'
import { api } from '../api'
import type { Job } from '../types'
import { Modal } from './Modal'
import { Badge, Empty } from './ui'

export function JobDrawer({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient(),
    [selected, setSelected] = useState<string>()
  const { data: jobs = [] } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => api<Job[]>('/jobs'),
    refetchInterval: 1500,
  })
  const detail = useQuery({
    queryKey: ['job', selected],
    queryFn: () => api<Job>('/jobs/' + selected),
    enabled: !!selected,
    refetchInterval: 2000,
  })
  async function action(job: Job, verb: string) {
    await api('/jobs/' + job.id + '/' + verb, {})
    void queryClient.invalidateQueries()
  }
  return (
    <Modal title="Job activity" onClose={onClose} wide>
      <div className="job-list">
        {jobs.length ? (
          jobs.map((job) => (
            <div className="job" key={job.id}>
              <div className="job-heading">
                <span className="job-symbol">
                  {job.status === 'completed' ? (
                    <Check size={18} />
                  ) : job.status === 'running' ? (
                    <LoaderCircle size={18} className="spin" />
                  ) : (
                    <Activity size={18} />
                  )}
                </span>
                <button className="text-button" onClick={() => setSelected(job.id)}>
                  {job.result?.name || job.kind}
                </button>
                <Badge
                  tone={job.status === 'failed' ? 'red' : job.status === 'completed' ? 'green' : 'neutral'}
                >
                  {job.status}
                </Badge>
                <div className="spacer" />
                {['queued', 'running'].includes(job.status) ? (
                  <button
                    className="icon-button"
                    onClick={() => action(job, 'cancel')}
                    aria-label="Cancel job"
                  >
                    <X size={16} />
                  </button>
                ) : (
                  <button className="icon-button" onClick={() => action(job, 'rerun')} aria-label="Rerun job">
                    <RefreshCw size={16} />
                  </button>
                )}
              </div>
              <p>{job.message || 'Waiting for worker'}</p>
              <progress value={job.progress} max={1} />
            </div>
          ))
        ) : (
          <Empty title="No jobs yet">Imports and experiments will appear here.</Empty>
        )}
      </div>
      {detail.data ? (
        <details open>
          <summary>
            Execution log <ChevronDown size={15} />
          </summary>
          <pre className="code">
            {detail.data.logs?.map((l) => l.time + ' ' + l.message).join('\n') || detail.data.message}
          </pre>
        </details>
      ) : null}
    </Modal>
  )
}
