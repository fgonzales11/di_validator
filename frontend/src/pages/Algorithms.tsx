import { useQuery } from '@tanstack/react-query'
import { ArrowUpRight, Boxes, Code2, FlaskConical, Radio } from 'lucide-react'
import { api } from '../api'
import { Badge, ErrorBox, PageTitle, Panel } from '../components/ui'
type Algorithm = {
  id: string
  name: string
  family: string
  status: string
  sampling?: string
  required_channels?: string[]
  parameters?: string[]
  reason?: string
}
export default function Algorithms() {
  const { data: algorithms = [], error } = useQuery({
    queryKey: ['algorithms'],
    queryFn: () => api<Algorithm[]>('/algorithms'),
  })
  return (
    <>
      <PageTitle
        eyebrow="METHOD REGISTRY"
        title="Algorithms with a clear contract"
        description="Inspect input requirements and choose a method that fits your measurements."
      >
        <a className="button" href="/experiments">
          <FlaskConical size={16} />
          Create experiment
        </a>
      </PageTitle>
      <ErrorBox error={error} />
      <div className="algorithm-grid">
        {algorithms
          .filter((a) => a.status === 'ready')
          .map((a) => (
            <article className="algorithm-card" key={a.id}>
              <div className="algorithm-top">
                <span className="algorithm-icon">
                  {a.family === 'Events' ? <Radio size={21} /> : <Boxes size={21} />}
                </span>
                <Badge tone="green">Ready</Badge>
              </div>
              <span className="eyebrow">{a.family}</span>
              <h2>{a.name}</h2>
              <div className="algorithm-contract">
                <div>
                  <span>Sampling</span>
                  <strong>{a.sampling}</strong>
                </div>
                <div>
                  <span>Input</span>
                  <strong>{a.required_channels?.join(', ')}</strong>
                </div>
              </div>
              <a href={a.family === 'Events' ? '/events' : '/experiments'}>
                Configure a run
                <ArrowUpRight size={15} />
              </a>
            </article>
          ))}
      </div>
      <Panel
        title="Legacy model inventory"
        subtitle="Stored artifacts are cataloged separately from verified, executable adapters"
      >
        <div className="notice">
          Legacy models require their original environment, feature order, sampling assumptions, and
          compatible channels. They are not automatically applied to hourly net-energy profiles.
        </div>
        <div className="legacy-list">
          {algorithms
            .filter((a) => a.status === 'unverified')
            .map((a) => (
              <div key={a.id}>
                <Code2 size={17} />
                <span>{a.name}</span>
                <Badge tone="amber">Compatibility unverified</Badge>
              </div>
            ))}
        </div>
      </Panel>
      <div className="insight-grid">
        <div>
          <span className="eyebrow">PYTHON EXTENSIONS</span>
          <h3>One analysis package. Two ways to work.</h3>
          <p>
            Use the application to configure experiments, or import the same analysis package in a notebook.
            Adapter contracts and examples are documented in the project README.
          </p>
        </div>
        <div>
          <span className="eyebrow">REPRODUCIBILITY</span>
          <h3>The experiment carries its context.</h3>
          <p>
            Each completed run records source fingerprints, label revisions, split membership, algorithm
            settings, and environment versions.
          </p>
        </div>
      </div>
    </>
  )
}
