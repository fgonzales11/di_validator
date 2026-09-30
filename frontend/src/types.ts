export type Channel = { name: string; kind: string; unit: string; phase: string }
export type Dataset = {
  id: string
  name: string
  format: 'wide_ami' | 'recording'
  asset_level: string
  circuit: string
  assets: number
  readings: number
  rows: number
  valid_rows: number
  start: string
  end: string
  timezone: string
  channels: Channel[]
  quality: Record<string, number>
  synthetic: boolean
  sample_rate: number
  duration_seconds: number
  asset_id: string
  source: string
  source_checksum: string
  interval_seconds: number
  interval_position: string
  config: Record<string, unknown>
  clock_verified?: boolean
  clock_basis?: string
  comtrade?: {
    reported_start: string
    reported_trigger: string
    timing_note: string
    instrument_note: string
    source_id: string
  }
}
export type Asset = { id: string; source_id: string; circuit: string; level: string }
export type Job = {
  id: string
  kind: string
  status: string
  progress: number
  message: string
  created_at: string
  result?: { id: string; name: string }
  error?: string
  logs?: { time: string; message: string }[]
}
export type Event = {
  id: string
  asset_id: string
  kind: string
  start: number
  end: number
  emitted_at: number
}
export type FaultAnalysis = {
  segment: number
  start: number
  end: number
  status: string
  reason?: string
  onset?: number
  inception_source?: string
  fault_type_heuristic?: string
  padding_left_samples?: number
  padding_right_samples?: number
  tensor_shape?: number[]
  distance?: {
    status: string
    reason?: string
    estimated_distance_km?: number
    apparent_distance_km?: number
    measurement?: string
    r_apparent_ohm?: number
    x_apparent_ohm?: number
    distance_percent_of_line?: number
    fault_loop?: string
    within_line?: boolean
    uncompensated_ground_estimate?: boolean
    cycles_used?: number
    cycle_min_km?: number
    cycle_max_km?: number
    absolute_error_km?: number
    cycles?: { cycle: number; start_after_fault_ms: number; x_apparent_ohm: number; distance_km: number }[]
  }
  network_location?: NetworkLocation
}
export type NetworkCandidate = {
  distance_km: number
  distance_mi: number
  section_id: string
  phases: string
  model_resistance_ohm: number
  fault_resistance_ohm: number
  longitude: number | null
  latitude: number | null
}
export type NetworkLocation = {
  status: string
  reason?: string
  feeder?: string
  substation?: string
  model?: string
  nominal_kv?: number
  loop?: string
  loop_source?: string
  measurement?: string
  measured_resistance_ohm?: number
  measured_reactance_ohm?: number
  candidate_count?: number
  candidates?: NetworkCandidate[]
  modeled_reach_ohm?: number
  estimated_impedance_share?: number | null
  relay_location_mi?: number
  relay_location_km?: number
  modeled_mi?: number
  assumptions?: string[]
}
export type Annotation = {
  id: string
  dataset_id: string
  asset_id: string
  kind: string
  start: number
  end: number
  note: string
  partition: string
}
export type ModelResult = {
  model: string
  name: string
  cv: Record<string, number>
  cv_std: Record<string, number>
  test: Record<string, number>
  selected: boolean
  threshold: number
  confusion: number[][]
  curves: { fpr: number[]; tpr: number[]; precision: number[]; recall: number[] }
}
export type Experiment = {
  id: string
  kind: 'classification' | 'events'
  name: string
  target: string
  created_at: string
  interpretation: string
  models: ModelResult[]
  counts: Record<string, number>
  config: Record<string, unknown>
  selected_model: string
  effective_start: string
  effective_end: string
  label_policy: string
  split_id: string
  evidence_id: string
  dataset_ids: string[]
  metrics: Record<string, number | null>
  events: Event[]
  synthetic: boolean
  beats_baseline: boolean
  fault_analyses?: FaultAnalysis[]
}
export type Trace = {
  asset_id?: string
  channel?: string
  unit?: string
  x: (string | number)[]
  y: (number | string | null)[]
  z?: (number | null)[][]
}
export type WindowData = {
  traces: Trace[]
  unit: string
  timezone: string
  view: string
  aggregation: string
  start: number
  end: number
  origin: string
  sample_rate: number
}
