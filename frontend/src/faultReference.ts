import type { Dataset } from './types'

// RECORD_NAME in the bundled fault_distance.ipynb configuration.
export const faultNotebookRecord = '1A_val1'

export function faultRecordName(dataset: Dataset): string {
  return (
    dataset.source
      ?.split(/[\\/]/)
      .pop()
      ?.replace(/\.cfg$/i, '') || dataset.name
  )
}

export function findFaultNotebookDataset(datasets: Dataset[]): Dataset | undefined {
  return datasets.find((d) => d.comtrade && faultRecordName(d) === faultNotebookRecord)
}
