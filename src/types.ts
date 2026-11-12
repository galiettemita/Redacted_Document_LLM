export interface PassageRect {
  x: number
  y: number
  width: number
  height: number
}

export interface Passage {
  id: string
  label: string
  page: number
  rects: PassageRect[]
}

export interface Prediction {
  passageId: string
  page: number
  model: string
  text: string
}

export type DocumentMode = 'sample' | 'uploaded'

export interface DocumentDescriptor {
  id: string
  filename: string
  source: string
  mode: DocumentMode
  pageCount?: number
  label?: string
  backendDocumentId?: string
}

export type PredictionPhase = 'idle' | 'predicting' | 'partial' | 'complete' | 'empty' | 'failed'
