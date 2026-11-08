import { MODEL_NAME } from '../config'
import type { DocumentDescriptor, Passage, Prediction } from '../types'
import geometry from './sampleGeometry.json'

export const SAMPLE_DOCUMENT: DocumentDescriptor = {
  id: 'fictional-briefing',
  filename: 'fictional-briefing.pdf',
  source: '/fictional-briefing.pdf',
  mode: 'sample',
  pageCount: 3,
  label: 'Fictional demonstration document',
}

export const SAMPLE_PASSAGES = geometry satisfies Passage[]

export const SAMPLE_PREDICTIONS: Prediction[] = [
  {
    passageId: 'passage-1',
    page: 1,
    model: MODEL_NAME,
    text: 'The review team assessed that the unusual signal pattern was most likely caused by a scheduled atmospheric research flight operating beyond the monitored corridor.',
  },
  {
    passageId: 'passage-2',
    page: 1,
    model: MODEL_NAME,
    text: 'Field offices were directed to preserve the original observation logs, compare them with civilian flight records, and report any material discrepancy through normal analytical channels.',
  },
  {
    passageId: 'passage-3',
    page: 2,
    model: MODEL_NAME,
    text: 'Analysts found no evidence that the event represented a threat, but recommended a short follow-up study to improve coordination between regional monitoring stations.',
  },
  {
    passageId: 'passage-4',
    page: 2,
    model: MODEL_NAME,
    text: 'The working group proposed a shared notification procedure so that future scientific flights could be distinguished quickly from events requiring additional review.',
  },
  {
    passageId: 'passage-5',
    page: 3,
    model: MODEL_NAME,
    text: 'This fictional memorandum should be retained solely as a training example and must not be interpreted as describing actual agencies, operations, personnel, or historical events.',
  },
]
