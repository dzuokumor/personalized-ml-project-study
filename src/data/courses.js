import { lstmcourse } from './courses/lstm-sentiment'
import { bartcourse } from './courses/bart-summarization'
import { cnncourse } from './courses/cnn-classification'
import { randomforestcourse } from './courses/random-forest-regression'
import { mlopscourse } from './courses/mlops-deployment'
import { audioprocessingcourse } from './courses/audio-processing'
import { healthcarecourse } from './courses/healthcare-pipeline'
import { ragcourse } from './courses/rag-chatbot'

export const courses = [
  ragcourse,
  lstmcourse,
  bartcourse,
  cnncourse,
  randomforestcourse,
  mlopscourse,
  audioprocessingcourse,
  healthcarecourse
]
