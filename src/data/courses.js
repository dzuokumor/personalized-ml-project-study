import { lstmcourse } from './courses/lstm-sentiment'
import { bartcourse } from './courses/bart-summarization'
import { cnncourse } from './courses/cnn-classification'
import { randomforestcourse } from './courses/random-forest-regression'
import { mlopscourse } from './courses/mlops-deployment'
import { audioprocessingcourse } from './courses/audio-processing'
import { healthcarecourse } from './courses/healthcare-pipeline'
import { ragcourse } from './courses/rag-chatbot'
import { mathforml } from './courses/math-for-ml'
import { pythonforml } from './courses/python-for-ml'
import { coremlconcepts } from './courses/core-ml-concepts'
import { supervisedlearning } from './courses/supervised-learning'
import { unsupervisedlearning } from './courses/unsupervised-learning'
import { modelevaluation } from './courses/model-evaluation'
import { neuralnetworks } from './courses/neural-networks'
import { trainingdeepnetworks } from './courses/training-deep-networks'
import { cnnscomputervision } from './courses/cnns-computer-vision'
import { rnnslstms } from './courses/rnns-lstms'
import { attentiontransformers } from './courses/attention-transformers'
import { nlpapplications } from './courses/nlp-applications'
import { generativemodels } from './courses/generative-models'
import { mlopscurriculum } from './courses/mlops-curriculum'
import { capstoneprojects } from './courses/capstone-projects'

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

export const curriculumcourses = [
  mathforml,
  pythonforml,
  coremlconcepts,
  supervisedlearning,
  unsupervisedlearning,
  modelevaluation,
  neuralnetworks,
  trainingdeepnetworks,
  cnnscomputervision,
  rnnslstms,
  attentiontransformers,
  nlpapplications,
  generativemodels,
  mlopscurriculum,
  capstoneprojects
]

export const allcourses = [...courses, ...curriculumcourses]

export const getcoursebyid = (id) => {
  return allcourses.find(c => c.id === id)
}
