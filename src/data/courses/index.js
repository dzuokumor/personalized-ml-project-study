import { lstmcourse } from './lstm-sentiment'
import { bartcourse } from './bart-summarization'
import { cnncourse } from './cnn-classification'
import { mathforml } from './math-for-ml'
import { pythonforml } from './python-for-ml'
import { coremlconcepts } from './core-ml-concepts'
import { supervisedlearning } from './supervised-learning'

export const courses = [
  lstmcourse,
  bartcourse,
  cnncourse
]

export const curriculumcourses = [
  mathforml,
  pythonforml,
  coremlconcepts,
  supervisedlearning
]

export const allcourses = [...courses, ...curriculumcourses]

export const getcoursebyid = (id) => {
  return allcourses.find(c => c.id === id)
}
