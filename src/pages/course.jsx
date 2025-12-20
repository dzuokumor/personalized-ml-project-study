import { useParams, Link } from 'react-router-dom'
import { courses } from '../data/courses'
import { usestore } from '../store/usestore'
import ProgressBar from '../components/course/progressbar'
import Icon from '../components/ui/icon'

export default function Course() {
  const { courseid } = useParams()
  const course = courses.find(c => c.id === courseid)

  const progress = usestore((state) => state.progress)
  const getcourseprogress = usestore((state) => state.getcourseprogress)

  if (!course) {
    return (
      <div className="text-center py-12">
        <h2 className="text-xl text-slate-600">Course not found</h2>
        <Link to="/" className="text-blue-600 hover:underline mt-4 inline-block">
          Back to home
        </Link>
      </div>
    )
  }

  const courseprogress = progress[courseid] || {}
  const overallprogress = getcourseprogress(courseid, course.lessons.length)

  return (
    <div>
      <Link to="/" className="text-sm text-slate-500 hover:text-slate-700 mb-4 inline-flex items-center gap-1">
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        All Courses
      </Link>

      <div className="bg-white border border-slate-200 rounded-xl p-8 mb-8">
        <div className="flex items-start gap-4 mb-6">
          <div className="w-12 h-12 bg-slate-100 rounded-xl flex items-center justify-center text-slate-600">
            <Icon name={course.icon} size={26} />
          </div>
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-slate-900 mb-2">{course.title}</h1>
            <p className="text-slate-600">{course.description}</p>
          </div>
        </div>

        <div className="flex items-center gap-6 text-sm text-slate-500 mb-4">
          <span className={`font-medium px-2 py-1 rounded-full ${
            course.difficulty === 'beginner' ? 'bg-slate-100 text-slate-600' :
            course.difficulty === 'intermediate' ? 'bg-slate-200 text-slate-700' :
            'bg-slate-300 text-slate-800'
          }`}>
            {course.difficulty}
          </span>
          <span>{course.lessons.length} lessons</span>
          <span>Based on: {course.sourceproject}</span>
        </div>

        <div className="flex items-center gap-4">
          <ProgressBar value={overallprogress} />
          <span className="text-sm text-slate-600 whitespace-nowrap">{overallprogress}% complete</span>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div className="p-4 border-b border-slate-200 bg-slate-50">
          <h2 className="font-semibold text-slate-900">Course Content</h2>
        </div>
        <div className="divide-y divide-slate-200">
          {course.lessons.map((lesson, idx) => {
            const iscomplete = courseprogress[lesson.id]
            return (
              <Link
                key={lesson.id}
                to={`/course/${courseid}/lesson/${lesson.id}`}
                className="flex items-center gap-4 p-4 hover:bg-slate-50 transition-colors"
              >
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                  iscomplete
                    ? 'bg-emerald-600 text-white'
                    : 'bg-slate-100 text-slate-600'
                }`}>
                  {iscomplete ? (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    idx + 1
                  )}
                </div>
                <div className="flex-1">
                  <h3 className="font-medium text-slate-900">{lesson.title}</h3>
                  <p className="text-sm text-slate-500">{lesson.duration}</p>
                </div>
                <svg className="w-5 h-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </Link>
            )
          })}
        </div>
      </div>
    </div>
  )
}
