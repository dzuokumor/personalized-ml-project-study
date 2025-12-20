import { Link } from 'react-router-dom'
import { courses } from '../data/courses'
import { usestore } from '../store/usestore'
import ProgressBar from '../components/course/progressbar'
import Icon from '../components/ui/icon'

export default function Home() {
  const getcourseprogress = usestore((state) => state.getcourseprogress)

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">My ML Study Hub</h1>
        <p className="text-slate-600">Learn machine learning concepts through my own projects.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {courses.map(course => {
          const progress = getcourseprogress(course.id, course.lessons.length)
          return (
            <Link
              key={course.id}
              to={`/course/${course.id}`}
              className="bg-white border border-slate-200 rounded-xl p-6 hover:border-emerald-400 hover:shadow-md transition-all group"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="w-10 h-10 bg-slate-100 rounded-lg flex items-center justify-center text-slate-600 group-hover:bg-emerald-50 group-hover:text-emerald-700 transition-colors">
                  <Icon name={course.icon} size={22} />
                </div>
                <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                  course.difficulty === 'beginner' ? 'bg-slate-100 text-slate-600' :
                  course.difficulty === 'intermediate' ? 'bg-slate-200 text-slate-700' :
                  'bg-slate-300 text-slate-800'
                }`}>
                  {course.difficulty}
                </span>
              </div>

              <h2 className="text-lg font-semibold text-slate-900 mb-2 group-hover:text-emerald-700 transition-colors">
                {course.title}
              </h2>
              <p className="text-sm text-slate-500 mb-4 line-clamp-2">{course.description}</p>

              <div className="flex items-center justify-between text-sm text-slate-500 mb-3">
                <span>{course.lessons.length} lessons</span>
                <span>{progress}% complete</span>
              </div>
              <ProgressBar value={progress} size="sm" />
            </Link>
          )
        })}
      </div>

      <div className="mt-12 bg-slate-100 rounded-xl p-8">
        <h2 className="text-xl font-semibold text-slate-900 mb-4">About This Platform</h2>
        <p className="text-slate-600 mb-4">
          This is my personalized machine learning study hub, built from my actual project code.
          Each course covers the concepts, algorithms, and techniques I used in my projects,
          with real code examples from my work.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <div className="text-center">
            <p className="text-2xl font-bold text-emerald-700">{courses.length}</p>
            <p className="text-sm text-slate-500">Courses</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-emerald-700">
              {courses.reduce((acc, c) => acc + c.lessons.length, 0)}
            </p>
            <p className="text-sm text-slate-500">Lessons</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-emerald-700">7</p>
            <p className="text-sm text-slate-500">Projects</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-emerald-700">100+</p>
            <p className="text-sm text-slate-500">Code Examples</p>
          </div>
        </div>
      </div>
    </div>
  )
}
