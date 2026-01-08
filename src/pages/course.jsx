import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getcoursebyid } from '../data/courses'
import { useprogress } from '../hooks/useprogress'
import { useauth } from '../contexts/authcontext'
import ProgressBar from '../components/course/progressbar'
import Login from '../components/auth/login'

export default function course() {
  const { courseid } = useParams()
  const coursedata = getcoursebyid(courseid)
  const { progress, getcourseprogress } = useprogress()
  const { user, loading } = useauth()
  const [showlogin, setshowlogin] = useState(false)

  if (!coursedata) {
    return (
      <div className="text-center py-12">
        <h2 className="text-xl text-slate-600">Course not found</h2>
        <Link to="/" className="text-emerald-600 hover:underline mt-4 inline-block">
          Back to home
        </Link>
      </div>
    )
  }

  if (!loading && !user) {
    return (
      <div className="max-w-2xl mx-auto text-center py-16">
        <div className="w-24 h-24 bg-gradient-to-br from-emerald-100 to-emerald-200 rounded-full mx-auto mb-6 flex items-center justify-center">
          <svg className="w-12 h-12 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/>
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-slate-900 mb-3">Sign in to Access Courses</h2>
        <p className="text-slate-500 mb-8 max-w-md mx-auto">
          Create a free account to unlock all courses, track your progress, earn achievements, and get personalized learning recommendations.
        </p>
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <button
            onClick={() => setshowlogin(true)}
            className="px-8 py-3 bg-emerald-600 text-white font-medium rounded-xl hover:bg-emerald-700 transition-colors"
          >
            Sign in to Continue
          </button>
          <Link
            to="/"
            className="px-8 py-3 bg-slate-100 text-slate-700 font-medium rounded-xl hover:bg-slate-200 transition-colors"
          >
            Back to Home
          </Link>
        </div>
        <div className="mt-12 grid grid-cols-3 gap-6">
          <div className="text-center">
            <div className="w-10 h-10 bg-blue-100 rounded-lg mx-auto mb-2 flex items-center justify-center">
              <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
            </div>
            <p className="text-sm font-medium text-slate-700">Track Progress</p>
          </div>
          <div className="text-center">
            <div className="w-10 h-10 bg-amber-100 rounded-lg mx-auto mb-2 flex items-center justify-center">
              <svg className="w-5 h-5 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z"/>
              </svg>
            </div>
            <p className="text-sm font-medium text-slate-700">Earn Badges</p>
          </div>
          <div className="text-center">
            <div className="w-10 h-10 bg-purple-100 rounded-lg mx-auto mb-2 flex items-center justify-center">
              <svg className="w-5 h-5 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
              </svg>
            </div>
            <p className="text-sm font-medium text-slate-700">Get Certificates</p>
          </div>
        </div>
        {showlogin && <Login onclose={() => setshowlogin(false)} />}
      </div>
    )
  }

  const courseprogress = progress[courseid] || {}
  const overallprogress = getcourseprogress(courseid, coursedata.lessons.length)

  return (
    <div>
      <Link to="/" className="text-sm text-slate-500 hover:text-slate-700 mb-4 inline-flex items-center gap-1">
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        All Courses
      </Link>

      <div className="glass-card rounded-xl p-8 mb-8">
        <div className="flex items-start gap-4 mb-6">
          <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center text-emerald-600">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
          </div>
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-slate-900 mb-2">{coursedata.title}</h1>
            <p className="text-slate-600">{coursedata.description}</p>
          </div>
        </div>

        <div className="flex items-center gap-6 text-sm text-slate-500 mb-4">
          <span className={`font-medium px-2 py-1 rounded-full ${
            coursedata.difficulty === 'beginner' ? 'bg-green-50 text-green-600' :
            coursedata.difficulty === 'intermediate' ? 'bg-amber-50 text-amber-600' :
            'bg-red-50 text-red-600'
          }`}>
            {coursedata.difficulty}
          </span>
          <span>{coursedata.lessons.length} lessons</span>
        </div>

        <div className="flex items-center gap-4">
          <ProgressBar value={overallprogress} />
          <span className="text-sm text-slate-600 whitespace-nowrap">{overallprogress}% complete</span>
        </div>
      </div>

      {overallprogress === 100 && (
        <div className="bg-gradient-to-r from-emerald-500 to-emerald-600 rounded-xl p-6 mb-8 text-white">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 bg-white/20 rounded-xl flex items-center justify-center">
              <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
            </div>
            <div>
              <h3 className="text-xl font-bold">Course Completed!</h3>
              <p className="text-emerald-100">Great job! Continue with other courses to earn your certificate.</p>
            </div>
          </div>
        </div>
      )}

      <div className="glass-card rounded-xl overflow-hidden">
        <div className="p-4 border-b border-slate-200 bg-slate-50">
          <h2 className="font-semibold text-slate-900">Course Content</h2>
        </div>
        <div className="divide-y divide-slate-200">
          {coursedata.lessons.map((lesson, idx) => {
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
