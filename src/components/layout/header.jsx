import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { courses } from '../../data/courses'
import { usestore } from '../../store/usestore'

export default function Header({ onmenuclick }) {
  const [query, setquery] = useState('')
  const [results, setresults] = useState([])
  const navigate = useNavigate()
  const setsyncmodalopen = usestore((state) => state.setsyncmodalopen)

  const handlesearch = (e) => {
    const value = e.target.value
    setquery(value)

    if (value.length < 2) {
      setresults([])
      return
    }

    const matches = []
    courses.forEach(course => {
      course.lessons.forEach(lesson => {
        if (
          lesson.title.toLowerCase().includes(value.toLowerCase()) ||
          lesson.concepts.some(c => c.toLowerCase().includes(value.toLowerCase()))
        ) {
          matches.push({
            courseid: course.id,
            coursetitle: course.title,
            lessonid: lesson.id,
            lessontitle: lesson.title
          })
        }
      })
    })

    setresults(matches.slice(0, 5))
  }

  const handleselect = (result) => {
    navigate(`/course/${result.courseid}/lesson/${result.lessonid}`)
    setquery('')
    setresults([])
  }

  return (
    <header className="h-16 bg-white border-b border-slate-200 flex items-center px-4 md:px-8 gap-4">
      <button
        onClick={onmenuclick}
        className="lg:hidden p-2 rounded-lg hover:bg-slate-100 text-slate-600"
      >
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      <div className="relative flex-1 max-w-md">
        <svg className="w-5 h-5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          type="text"
          value={query}
          onChange={handlesearch}
          placeholder="Search lessons, concepts..."
          className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
        />
        {results.length > 0 && (
          <div className="absolute top-full left-0 right-0 mt-2 bg-white border border-slate-200 rounded-lg shadow-lg overflow-hidden z-50">
            {results.map((result, idx) => (
              <button
                key={idx}
                onClick={() => handleselect(result)}
                className="w-full px-4 py-3 text-left hover:bg-slate-50 border-b border-slate-100 last:border-0"
              >
                <p className="text-sm font-medium text-slate-900">{result.lessontitle}</p>
                <p className="text-xs text-slate-500">{result.coursetitle}</p>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="flex-1" />

      <button
        onClick={() => setsyncmodalopen(true)}
        className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-lg text-sm font-medium hover:bg-emerald-700 transition-colors"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
        <span className="hidden sm:inline">Sync Progress</span>
        <span className="sm:hidden">Sync</span>
      </button>
    </header>
  )
}
