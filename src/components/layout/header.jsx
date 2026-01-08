import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { courses } from '../../data/courses'
import { useauth } from '../../contexts/authcontext'
import { usestats } from '../../hooks/usestats'
import Login from '../auth/login'
import { getoptimizedurl } from '../../services/cloudinary'

export default function Header({ onmenuclick }) {
  const [query, setquery] = useState('')
  const [results, setresults] = useState([])
  const [showlogin, setshowlogin] = useState(false)
  const [showusermenu, setshowusermenu] = useState(false)
  const [searchfocused, setsearchfocused] = useState(false)
  const navigate = useNavigate()
  const { user, signout, avatar } = useauth()
  const { stats } = usestats()

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
    <header className="h-16 glass-navbar flex items-center px-4 md:px-8 gap-4 sticky top-0 z-30">
      <button
        onClick={onmenuclick}
        className="lg:hidden p-2 rounded-lg hover:bg-slate-100/80 text-slate-600"
      >
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      <div className={`relative flex-1 max-w-lg transition-all duration-200 ${searchfocused ? 'max-w-xl' : ''}`}>
        <div className={`flex items-center gap-2 px-4 py-2.5 rounded-xl border transition-all duration-200 ${
          searchfocused
            ? 'bg-white border-emerald-300 shadow-lg shadow-emerald-100/50 ring-2 ring-emerald-100'
            : 'bg-slate-50/80 border-slate-200/60 hover:bg-white hover:border-slate-300'
        }`}>
          <svg className={`w-5 h-5 transition-colors ${searchfocused ? 'text-emerald-500' : 'text-slate-400'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            value={query}
            onChange={handlesearch}
            onFocus={() => setsearchfocused(true)}
            onBlur={() => setsearchfocused(false)}
            placeholder="Search lessons, concepts, topics..."
            className="flex-1 bg-transparent text-sm focus:outline-none placeholder:text-slate-400"
          />
          {!searchfocused && !query && (
            <div className="hidden md:flex items-center gap-1 text-xs text-slate-400">
              <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded text-[10px] font-medium">⌘</kbd>
              <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded text-[10px] font-medium">K</kbd>
            </div>
          )}
        </div>
        {results.length > 0 && (
          <div className="absolute top-full left-0 right-0 mt-2 glass-card rounded-xl overflow-hidden z-50 shadow-xl">
            <div className="px-3 py-2 border-b border-slate-100 bg-slate-50/50">
              <span className="text-xs font-medium text-slate-500">{results.length} results found</span>
            </div>
            {results.map((result, idx) => (
              <button
                key={idx}
                onClick={() => handleselect(result)}
                className="w-full px-4 py-3 text-left hover:bg-emerald-50/50 border-b border-slate-100/50 last:border-0 flex items-center gap-3 transition-colors"
              >
                <div className="w-8 h-8 rounded-lg bg-emerald-100 flex items-center justify-center flex-shrink-0">
                  <svg className="w-4 h-4 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-900 truncate">{result.lessontitle}</p>
                  <p className="text-xs text-slate-500 truncate">{result.coursetitle}</p>
                </div>
                <svg className="w-4 h-4 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="flex-1" />

      <div className="flex items-center gap-2">
        {user && (
          <>
            <Link
              to="/achievements"
              className="hidden sm:flex items-center gap-2 px-3 py-2 rounded-xl hover:bg-slate-100/80 transition-colors group"
              title="Achievements"
            >
              <svg className="w-5 h-5 text-amber-500" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
              </svg>
              <span className="text-sm font-medium text-slate-600 group-hover:text-slate-900">{stats.xp} XP</span>
            </Link>

            {stats.currentStreak > 0 && (
              <div className="hidden sm:flex items-center gap-1.5 px-3 py-2 bg-gradient-to-r from-orange-50 to-amber-50 rounded-xl border border-orange-100">
                <span className="text-orange-500">🔥</span>
                <span className="text-sm font-semibold text-orange-600">{stats.currentStreak}</span>
              </div>
            )}
          </>
        )}

        {user ? (
          <div className="relative">
            <button
              onClick={() => setshowusermenu(!showusermenu)}
              className="flex items-center gap-2 px-2 py-1.5 rounded-xl hover:bg-slate-100/80 transition-colors"
            >
              {avatar ? (
                <img src={getoptimizedurl(avatar, { width: 72, height: 72 })} alt="" className="w-9 h-9 rounded-xl object-cover shadow-lg shadow-emerald-200/50" />
              ) : (
                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center text-white font-semibold text-sm shadow-lg shadow-emerald-200/50">
                  {user.email?.charAt(0).toUpperCase()}
                </div>
              )}
              <svg className="w-4 h-4 text-slate-400 hidden sm:block" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {showusermenu && (
              <div className="absolute right-0 top-full mt-2 w-56 glass-card rounded-xl py-2 z-50 shadow-xl">
                <div className="px-4 py-3 border-b border-slate-100">
                  <p className="text-sm font-semibold text-slate-900 truncate">{user.email}</p>
                  <p className="text-xs text-emerald-600 font-medium">Level {stats.level} • {stats.title}</p>
                </div>
                <div className="py-1">
                  <Link
                    to="/profile"
                    onClick={() => setshowusermenu(false)}
                    className="flex items-center gap-3 px-4 py-2.5 text-sm text-slate-700 hover:bg-slate-50 transition-colors"
                  >
                    <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                    View Profile
                  </Link>
                  <Link
                    to="/achievements"
                    onClick={() => setshowusermenu(false)}
                    className="flex items-center gap-3 px-4 py-2.5 text-sm text-slate-700 hover:bg-slate-50 transition-colors"
                  >
                    <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                    </svg>
                    Achievements
                  </Link>
                </div>
                <div className="border-t border-slate-100 pt-1">
                  <button
                    onClick={async () => {
                      await signout()
                      setshowusermenu(false)
                    }}
                    className="flex items-center gap-3 w-full px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 transition-colors"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                    </svg>
                    Sign out
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          <button
            onClick={() => setshowlogin(true)}
            className="flex items-center gap-2 px-5 py-2.5 btn-primary text-white rounded-xl text-sm font-medium"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
            <span>Sign in</span>
          </button>
        )}
      </div>

      {showlogin && <Login onclose={() => setshowlogin(false)} />}
    </header>
  )
}
