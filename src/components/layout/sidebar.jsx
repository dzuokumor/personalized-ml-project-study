import { NavLink, useLocation } from 'react-router-dom'
import { useState } from 'react'
import { courses } from '../../data/courses'
import { learningpaths, pathcolors } from '../../data/learningpaths'
import Logo from '../ui/logo'
import { usestats } from '../../hooks/usestats'
import { useauth } from '../../contexts/authcontext'
import { getoptimizedurl } from '../../services/cloudinary'

export default function sidebar({ isopen, onclose }) {
  const [expandedpath, setexpandedpath] = useState(null)
  const [showpaths, setshowpaths] = useState(true)
  const [showprojects, setshowprojects] = useState(false)
  const location = useLocation()
  const { stats, getlevel, getnextlevel } = usestats()
  const { user, avatar } = useauth()

  const currentlevel = getlevel(stats.xp)
  const nextlevel = getnextlevel(currentlevel.level)
  const xpprogress = nextlevel
    ? ((stats.xp - currentlevel.xp) / (nextlevel.xp - currentlevel.xp)) * 100
    : 100

  const togglepath = (pathid) => {
    setexpandedpath(expandedpath === pathid ? null : pathid)
  }

  return (
    <aside className={`w-72 glass-sidebar fixed h-full overflow-y-auto z-50 transition-transform lg:translate-x-0 ${
      isopen ? 'translate-x-0' : '-translate-x-full'
    }`}>
      <div className="absolute bottom-0 left-0 -translate-x-1/2 translate-y-1/4 pointer-events-none opacity-20">
        <svg
          width={320}
          height={320}
          viewBox="0 0 64 64"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <g stroke="#10b981" strokeWidth="1.5" opacity="0.6">
            <line x1="12" y1="16" x2="32" y2="10" />
            <line x1="12" y1="16" x2="32" y2="32" />
            <line x1="12" y1="16" x2="32" y2="54" />
            <line x1="12" y1="32" x2="32" y2="10" />
            <line x1="12" y1="32" x2="32" y2="32" />
            <line x1="12" y1="32" x2="32" y2="54" />
            <line x1="12" y1="48" x2="32" y2="10" />
            <line x1="12" y1="48" x2="32" y2="32" />
            <line x1="12" y1="48" x2="32" y2="54" />
            <line x1="32" y1="10" x2="52" y2="24" />
            <line x1="32" y1="32" x2="52" y2="24" />
            <line x1="32" y1="54" x2="52" y2="24" />
            <line x1="32" y1="10" x2="52" y2="40" />
            <line x1="32" y1="32" x2="52" y2="40" />
            <line x1="32" y1="54" x2="52" y2="40" />
          </g>
          <circle cx="12" cy="16" r="5" fill="#d1fae5" stroke="#10b981" strokeWidth="2" />
          <circle cx="12" cy="32" r="5" fill="#d1fae5" stroke="#10b981" strokeWidth="2" />
          <circle cx="12" cy="48" r="5" fill="#d1fae5" stroke="#10b981" strokeWidth="2" />
          <circle cx="32" cy="10" r="6" fill="#d1fae5" stroke="#059669" strokeWidth="2" />
          <circle cx="32" cy="32" r="6" fill="#d1fae5" stroke="#059669" strokeWidth="2" />
          <circle cx="32" cy="54" r="6" fill="#d1fae5" stroke="#059669" strokeWidth="2" />
          <circle cx="52" cy="24" r="5" fill="#d1fae5" stroke="#10b981" strokeWidth="2" />
          <circle cx="52" cy="40" r="5" fill="#d1fae5" stroke="#10b981" strokeWidth="2" />
          <circle cx="12" cy="16" r="2" fill="#10b981" />
          <circle cx="12" cy="32" r="2" fill="#10b981" />
          <circle cx="12" cy="48" r="2" fill="#10b981" />
          <circle cx="32" cy="10" r="2.5" fill="#059669" />
          <circle cx="32" cy="32" r="2.5" fill="#059669" />
          <circle cx="32" cy="54" r="2.5" fill="#059669" />
          <circle cx="52" cy="24" r="2" fill="#10b981" />
          <circle cx="52" cy="40" r="2" fill="#10b981" />
        </svg>
      </div>

      <div className="p-5 border-b border-slate-100/50 relative">
        <div className="flex items-center justify-between mb-4">
          <NavLink to="/" className="flex items-center gap-3" onClick={onclose}>
            <Logo size={40} />
            <div>
              <span className="font-bold text-slate-900 text-lg">Neuron</span>
              <span className="text-xs text-slate-400 block -mt-1">ML Learning Platform</span>
            </div>
          </NavLink>
          <button
            onClick={onclose}
            className="lg:hidden p-1.5 rounded-lg hover:bg-slate-100 text-slate-400"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {user && (
          <div className="bg-gradient-to-r from-emerald-50/80 to-teal-50/80 rounded-xl p-3 border border-emerald-100/60 backdrop-blur-sm">
            <div className="flex items-center gap-3 mb-3">
              {avatar ? (
                <img src={getoptimizedurl(avatar, { width: 80, height: 80 })} alt="" className="w-10 h-10 rounded-full object-cover ring-2 ring-emerald-200" />
              ) : (
                <div className="w-10 h-10 rounded-full bg-emerald-600 flex items-center justify-center text-white text-sm font-bold ring-2 ring-emerald-200">
                  {currentlevel.level}
                </div>
              )}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <p className="text-sm font-semibold text-slate-900 truncate">{stats.username || 'Learner'}</p>
                  <span className="text-[10px] px-1.5 py-0.5 bg-emerald-600 text-white rounded font-medium shrink-0">{currentlevel.title}</span>
                </div>
                {stats.fullname && (
                  <p className="text-xs text-slate-500 truncate">{stats.fullname}</p>
                )}
              </div>
            </div>

            <div className="flex items-center gap-2 mb-3">
              <div className="flex-1 flex items-center gap-1.5 px-2 py-1.5 bg-white/60 rounded-lg">
                <svg className="w-3.5 h-3.5 text-emerald-600" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                </svg>
                <span className="text-xs font-semibold text-slate-700">{stats.xp} XP</span>
              </div>
              <div className="flex items-center gap-1.5 px-2 py-1.5 bg-white/60 rounded-lg">
                <span className="text-sm">🔥</span>
                <span className="text-xs font-semibold text-slate-700">{stats.currentStreak}</span>
              </div>
            </div>

            {nextlevel && (
              <div>
                <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                  <span>Progress to {nextlevel.title}</span>
                  <span>{Math.round(xpprogress)}%</span>
                </div>
                <div className="h-1.5 bg-emerald-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                    style={{ width: `${xpprogress}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <nav className="p-4 relative">
        <NavLink
          to="/"
          onClick={onclose}
          className={({ isActive }) =>
            `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all mb-4 ${
              isActive
                ? 'bg-emerald-50 text-emerald-700'
                : 'text-slate-600 hover:bg-slate-50'
            }`
          }
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
          </svg>
          <span>Dashboard</span>
        </NavLink>

        <div className="mb-4">
          <button
            onClick={() => setshowpaths(!showpaths)}
            className="flex items-center justify-between w-full px-3 py-2 text-xs font-semibold text-slate-400 uppercase tracking-wider hover:text-slate-600"
          >
            <span>Learning Paths</span>
            <svg className={`w-4 h-4 transition-transform ${showpaths ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {showpaths && (
            <div className="mt-2 space-y-1">
              {learningpaths.map(path => {
                const colors = pathcolors[path.color]
                const isexpanded = expandedpath === path.id
                return (
                  <div key={path.id}>
                    <button
                      onClick={() => togglepath(path.id)}
                      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all ${
                        isexpanded ? colors.bg + ' ' + colors.text : 'text-slate-600 hover:bg-slate-50'
                      }`}
                    >
                      <div className={`w-8 h-8 rounded-lg ${colors.bg} ${colors.text} flex items-center justify-center`}>
                        {path.id === 'foundations' && (
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                          </svg>
                        )}
                        {path.id === 'classical-ml' && (
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                          </svg>
                        )}
                        {path.id === 'deep-learning' && (
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                          </svg>
                        )}
                        {path.id === 'sequence-nlp' && (
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                          </svg>
                        )}
                        {path.id === 'advanced-production' && (
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                          </svg>
                        )}
                      </div>
                      <div className="flex-1 text-left">
                        <p className="font-medium">{path.title}</p>
                        <p className="text-xs text-slate-400">{path.estimatedhours}h • {path.courses.length} courses</p>
                      </div>
                      <svg className={`w-4 h-4 text-slate-400 transition-transform ${isexpanded ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>

                    {isexpanded && (
                      <div className="ml-4 mt-1 pl-4 border-l-2 border-slate-100 space-y-1">
                        {path.courses.map((courseid, idx) => (
                          <NavLink
                            key={courseid}
                            to={`/course/${courseid}`}
                            onClick={onclose}
                            className={({ isActive }) =>
                              `flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                                isActive
                                  ? 'bg-slate-100 text-slate-900 font-medium'
                                  : 'text-slate-500 hover:text-slate-700 hover:bg-slate-50'
                              }`
                            }
                          >
                            <span className={`w-5 h-5 rounded-full ${colors.bg} ${colors.text} flex items-center justify-center text-xs font-medium`}>
                              {idx + 1}
                            </span>
                            <span className="truncate">{courseid.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}</span>
                          </NavLink>
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <div className="mb-4">
          <button
            onClick={() => setshowprojects(!showprojects)}
            className="flex items-center justify-between w-full px-3 py-2 text-xs font-semibold text-slate-400 uppercase tracking-wider hover:text-slate-600"
          >
            <span>Project Studies</span>
            <svg className={`w-4 h-4 transition-transform ${showprojects ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {showprojects && (
            <div className="mt-2 space-y-1">
              {courses.map(course => (
                <NavLink
                  key={course.id}
                  to={`/course/${course.id}`}
                  onClick={onclose}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3 py-2 rounded-xl text-sm transition-colors ${
                      isActive
                        ? 'bg-slate-100 text-slate-900 font-medium'
                        : 'text-slate-500 hover:bg-slate-50 hover:text-slate-700'
                    }`
                  }
                >
                  <div className="w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center flex-shrink-0">
                    <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                    </svg>
                  </div>
                  <span className="truncate flex-1 min-w-0">{course.title}</span>
                </NavLink>
              ))}
            </div>
          )}
        </div>

        <div className="pt-4 border-t border-slate-100">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-3 mb-2">Resources</p>
          <div className="space-y-1">
            <NavLink
              to="/glossary"
              onClick={onclose}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-xl text-sm transition-colors ${
                  isActive
                    ? 'bg-emerald-50 text-emerald-700 font-medium'
                    : 'text-slate-500 hover:bg-slate-50 hover:text-slate-700'
                }`
              }
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
              <span>ML Glossary</span>
            </NavLink>
            <NavLink
              to="/achievements"
              onClick={onclose}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-xl text-sm transition-colors ${
                  isActive
                    ? 'bg-emerald-50 text-emerald-700 font-medium'
                    : 'text-slate-500 hover:bg-slate-50 hover:text-slate-700'
                }`
              }
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
              </svg>
              <span>Achievements</span>
            </NavLink>
            {user && (
              <NavLink
                to="/profile"
                onClick={onclose}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2 rounded-xl text-sm transition-colors ${
                    isActive
                      ? 'bg-emerald-50 text-emerald-700 font-medium'
                      : 'text-slate-500 hover:bg-slate-50 hover:text-slate-700'
                  }`
                }
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
                <span>Profile</span>
              </NavLink>
            )}
          </div>
        </div>
      </nav>
    </aside>
  )
}
