import { NavLink } from 'react-router-dom'
import { courses } from '../../data/courses'
import Icon from '../ui/icon'

export default function Sidebar({ isopen, onclose }) {
  return (
    <aside className={`w-64 bg-white border-r border-slate-200 fixed h-full overflow-y-auto z-50 transition-transform lg:translate-x-0 ${
      isopen ? 'translate-x-0' : '-translate-x-full'
    }`}>
      <div className="p-6 border-b border-slate-200 flex items-center justify-between">
        <NavLink to="/" className="flex items-center gap-3" onClick={onclose}>
          <div className="w-8 h-8 bg-emerald-700 rounded-lg flex items-center justify-center">
            <Icon name="brain" size={20} className="text-white" />
          </div>
          <span className="font-semibold text-slate-900">ML Study Hub</span>
        </NavLink>
        <button
          onClick={onclose}
          className="lg:hidden p-1 rounded-lg hover:bg-slate-100 text-slate-500"
        >
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <nav className="p-4">
        <div className="mb-6">
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-3 px-3">courses</p>
          <ul className="space-y-1">
            {courses.map(course => (
              <li key={course.id}>
                <NavLink
                  to={`/course/${course.id}`}
                  onClick={onclose}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                      isActive
                        ? 'bg-emerald-50 text-emerald-700 font-medium'
                        : 'text-slate-600 hover:bg-slate-100'
                    }`
                  }
                >
                  <Icon name={course.icon} size={22} />
                  <span className="truncate">{course.title}</span>
                </NavLink>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-3 px-3">resources</p>
          <ul className="space-y-1">
            <li>
              <NavLink
                to="/glossary"
                onClick={onclose}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                    isActive
                      ? 'bg-emerald-50 text-emerald-700 font-medium'
                      : 'text-slate-600 hover:bg-slate-100'
                  }`
                }
              >
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
                <span>Glossary</span>
              </NavLink>
            </li>
          </ul>
        </div>
      </nav>
    </aside>
  )
}
