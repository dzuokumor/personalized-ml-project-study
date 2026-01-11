import { Link } from 'react-router-dom'
import { courses } from '../data/courses'
import { learningpaths, pathcolors } from '../data/learningpaths'
import { usestats } from '../hooks/usestats'
import { useauth } from '../contexts/authcontext'
import Logo from '../components/ui/logo'

export default function home() {
  const { user } = useauth()
  const { stats, getlevel, getnextlevel, achievements, getunlockedachievements } = usestats()

  const currentlevel = getlevel(stats.xp)
  const nextlevel = getnextlevel(currentlevel.level)
  const unlockedachievements = getunlockedachievements()

  return (
    <div className="max-w-6xl mx-auto">
      <div className="bg-gradient-to-br from-emerald-600 via-emerald-700 to-teal-700 rounded-2xl p-4 sm:p-8 mb-6 sm:mb-8 text-white relative overflow-hidden">
        <div className="absolute top-0 right-0 w-32 sm:w-64 h-32 sm:h-64 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-24 sm:w-48 h-24 sm:h-48 bg-white/5 rounded-full translate-y-1/2 -translate-x-1/2" />

        <div className="relative z-10">
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-2xl sm:text-3xl font-bold mb-2">
                {user ? `Welcome back!` : 'Master Machine Learning'}
              </h1>
              <p className="text-emerald-100 text-sm sm:text-lg mb-4 sm:mb-6 max-w-xl">
                {user
                  ? `You're making great progress. Continue your learning journey and unlock new achievements.`
                  : 'From linear regression to transformers. Learn ML fundamentals with intuitive explanations and hands-on code.'}
              </p>
              <div className="flex flex-wrap gap-3">
                <Link
                  to="/course/math-for-ml"
                  className="inline-flex items-center gap-2 px-5 py-2.5 bg-white text-emerald-700 font-medium rounded-xl hover:bg-emerald-50 transition-colors"
                >
                  {user ? 'Continue Learning' : 'Start Learning'}
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </Link>
                <Link
                  to="/glossary"
                  className="inline-flex items-center gap-2 px-5 py-2.5 bg-emerald-600/50 text-white font-medium rounded-xl hover:bg-emerald-600/70 transition-colors border border-emerald-400/30"
                >
                  Browse Glossary
                </Link>
              </div>
            </div>

            <div className="hidden md:block">
              <Logo size={120} className="opacity-20" />
            </div>
          </div>

          {user && (
            <div className="mt-4 sm:mt-6 pt-4 sm:pt-6 border-t border-emerald-500/30 grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-6">
              <div>
                <p className="text-emerald-200 text-xs sm:text-sm mb-1">Level</p>
                <p className="text-lg sm:text-2xl font-bold">{currentlevel.level}</p>
                <p className="text-xs text-emerald-200 hidden sm:block">{currentlevel.title}</p>
              </div>
              <div>
                <p className="text-emerald-200 text-xs sm:text-sm mb-1">Total XP</p>
                <p className="text-lg sm:text-2xl font-bold">{stats.xp}</p>
              </div>
              <div>
                <p className="text-emerald-200 text-xs sm:text-sm mb-1">Lessons</p>
                <p className="text-lg sm:text-2xl font-bold">{stats.lessonsCompleted}</p>
              </div>
              <div>
                <p className="text-emerald-200 text-xs sm:text-sm mb-1">Streak</p>
                <p className="text-lg sm:text-2xl font-bold flex items-center gap-1">
                  {stats.currentStreak} {stats.currentStreak > 0 && '🔥'}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-slate-900">Learning Paths</h2>
          <span className="text-sm text-slate-500">{learningpaths.length} paths • 15 courses</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {learningpaths.map((path, idx) => {
            const colors = pathcolors[path.color]
            const isLocked = path.prerequisites && path.prerequisites.length > 0

            return (
              <div
                key={path.id}
                className={`glass-card rounded-xl p-5 card-hover ${isLocked ? 'opacity-75' : ''}`}
              >
                <div className="flex items-start gap-4 mb-4">
                  <div className={`w-12 h-12 rounded-xl ${colors.bg} ${colors.text} flex items-center justify-center flex-shrink-0`}>
                    <span className="text-xl font-bold">{idx + 1}</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-slate-900 mb-1">{path.title}</h3>
                    <p className="text-sm text-slate-500 line-clamp-2">{path.description}</p>
                  </div>
                </div>

                <div className="flex items-center justify-between text-sm mb-3">
                  <span className="text-slate-500">{path.courses.length} courses</span>
                  <span className="text-slate-500">{path.estimatedhours}h</span>
                </div>

                <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden mb-4">
                  <div className={`h-full ${colors.progress} rounded-full`} style={{ width: '0%' }} />
                </div>

                <div className="flex items-center justify-between">
                  <span className={`text-xs font-medium px-2 py-1 rounded-full ${
                    path.difficulty === 'beginner' ? 'bg-green-50 text-green-600' :
                    path.difficulty === 'intermediate' ? 'bg-amber-50 text-amber-600' :
                    'bg-red-50 text-red-600'
                  }`}>
                    {path.difficulty}
                  </span>

                  <Link
                    to={`/course/${path.courses[0]}`}
                    className={`text-sm font-medium ${colors.text} hover:underline flex items-center gap-1`}
                  >
                    Start Path
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </Link>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <div className="lg:col-span-2 glass-card rounded-xl p-6">
          <h2 className="text-lg font-bold text-slate-900 mb-4">Quick Start Courses</h2>
          <div className="space-y-3">
            {[
              { id: 'math-for-ml', title: 'Mathematics for ML', desc: 'Linear algebra, calculus, and probability foundations', time: '8 lessons', color: 'blue' },
              { id: 'python-for-ml', title: 'Python for ML', desc: 'NumPy, Pandas, and Scikit-learn essentials', time: '6 lessons', color: 'purple' },
              { id: 'core-ml-concepts', title: 'Core ML Concepts', desc: 'Supervised, unsupervised learning, and the ML pipeline', time: '7 lessons', color: 'emerald' }
            ].map(course => (
              <Link
                key={course.id}
                to={`/course/${course.id}`}
                className="flex items-center gap-4 p-4 rounded-xl hover:bg-slate-50 transition-colors group"
              >
                <div className={`w-10 h-10 rounded-lg bg-${course.color}-50 text-${course.color}-600 flex items-center justify-center flex-shrink-0`}>
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-medium text-slate-900 group-hover:text-emerald-700 transition-colors">{course.title}</h3>
                  <p className="text-sm text-slate-500 truncate">{course.desc}</p>
                </div>
                <div className="text-sm text-slate-400">{course.time}</div>
                <svg className="w-5 h-5 text-slate-300 group-hover:text-emerald-600 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </Link>
            ))}
          </div>
        </div>

        <div className="glass-card rounded-xl p-6">
          <h2 className="text-lg font-bold text-slate-900 mb-4">Achievements</h2>
          <div className="space-y-3">
            {achievements.slice(0, 4).map(achievement => {
              const isunlocked = unlockedachievements.some(a => a.id === achievement.id)
              return (
                <div
                  key={achievement.id}
                  className={`flex items-center gap-3 p-3 rounded-lg ${isunlocked ? 'bg-amber-50' : 'bg-slate-50 opacity-50'}`}
                >
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center ${isunlocked ? 'bg-amber-100 text-amber-600' : 'bg-slate-200 text-slate-400'}`}>
                    {isunlocked ? '⭐' : '🔒'}
                  </div>
                  <div>
                    <p className={`font-medium text-sm ${isunlocked ? 'text-slate-900' : 'text-slate-500'}`}>{achievement.title}</p>
                    <p className="text-xs text-slate-400">{achievement.description}</p>
                  </div>
                </div>
              )
            })}
          </div>
          <Link to="/achievements" className="block mt-4 text-center text-sm text-emerald-600 font-medium hover:underline">
            View All Achievements
          </Link>
        </div>
      </div>

      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-slate-900">Project Studies</h2>
          <span className="text-sm text-slate-500">Real ML projects with code</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {courses.map(course => (
            <Link
              key={course.id}
              to={`/course/${course.id}`}
              className="glass-card rounded-xl p-5 card-hover group"
            >
              <div className="flex items-start gap-3 mb-3">
                <div className="w-10 h-10 bg-slate-100 rounded-lg flex items-center justify-center text-slate-500 group-hover:bg-emerald-50 group-hover:text-emerald-600 transition-colors">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                  </svg>
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-slate-900 group-hover:text-emerald-700 transition-colors line-clamp-1">
                    {course.title}
                  </h3>
                  <p className="text-xs text-slate-400">{course.lessons.length} lessons</p>
                </div>
              </div>
              <p className="text-sm text-slate-500 line-clamp-2">{course.description}</p>
            </Link>
          ))}
        </div>
      </div>

      <div className="bg-gradient-to-r from-slate-800 to-slate-900 rounded-2xl p-4 sm:p-8 text-white">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 sm:gap-6 text-center md:text-left">
          <div>
            <h2 className="text-xl sm:text-2xl font-bold mb-2">Ready to master ML?</h2>
            <p className="text-slate-300 text-sm sm:text-base">
              Start with the foundations and work your way up to advanced topics like transformers and generative models.
            </p>
          </div>
          <Link
            to="/course/math-for-ml"
            className="w-full md:w-auto flex-shrink-0 inline-flex items-center justify-center gap-2 px-6 py-3 bg-emerald-500 text-white font-medium rounded-xl hover:bg-emerald-600 transition-colors"
          >
            Begin Your Journey
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
        </div>
      </div>
    </div>
  )
}
