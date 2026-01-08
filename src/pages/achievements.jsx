import { usestats } from '../hooks/usestats'
import { useauth } from '../contexts/authcontext'
import { Link } from 'react-router-dom'

const achievementicons = {
  star: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
    </svg>
  ),
  fire: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 23c-3.866 0-7-3.134-7-7 0-2.551 1.37-4.793 3.414-6.013L9 9l1.793-1.793A5.982 5.982 0 0112 6c1.657 0 3.156.672 4.243 1.757L18 9l.586.987C20.63 11.207 22 13.449 22 16c0 3.866-3.134 7-7 7h-3z"/>
    </svg>
  ),
  trophy: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2C9.243 2 7 4.243 7 7v1H4a1 1 0 00-1 1v2c0 2.206 1.794 4 4 4h.555A7.008 7.008 0 0011 18.92V21H8v2h8v-2h-3v-2.08A7.008 7.008 0 0016.445 15H17c2.206 0 4-1.794 4-4V9a1 1 0 00-1-1h-3V7c0-2.757-2.243-5-5-5z"/>
    </svg>
  ),
  check: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
    </svg>
  ),
  crown: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 1L9 9l-7-3 3 14h14l3-14-7 3-3-8zM5 20h14v2H5v-2z"/>
    </svg>
  ),
  flame: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 23c-3.866 0-7-3.134-7-7 0-2.551 1.37-4.793 3.414-6.013L9 9l1.793-1.793A5.982 5.982 0 0112 6c1.657 0 3.156.672 4.243 1.757L18 9l.586.987C20.63 11.207 22 13.449 22 16c0 3.866-3.134 7-7 7h-3z"/>
    </svg>
  ),
  calendar: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M19 4h-1V2h-2v2H8V2H6v2H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 16H5V10h14v10z"/>
    </svg>
  ),
  medal: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2C9.243 2 7 4.243 7 7s2.243 5 5 5 5-2.243 5-5-2.243-5-5-5zm0 8c-1.654 0-3-1.346-3-3s1.346-3 3-3 3 1.346 3 3-1.346 3-3 3zm-1 4l1 10 1-10H11z"/>
    </svg>
  ),
  book: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M18 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zM6 4h5v8l-2.5-1.5L6 12V4z"/>
    </svg>
  ),
  map: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M15 5.1L9 3 3 5.02v16.2l6-2.33 6 2.1 6-2.02V2.77L15 5.1zm0 13.79l-6-2.11V5.11l6 2.11v11.68z"/>
    </svg>
  ),
  calculator: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 14h-2v-2h2v2zm0-4h-2v-2h2v2z"/>
    </svg>
  ),
  code: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M9.4 16.6L4.8 12l4.6-4.6L8 6l-6 6 6 6 1.4-1.4zm5.2 0l4.6-4.6-4.6-4.6L16 6l6 6-6 6-1.4-1.4z"/>
    </svg>
  ),
  brain: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2a9 9 0 00-9 9c0 4.17 2.84 7.67 6.69 8.69L12 22l2.31-2.31C18.16 18.67 21 15.17 21 11a9 9 0 00-9-9z"/>
    </svg>
  ),
  zap: (
    <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
      <path d="M11 21h-1l1-7H7.5c-.58 0-.57-.32-.38-.66l.1-.16L13 4h1l-1 7h3.5c.49 0 .56.33.47.51l-.07.15L11 21z"/>
    </svg>
  )
}

const badgecolors = {
  bronze: { bg: 'bg-amber-100', border: 'border-amber-300', text: 'text-amber-700', glow: 'shadow-amber-200' },
  silver: { bg: 'bg-slate-100', border: 'border-slate-400', text: 'text-slate-700', glow: 'shadow-slate-300' },
  gold: { bg: 'bg-yellow-100', border: 'border-yellow-400', text: 'text-yellow-700', glow: 'shadow-yellow-300' },
  platinum: { bg: 'bg-cyan-100', border: 'border-cyan-400', text: 'text-cyan-700', glow: 'shadow-cyan-300' }
}

const getbadgetier = (achievement) => {
  if (['fifty-lessons', 'thirty-streak', 'transformer-titan'].includes(achievement.id)) return 'platinum'
  if (['ten-lessons', 'seven-streak', 'first-course', 'neural-ninja'].includes(achievement.id)) return 'gold'
  if (['perfect-quiz', 'three-streak', 'python-pro', 'math-master'].includes(achievement.id)) return 'silver'
  return 'bronze'
}

export default function achievementspage() {
  const { user } = useauth()
  const { stats, achievements, getunlockedachievements } = usestats()
  const unlockedachievements = getunlockedachievements()

  const categories = {
    learning: achievements.filter(a => ['first-lesson', 'ten-lessons', 'fifty-lessons'].includes(a.id)),
    quizzes: achievements.filter(a => ['first-quiz', 'perfect-quiz'].includes(a.id)),
    streaks: achievements.filter(a => ['three-streak', 'seven-streak', 'thirty-streak'].includes(a.id)),
    courses: achievements.filter(a => ['first-course', 'first-path', 'math-master', 'python-pro', 'neural-ninja', 'transformer-titan'].includes(a.id))
  }

  if (!user) {
    return (
      <div className="max-w-4xl mx-auto text-center py-16">
        <div className="w-24 h-24 bg-slate-100 rounded-full mx-auto mb-6 flex items-center justify-center">
          <svg className="w-12 h-12 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/>
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-slate-900 mb-2">Sign in to View Achievements</h2>
        <p className="text-slate-500 mb-6">Create an account to track your progress and earn achievements</p>
        <Link to="/" className="text-emerald-600 hover:underline">Back to Home</Link>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Achievements</h1>
        <p className="text-slate-500">
          {unlockedachievements.length} of {achievements.length} achievements unlocked
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="glass-card bg-gradient-to-br from-amber-50/80 to-amber-100/80 rounded-xl p-4 text-center">
          <div className="w-10 h-10 bg-amber-200 rounded-full mx-auto mb-2 flex items-center justify-center">
            <span className="text-amber-700 font-bold">B</span>
          </div>
          <p className="text-sm font-medium text-amber-800">Bronze</p>
          <p className="text-xs text-amber-600">{unlockedachievements.filter(a => getbadgetier(a) === 'bronze').length} earned</p>
        </div>
        <div className="glass-card bg-gradient-to-br from-slate-50/80 to-slate-100/80 rounded-xl p-4 text-center">
          <div className="w-10 h-10 bg-slate-200 rounded-full mx-auto mb-2 flex items-center justify-center">
            <span className="text-slate-700 font-bold">S</span>
          </div>
          <p className="text-sm font-medium text-slate-700">Silver</p>
          <p className="text-xs text-slate-500">{unlockedachievements.filter(a => getbadgetier(a) === 'silver').length} earned</p>
        </div>
        <div className="glass-card bg-gradient-to-br from-yellow-50/80 to-yellow-100/80 rounded-xl p-4 text-center">
          <div className="w-10 h-10 bg-yellow-200 rounded-full mx-auto mb-2 flex items-center justify-center">
            <span className="text-yellow-700 font-bold">G</span>
          </div>
          <p className="text-sm font-medium text-yellow-700">Gold</p>
          <p className="text-xs text-yellow-600">{unlockedachievements.filter(a => getbadgetier(a) === 'gold').length} earned</p>
        </div>
        <div className="glass-card bg-gradient-to-br from-cyan-50/80 to-cyan-100/80 rounded-xl p-4 text-center">
          <div className="w-10 h-10 bg-cyan-200 rounded-full mx-auto mb-2 flex items-center justify-center">
            <span className="text-cyan-700 font-bold">P</span>
          </div>
          <p className="text-sm font-medium text-cyan-700">Platinum</p>
          <p className="text-xs text-cyan-600">{unlockedachievements.filter(a => getbadgetier(a) === 'platinum').length} earned</p>
        </div>
      </div>

      {Object.entries(categories).map(([category, categoryachievements]) => (
        <div key={category} className="mb-8">
          <h2 className="text-lg font-semibold text-slate-900 mb-4 capitalize">{category}</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {categoryachievements.map(achievement => {
              const unlocked = unlockedachievements.find(a => a.id === achievement.id)
              const tier = getbadgetier(achievement)
              const colors = badgecolors[tier]

              return (
                <div
                  key={achievement.id}
                  className={`relative overflow-hidden rounded-xl border-2 p-6 transition-all ${
                    unlocked
                      ? `${colors.bg} ${colors.border} shadow-lg ${colors.glow}`
                      : 'bg-slate-50 border-slate-200'
                  }`}
                >
                  {unlocked && (
                    <div className="absolute top-2 right-2">
                      <svg className="w-6 h-6 text-emerald-600" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
                      </svg>
                    </div>
                  )}
                  <div className={`w-16 h-16 rounded-xl flex items-center justify-center mb-4 ${
                    unlocked ? colors.bg : 'bg-slate-200'
                  } border-2 ${unlocked ? colors.border : 'border-slate-300'}`}>
                    <div className={unlocked ? colors.text : 'text-slate-400'}>
                      {achievementicons[achievement.icon] || achievementicons.star}
                    </div>
                  </div>
                  <h3 className={`font-semibold mb-1 ${unlocked ? colors.text : 'text-slate-400'}`}>
                    {achievement.title}
                  </h3>
                  <p className={`text-sm ${unlocked ? 'text-slate-600' : 'text-slate-400'}`}>
                    {achievement.description}
                  </p>
                  <div className={`mt-3 inline-block px-2 py-1 rounded text-xs font-medium ${
                    unlocked ? `${colors.bg} ${colors.text}` : 'bg-slate-100 text-slate-400'
                  }`}>
                    {tier.charAt(0).toUpperCase() + tier.slice(1)}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}
