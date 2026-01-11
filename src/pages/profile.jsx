import { useState, useEffect } from 'react'
import { useauth } from '../contexts/authcontext'
import { usestats } from '../hooks/usestats'
import { useprogress } from '../hooks/useprogress'
import { usestore } from '../store/usestore'
import { allcourses, curriculumcourses } from '../data/courses'
import { learningpaths } from '../data/learningpaths'
import { Link } from 'react-router-dom'
import Certificate from '../components/certificate/certificate'
import Avatarupload from '../components/profile/avatarupload'
import Publishmodal from '../components/github/publishmodal'
import { generatecardmarkdown, generatebadgemarkdown } from '../services/github'
import Statsbadge from '../components/profile/statsbadge'
import Statscard from '../components/profile/statscard'

const achievementicons = {
  star: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
    </svg>
  ),
  fire: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 23c-3.866 0-7-3.134-7-7 0-2.551 1.37-4.793 3.414-6.013L9 9l1.793-1.793A5.982 5.982 0 0112 6c1.657 0 3.156.672 4.243 1.757L18 9l.586.987C20.63 11.207 22 13.449 22 16c0 3.866-3.134 7-7 7h-3z"/>
    </svg>
  ),
  trophy: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2C9.243 2 7 4.243 7 7v1H4a1 1 0 00-1 1v2c0 2.206 1.794 4 4 4h.555A7.008 7.008 0 0011 18.92V21H8v2h8v-2h-3v-2.08A7.008 7.008 0 0016.445 15H17c2.206 0 4-1.794 4-4V9a1 1 0 00-1-1h-3V7c0-2.757-2.243-5-5-5zm-7 7h2v2c0 .737.166 1.435.457 2.063A2.001 2.001 0 015 11V9zm14 2a2.001 2.001 0 01-2.457 1.063c.291-.628.457-1.326.457-2.063V9h2v2z"/>
    </svg>
  ),
  check: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
    </svg>
  ),
  crown: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 1L9 9l-7-3 3 14h14l3-14-7 3-3-8zM5 20h14v2H5v-2z"/>
    </svg>
  ),
  flame: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 23c-3.866 0-7-3.134-7-7 0-2.551 1.37-4.793 3.414-6.013L9 9l1.793-1.793A5.982 5.982 0 0112 6c1.657 0 3.156.672 4.243 1.757L18 9l.586.987C20.63 11.207 22 13.449 22 16c0 3.866-3.134 7-7 7h-3z"/>
    </svg>
  ),
  calendar: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M19 4h-1V2h-2v2H8V2H6v2H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 16H5V10h14v10zm0-12H5V6h14v2z"/>
    </svg>
  ),
  medal: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2C9.243 2 7 4.243 7 7s2.243 5 5 5 5-2.243 5-5-2.243-5-5-5zm0 8c-1.654 0-3-1.346-3-3s1.346-3 3-3 3 1.346 3 3-1.346 3-3 3zm-1 4l1 10 1-10H11zm5.22 2.095l-3.94 1.14L15.5 22l2.72-5.905zm-8.44 0L5.5 22l3.22-4.765-3.94-1.14z"/>
    </svg>
  ),
  book: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M18 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zM6 4h5v8l-2.5-1.5L6 12V4z"/>
    </svg>
  ),
  map: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M15 5.1L9 3 3 5.02v16.2l6-2.33 6 2.1 6-2.02V2.77L15 5.1zm0 13.79l-6-2.11V5.11l6 2.11v11.68z"/>
    </svg>
  ),
  calculator: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 14h-2v-2h2v2zm0-4h-2v-2h2v2zm0-4h-2V7h2v2zm4 8h-2v-2h2v2zm0-4h-2v-2h2v2zm0-4h-2V7h2v2zm-8 8H6v-2h2v2zm0-4H6v-2h2v2zm0-4H6V7h2v2z"/>
    </svg>
  ),
  code: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M9.4 16.6L4.8 12l4.6-4.6L8 6l-6 6 6 6 1.4-1.4zm5.2 0l4.6-4.6-4.6-4.6L16 6l6 6-6 6-1.4-1.4z"/>
    </svg>
  ),
  brain: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2a9 9 0 00-9 9c0 4.17 2.84 7.67 6.69 8.69L12 22l2.31-2.31C18.16 18.67 21 15.17 21 11a9 9 0 00-9-9zm0 16c-3.87 0-7-3.13-7-7s3.13-7 7-7 7 3.13 7 7-3.13 7-7 7z"/>
    </svg>
  ),
  zap: (
    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
      <path d="M11 21h-1l1-7H7.5c-.58 0-.57-.32-.38-.66l.1-.16L13 4h1l-1 7h3.5c.49 0 .56.33.47.51l-.07.15L11 21z"/>
    </svg>
  )
}

export default function profile() {
  const { user, signout, avatar, githubconnected, connectgithubforrepos } = useauth()
  const {
    stats, levels, achievements, getunlockedachievements, getnextlevel, repairstats,
    canchangeusername, canchangefullname, getusernamecoolddown, getfullnamecooldown,
    setusername, setfullname
  } = usestats()
  const { progress, syncfrommlocal } = useprogress()
  const storeprogress = usestore((state) => state.progress)
  const quizscores = usestore((state) => state.quizscores)
  const [activeTab, setactiveTab] = useState('overview')
  const [showcertificate, setshowcertificate] = useState(false)
  const [copied, setcopied] = useState(null)
  const [cardtheme, setcardtheme] = useState('light')
  const [publishcourse, setpublishcourse] = useState(null)
  const [editingusername, seteditingusername] = useState(false)
  const [editingfullname, seteditingfullname] = useState(false)
  const [newusername, setnewusername] = useState('')
  const [newfullname, setnewfullname] = useState('')
  const [nameerror, setnameerror] = useState('')

  useEffect(() => {
    if (Object.keys(storeprogress).length > 0 || Object.keys(quizscores).length > 0) {
      repairstats(storeprogress, quizscores)
      syncfrommlocal(storeprogress)
    }
  }, [storeprogress, quizscores, repairstats, syncfrommlocal])

  const unlockedachievements = getunlockedachievements()
  const nextlevel = getnextlevel(stats.level)
  const xptonext = nextlevel ? nextlevel.xp - stats.xp : 0
  const progresspercent = nextlevel ? ((stats.xp - levels[stats.level - 1].xp) / (nextlevel.xp - levels[stats.level - 1].xp)) * 100 : 100

  const getcourseprogress = () => {
    return allcourses.map(course => {
      const completed = course.lessons.filter(l => progress[course.id]?.[l.id]).length
      return {
        ...course,
        completed,
        total: course.lessons.length,
        percent: Math.round((completed / course.lessons.length) * 100)
      }
    }).filter(c => c.completed > 0)
  }

  const getcurriculumprogress = () => {
    let totallessons = 0
    let completedlessons = 0
    curriculumcourses.forEach(course => {
      totallessons += course.lessons.length
      completedlessons += course.lessons.filter(l => progress[course.id]?.[l.id]).length
    })
    return {
      completed: completedlessons,
      total: totallessons,
      percent: totallessons > 0 ? Math.round((completedlessons / totallessons) * 100) : 0,
      iscomplete: completedlessons === totallessons && totallessons > 0
    }
  }

  const courseprogress = getcourseprogress()
  const curriculumprogress = getcurriculumprogress()
  const totalhours = learningpaths.reduce((sum, path) => sum + path.estimatedhours, 0)
  const completedcourses = courseprogress.filter(c => c.percent === 100)

  const handleSignout = async () => {
    await signout()
  }

  const handlesaveusername = () => {
    if (!newusername.trim()) {
      setnameerror('Username cannot be empty')
      return
    }
    if (newusername.trim().length < 3) {
      setnameerror('Username must be at least 3 characters')
      return
    }
    const result = setusername(newusername)
    if (result.success) {
      seteditingusername(false)
      setnewusername('')
      setnameerror('')
    } else {
      setnameerror(result.error)
    }
  }

  const handlesavefullname = () => {
    if (!newfullname.trim()) {
      setnameerror('Full name cannot be empty')
      return
    }
    const result = setfullname(newfullname)
    if (result.success) {
      seteditingfullname(false)
      setnewfullname('')
      setnameerror('')
    } else {
      setnameerror(result.error)
    }
  }

  const copytoClipboard = async (text, id) => {
    await navigator.clipboard.writeText(text)
    setcopied(id)
    setTimeout(() => setcopied(null), 2000)
  }

  const displayname = stats.username || user?.email?.split('@')[0] || 'learner'
  const cardprops = {
    username: displayname,
    level: stats.level.toString(),
    xp: stats.xp.toString(),
    streak: stats.currentStreak.toString(),
    lessons: stats.lessonsCompleted.toString(),
    courses: completedcourses.length.toString(),
    theme: cardtheme
  }

  if (!user) {
    return (
      <div className="text-center py-12">
        <h2 className="text-xl text-slate-600">Please sign in to view your profile</h2>
      </div>
    )
  }

  return (
    <div className="max-w-5xl mx-auto">
      <div className="bg-gradient-to-br from-emerald-600 to-emerald-700 rounded-2xl p-8 mb-8 text-white relative overflow-hidden">
        <div className="absolute inset-0 opacity-10">
          <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
            <defs>
              <pattern id="dots" x="0" y="0" width="10" height="10" patternUnits="userSpaceOnUse">
                <circle cx="2" cy="2" r="1" fill="white"/>
              </pattern>
            </defs>
            <rect width="100" height="100" fill="url(#dots)"/>
          </svg>
        </div>
        <div className="relative z-10 flex items-start gap-6">
          <Avatarupload size="large" />
          <div className="flex-1">
            <h1 className="text-3xl font-bold mb-1">{stats.username || user.email?.split('@')[0]}</h1>
            <p className="text-emerald-100 mb-4">{stats.fullname || user.email}</p>
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-2">
                <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center">
                  <span className="text-lg font-bold">{stats.level}</span>
                </div>
                <div>
                  <p className="text-sm text-emerald-100">Level</p>
                  <p className="font-semibold">{stats.title}</p>
                </div>
              </div>
              <div>
                <p className="text-sm text-emerald-100">Total XP</p>
                <p className="text-xl font-bold">{stats.xp.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-emerald-100">Current Streak</p>
                <p className="text-xl font-bold">{stats.currentStreak} days</p>
              </div>
              <div>
                <p className="text-sm text-emerald-100">Achievements</p>
                <p className="text-xl font-bold">{unlockedachievements.length}/{achievements.length}</p>
              </div>
            </div>
            {nextlevel && (
              <div className="mt-4">
                <div className="flex justify-between text-sm text-emerald-100 mb-1">
                  <span>Progress to {nextlevel.title}</span>
                  <span>{xptonext} XP needed</span>
                </div>
                <div className="h-2 bg-white/20 rounded-full overflow-hidden">
                  <div className="h-full bg-white rounded-full transition-all duration-500" style={{ width: `${progresspercent}%` }}/>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="flex gap-2 mb-6 border-b border-slate-200 overflow-x-auto">
        {['overview', 'achievements', 'courses', 'github', 'settings'].map(tab => (
          <button
            key={tab}
            onClick={() => setactiveTab(tab)}
            className={`px-6 py-3 font-medium capitalize transition-colors relative whitespace-nowrap ${
              activeTab === tab ? 'text-emerald-600' : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            {tab === 'github' ? 'GitHub & Share' : tab}
            {activeTab === tab && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-600"/>
            )}
          </button>
        ))}
      </div>

      {activeTab === 'overview' && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <div className="glass-card rounded-xl p-6">
              <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mb-3">
                <svg className="w-6 h-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>
                </svg>
              </div>
              <p className="text-2xl font-bold text-slate-900">{stats.lessonsCompleted}</p>
              <p className="text-sm text-slate-500">Lessons Completed</p>
            </div>
            <div className="glass-card rounded-xl p-6">
              <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mb-3">
                <svg className="w-6 h-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
              </div>
              <p className="text-2xl font-bold text-slate-900">{stats.quizzesPassed}</p>
              <p className="text-sm text-slate-500">Quizzes Passed</p>
            </div>
            <div className="glass-card rounded-xl p-6">
              <div className="w-12 h-12 bg-yellow-100 rounded-xl flex items-center justify-center mb-3">
                <svg className="w-6 h-6 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"/>
                </svg>
              </div>
              <p className="text-2xl font-bold text-slate-900">{stats.perfectQuizzes}</p>
              <p className="text-sm text-slate-500">Perfect Scores</p>
            </div>
            <div className="glass-card rounded-xl p-6">
              <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mb-3">
                <svg className="w-6 h-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 18.657A8 8 0 016.343 7.343S7 9 9 10c0-2 .5-5 2.986-7C14 5 16.09 5.777 17.656 7.343A7.975 7.975 0 0120 13a7.975 7.975 0 01-2.343 5.657z"/>
                </svg>
              </div>
              <p className="text-2xl font-bold text-slate-900">{stats.longestStreak}</p>
              <p className="text-sm text-slate-500">Longest Streak</p>
            </div>
          </div>

          <div className="glass-card rounded-xl p-6 mb-8">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">ML Curriculum Progress</h3>
                <p className="text-sm text-slate-500">{curriculumprogress.completed} of {curriculumprogress.total} lessons completed</p>
              </div>
              <span className={`px-4 py-2 rounded-full text-sm font-semibold ${
                curriculumprogress.iscomplete ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600'
              }`}>
                {curriculumprogress.percent}%
              </span>
            </div>
            <div className="h-3 bg-slate-100 rounded-full overflow-hidden mb-4">
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full transition-all duration-500"
                style={{ width: `${curriculumprogress.percent}%` }}
              />
            </div>
            {curriculumprogress.iscomplete ? (
              <div className="bg-gradient-to-r from-emerald-500 to-teal-500 rounded-xl p-6 text-white">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 bg-white/20 rounded-xl flex items-center justify-center">
                      <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                      </svg>
                    </div>
                    <div>
                      <h4 className="text-xl font-bold">Curriculum Completed!</h4>
                      <p className="text-emerald-100">You've mastered the entire ML curriculum</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setshowcertificate(true)}
                    className="px-6 py-3 bg-white text-emerald-600 rounded-xl font-semibold hover:bg-emerald-50 transition-colors flex items-center gap-2"
                  >
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    Get Certificate
                  </button>
                </div>
              </div>
            ) : (
              <p className="text-sm text-slate-500">
                Complete all {curriculumcourses.length} courses in the ML curriculum to earn your certificate.
              </p>
            )}
          </div>
        </>
      )}

      {activeTab === 'achievements' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {achievements.map(achievement => {
            const unlocked = unlockedachievements.find(a => a.id === achievement.id)
            return (
              <div
                key={achievement.id}
                className={`glass-card rounded-xl p-6 flex items-center gap-4 transition-all ${
                  unlocked ? 'border-emerald-200 shadow-sm' : 'opacity-60'
                }`}
              >
                <div className={`w-14 h-14 rounded-xl flex items-center justify-center ${
                  unlocked ? 'bg-emerald-100 text-emerald-600' : 'bg-slate-100 text-slate-400'
                }`}>
                  {achievementicons[achievement.icon] || achievementicons.star}
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className={`font-semibold ${unlocked ? 'text-slate-900' : 'text-slate-500'}`}>
                      {achievement.title}
                    </h3>
                    {unlocked && (
                      <svg className="w-5 h-5 text-emerald-600" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
                      </svg>
                    )}
                  </div>
                  <p className="text-sm text-slate-500">{achievement.description}</p>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {activeTab === 'courses' && (
        <div className="space-y-4">
          {courseprogress.length === 0 ? (
            <div className="text-center py-12 glass-card rounded-xl">
              <p className="text-slate-500">No courses started yet</p>
              <Link to="/" className="text-emerald-600 hover:underline mt-2 inline-block">Browse courses</Link>
            </div>
          ) : (
            courseprogress.map(course => (
              <div key={course.id} className="glass-card rounded-xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-semibold text-slate-900">{course.title}</h3>
                    <p className="text-sm text-slate-500">{course.completed} of {course.total} lessons</p>
                  </div>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    course.percent === 100 ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600'
                  }`}>
                    {course.percent}%
                  </span>
                </div>
                <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-emerald-600 rounded-full transition-all duration-500"
                    style={{ width: `${course.percent}%` }}
                  />
                </div>
                <Link
                  to={`/course/${course.id}`}
                  className="inline-block mt-4 text-sm text-emerald-600 hover:underline"
                >
                  Continue learning
                </Link>
              </div>
            ))
          )}
        </div>
      )}

      {activeTab === 'github' && (
        <div className="space-y-6">
          <div className="glass-card rounded-xl p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">GitHub Connection</h3>
                <p className="text-sm text-slate-500">Connect GitHub to push projects and showcase your work</p>
              </div>
              {githubconnected ? (
                <span className="flex items-center gap-2 px-4 py-2 bg-emerald-50 text-emerald-700 rounded-lg text-sm font-medium">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
                  </svg>
                  Connected
                </span>
              ) : (
                <button
                  onClick={connectgithubforrepos}
                  className="flex items-center gap-2 px-4 py-2 bg-slate-900 text-white rounded-lg text-sm font-medium hover:bg-slate-800 transition-colors"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                  </svg>
                  Connect GitHub
                </button>
              )}
            </div>
          </div>

          <div className="glass-card rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-slate-900">Profile Card</h3>
              <div className="flex gap-2">
                <button
                  onClick={() => setcardtheme('light')}
                  className={`px-3 py-1 text-sm rounded-lg transition-colors ${cardtheme === 'light' ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
                >
                  Light
                </button>
                <button
                  onClick={() => setcardtheme('dark')}
                  className={`px-3 py-1 text-sm rounded-lg transition-colors ${cardtheme === 'dark' ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
                >
                  Dark
                </button>
              </div>
            </div>
            <p className="text-sm text-slate-500 mb-4">Add this card to your GitHub profile README to showcase your ML learning progress</p>
            <div className={`rounded-xl p-6 mb-4 flex justify-center ${cardtheme === 'dark' ? 'bg-slate-900' : 'bg-slate-50'}`}>
              <Statscard {...cardprops} />
            </div>
            <details className="group">
              <summary className="flex items-center justify-between cursor-pointer p-3 bg-slate-50 rounded-lg hover:bg-slate-100 transition-colors">
                <span className="text-sm font-medium text-slate-600 flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"/>
                  </svg>
                  View Markdown
                </span>
                <svg className="w-4 h-4 text-slate-400 transition-transform group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/>
                </svg>
              </summary>
              <div className="mt-3 p-4 bg-slate-900 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-slate-400 font-mono">markdown</span>
                  <button
                    onClick={() => copytoClipboard(generatecardmarkdown(cardprops), 'card')}
                    className="text-xs text-emerald-400 hover:text-emerald-300 font-medium flex items-center gap-1"
                  >
                    {copied === 'card' ? (
                      <>
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/></svg>
                        Copied!
                      </>
                    ) : (
                      <>
                        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
                        Copy
                      </>
                    )}
                  </button>
                </div>
                <code className="text-xs text-emerald-300 font-mono break-all block">{generatecardmarkdown(cardprops)}</code>
              </div>
            </details>
          </div>

          <div className="glass-card rounded-xl p-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-2">Achievement Badges</h3>
            <p className="text-sm text-slate-500 mb-4">Add individual badges to your README</p>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl relative">
                <Statsbadge type="level" value={stats.level.toString()} theme="light" />
                <details className="group ml-4 relative">
                  <summary className="cursor-pointer text-sm text-emerald-600 hover:text-emerald-700 font-medium flex items-center gap-1">
                    {copied === 'level' ? 'Copied!' : 'Copy'}
                    <svg className="w-4 h-4 transition-transform group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/>
                    </svg>
                  </summary>
                  <div className="absolute right-0 mt-2 p-3 bg-slate-900 rounded-lg shadow-xl z-10 min-w-[300px]">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-slate-400 font-mono">markdown</span>
                      <button onClick={() => copytoClipboard(generatebadgemarkdown('level', stats.level, 'light'), 'level')} className="text-xs text-emerald-400">Copy</button>
                    </div>
                    <code className="text-xs text-emerald-300 font-mono break-all">{generatebadgemarkdown('level', stats.level, 'light')}</code>
                  </div>
                </details>
              </div>
              <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl relative">
                <Statsbadge type="xp" value={stats.xp.toLocaleString()} theme="light" />
                <details className="group ml-4 relative">
                  <summary className="cursor-pointer text-sm text-emerald-600 hover:text-emerald-700 font-medium flex items-center gap-1">
                    {copied === 'xp' ? 'Copied!' : 'Copy'}
                    <svg className="w-4 h-4 transition-transform group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/>
                    </svg>
                  </summary>
                  <div className="absolute right-0 mt-2 p-3 bg-slate-900 rounded-lg shadow-xl z-10 min-w-[300px]">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-slate-400 font-mono">markdown</span>
                      <button onClick={() => copytoClipboard(generatebadgemarkdown('xp', stats.xp.toLocaleString(), 'light'), 'xp')} className="text-xs text-emerald-400">Copy</button>
                    </div>
                    <code className="text-xs text-emerald-300 font-mono break-all">{generatebadgemarkdown('xp', stats.xp.toLocaleString(), 'light')}</code>
                  </div>
                </details>
              </div>
              <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl relative">
                <Statsbadge type="streak" value={`${stats.currentStreak}`} theme="light" />
                <details className="group ml-4 relative">
                  <summary className="cursor-pointer text-sm text-emerald-600 hover:text-emerald-700 font-medium flex items-center gap-1">
                    {copied === 'streak' ? 'Copied!' : 'Copy'}
                    <svg className="w-4 h-4 transition-transform group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/>
                    </svg>
                  </summary>
                  <div className="absolute right-0 mt-2 p-3 bg-slate-900 rounded-lg shadow-xl z-10 min-w-[300px]">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-slate-400 font-mono">markdown</span>
                      <button onClick={() => copytoClipboard(generatebadgemarkdown('streak', `${stats.currentStreak} days`, 'light'), 'streak')} className="text-xs text-emerald-400">Copy</button>
                    </div>
                    <code className="text-xs text-emerald-300 font-mono break-all">{generatebadgemarkdown('streak', `${stats.currentStreak} days`, 'light')}</code>
                  </div>
                </details>
              </div>
              {completedcourses.map(course => (
                <div key={course.id} className="flex items-center justify-between p-4 bg-slate-50 rounded-xl relative">
                  <Statsbadge type="course" value={course.title.substring(0, 15)} theme="light" />
                  <details className="group ml-4 relative">
                    <summary className="cursor-pointer text-sm text-emerald-600 hover:text-emerald-700 font-medium flex items-center gap-1">
                      {copied === `course-${course.id}` ? 'Copied!' : 'Copy'}
                      <svg className="w-4 h-4 transition-transform group-open:rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7"/>
                      </svg>
                    </summary>
                    <div className="absolute right-0 mt-2 p-3 bg-slate-900 rounded-lg shadow-xl z-10 min-w-[300px]">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-slate-400 font-mono">markdown</span>
                        <button onClick={() => copytoClipboard(generatebadgemarkdown('course', course.title.substring(0, 20), 'light'), `course-${course.id}`)} className="text-xs text-emerald-400">Copy</button>
                      </div>
                      <code className="text-xs text-emerald-300 font-mono break-all">{generatebadgemarkdown('course', course.title.substring(0, 20), 'light')}</code>
                    </div>
                  </details>
                </div>
              ))}
            </div>
          </div>

          {completedcourses.length > 0 && (
            <div className="glass-card rounded-xl p-6">
              <h3 className="text-lg font-semibold text-slate-900 mb-4">Push Projects to GitHub</h3>
              <p className="text-sm text-slate-500 mb-4">Create repositories from your completed projects to showcase on GitHub</p>
              {!githubconnected ? (
                <div className="text-center py-8 bg-slate-50 rounded-xl">
                  <p className="text-slate-500 mb-4">Connect GitHub to push your projects</p>
                  <button
                    onClick={connectgithubforrepos}
                    className="inline-flex items-center gap-2 px-4 py-2 bg-slate-900 text-white rounded-lg text-sm font-medium hover:bg-slate-800 transition-colors"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                    </svg>
                    Connect GitHub
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  {completedcourses.map(course => (
                    <div key={course.id} className="flex items-center justify-between p-4 bg-slate-50 rounded-xl">
                      <div>
                        <h4 className="font-medium text-slate-900">{course.title}</h4>
                        <p className="text-sm text-slate-500">{course.total} lessons completed</p>
                      </div>
                      <button
                        onClick={() => setpublishcourse(course)}
                        className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-sm font-medium hover:bg-emerald-700 transition-colors flex items-center gap-2"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        Publish
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === 'settings' && (
        <div className="space-y-6">
          <div className="glass-card rounded-xl p-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-6">Profile Picture</h3>
            <div className="flex items-center gap-6">
              <Avatarupload size="small" />
              <div>
                <p className="text-sm text-slate-500">Click on the image to upload a new photo</p>
                <p className="text-xs text-slate-400 mt-1">JPG, PNG or GIF. Max 5MB.</p>
              </div>
            </div>
          </div>

          <div className="glass-card rounded-xl p-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-6">Profile Info</h3>
            {nameerror && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
                {nameerror}
              </div>
            )}
            <div className="space-y-4">
              <div className="flex items-center justify-between py-4 border-b border-slate-100">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <p className="font-medium text-slate-900">Username</p>
                    {!canchangeusername() && (
                      <span className="text-xs px-2 py-0.5 bg-amber-100 text-amber-700 rounded-full">
                        {getusernamecoolddown()} days left
                      </span>
                    )}
                  </div>
                  {editingusername ? (
                    <div className="flex items-center gap-2 mt-2">
                      <input
                        type="text"
                        value={newusername}
                        onChange={(e) => setnewusername(e.target.value)}
                        placeholder="Enter username"
                        className="flex-1 px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
                        maxLength={20}
                      />
                      <button
                        onClick={handlesaveusername}
                        className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-sm font-medium hover:bg-emerald-700"
                      >
                        Save
                      </button>
                      <button
                        onClick={() => { seteditingusername(false); setnewusername(''); setnameerror('') }}
                        className="px-4 py-2 bg-slate-100 text-slate-600 rounded-lg text-sm font-medium hover:bg-slate-200"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <p className="text-sm text-slate-500">{stats.username || 'Not set'}</p>
                  )}
                </div>
                {!editingusername && canchangeusername() && (
                  <button
                    onClick={() => { seteditingusername(true); setnewusername(stats.username || '') }}
                    className="text-sm text-emerald-600 hover:text-emerald-700 font-medium"
                  >
                    Edit
                  </button>
                )}
              </div>

              <div className="flex items-center justify-between py-4 border-b border-slate-100">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <p className="font-medium text-slate-900">Full Name</p>
                    {!canchangefullname() && (
                      <span className="text-xs px-2 py-0.5 bg-amber-100 text-amber-700 rounded-full">
                        {getfullnamecooldown()} days left
                      </span>
                    )}
                  </div>
                  {editingfullname ? (
                    <div className="flex items-center gap-2 mt-2">
                      <input
                        type="text"
                        value={newfullname}
                        onChange={(e) => setnewfullname(e.target.value)}
                        placeholder="Enter full name"
                        className="flex-1 px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
                        maxLength={50}
                      />
                      <button
                        onClick={handlesavefullname}
                        className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-sm font-medium hover:bg-emerald-700"
                      >
                        Save
                      </button>
                      <button
                        onClick={() => { seteditingfullname(false); setnewfullname(''); setnameerror('') }}
                        className="px-4 py-2 bg-slate-100 text-slate-600 rounded-lg text-sm font-medium hover:bg-slate-200"
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <p className="text-sm text-slate-500">{stats.fullname || 'Not set'}</p>
                  )}
                </div>
                {!editingfullname && canchangefullname() && (
                  <button
                    onClick={() => { seteditingfullname(true); setnewfullname(stats.fullname || '') }}
                    className="text-sm text-emerald-600 hover:text-emerald-700 font-medium"
                  >
                    Edit
                  </button>
                )}
              </div>
            </div>
            <p className="text-xs text-slate-400 mt-4">Username can be changed once every 7 days. Full name can be changed once every 30 days.</p>
          </div>

          <div className="glass-card rounded-xl p-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-6">Account Settings</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between py-4 border-b border-slate-100">
                <div>
                  <p className="font-medium text-slate-900">Email</p>
                  <p className="text-sm text-slate-500">{user.email}</p>
                </div>
              </div>
              <div className="flex items-center justify-between py-4 border-b border-slate-100">
                <div>
                  <p className="font-medium text-slate-900">Account Created</p>
                  <p className="text-sm text-slate-500">{new Date(user.created_at).toLocaleDateString()}</p>
                </div>
              </div>
              <div className="pt-4">
                <button
                  onClick={handleSignout}
                  className="px-6 py-2 bg-red-50 text-red-600 rounded-lg font-medium hover:bg-red-100 transition-colors"
                >
                  Sign Out
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {showcertificate && (
        <Certificate
          completiondate={new Date()}
          totalcourses={curriculumcourses.length}
          totalhours={totalhours}
          onclose={() => setshowcertificate(false)}
        />
      )}

      {publishcourse && (
        <Publishmodal course={publishcourse} onclose={() => setpublishcourse(null)} />
      )}
    </div>
  )
}

