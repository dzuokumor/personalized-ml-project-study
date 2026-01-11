import { useState, useEffect, useCallback, useRef } from 'react'
import { useauth } from '../contexts/authcontext'
import { syncservice } from '../lib/sync'

const xpvalues = {
  lessoncomplete: 10,
  quizpass: 20,
  quizperfect: 35,
  streakday: 5,
  coursecomplete: 100,
  pathcomplete: 250
}

const levels = [
  { level: 1, xp: 0, title: 'Novice' },
  { level: 2, xp: 50, title: 'Learner' },
  { level: 3, xp: 150, title: 'Student' },
  { level: 4, xp: 300, title: 'Practitioner' },
  { level: 5, xp: 500, title: 'Explorer' },
  { level: 6, xp: 750, title: 'Adept' },
  { level: 7, xp: 1050, title: 'Specialist' },
  { level: 8, xp: 1400, title: 'Expert' },
  { level: 9, xp: 1800, title: 'Master' },
  { level: 10, xp: 2250, title: 'Sage' },
  { level: 11, xp: 2750, title: 'Grandmaster' },
  { level: 12, xp: 3300, title: 'Legend' }
]

const achievements = [
  { id: 'first-lesson', title: 'First Steps', description: 'Complete your first lesson', icon: 'star', condition: (stats) => stats.lessonsCompleted >= 1 },
  { id: 'ten-lessons', title: 'Dedicated Learner', description: 'Complete 10 lessons', icon: 'fire', condition: (stats) => stats.lessonsCompleted >= 10 },
  { id: 'fifty-lessons', title: 'Knowledge Seeker', description: 'Complete 50 lessons', icon: 'trophy', condition: (stats) => stats.lessonsCompleted >= 50 },
  { id: 'first-quiz', title: 'Quiz Taker', description: 'Pass your first quiz', icon: 'check', condition: (stats) => stats.quizzesPassed >= 1 },
  { id: 'perfect-quiz', title: 'Perfect Score', description: 'Get 100% on a quiz', icon: 'crown', condition: (stats) => stats.perfectQuizzes >= 1 },
  { id: 'three-streak', title: 'On Fire', description: 'Maintain a 3-day streak', icon: 'flame', condition: (stats) => stats.currentStreak >= 3 },
  { id: 'seven-streak', title: 'Week Warrior', description: 'Maintain a 7-day streak', icon: 'calendar', condition: (stats) => stats.currentStreak >= 7 },
  { id: 'thirty-streak', title: 'Monthly Master', description: 'Maintain a 30-day streak', icon: 'medal', condition: (stats) => stats.currentStreak >= 30 },
  { id: 'first-course', title: 'Course Complete', description: 'Complete your first course', icon: 'book', condition: (stats) => stats.coursesCompleted >= 1 },
  { id: 'first-path', title: 'Pathfinder', description: 'Complete a learning path', icon: 'map', condition: (stats) => stats.pathsCompleted >= 1 },
  { id: 'math-master', title: 'Math Master', description: 'Complete Mathematics for ML', icon: 'calculator', condition: (stats) => stats.completedCourses?.includes('math-for-ml') },
  { id: 'python-pro', title: 'Python Pro', description: 'Complete Python for ML', icon: 'code', condition: (stats) => stats.completedCourses?.includes('python-for-ml') },
  { id: 'neural-ninja', title: 'Neural Ninja', description: 'Complete Neural Network Fundamentals', icon: 'brain', condition: (stats) => stats.completedCourses?.includes('neural-network-fundamentals') },
  { id: 'transformer-titan', title: 'Transformer Titan', description: 'Complete Attention & Transformers', icon: 'zap', condition: (stats) => stats.completedCourses?.includes('attention-transformers') }
]

export function usestats() {
  const { user } = useauth()
  const defaultstats = {
    xp: 0,
    level: 1,
    title: 'Novice',
    username: '',
    fullname: '',
    usernamechangedat: null,
    fullnamechangedat: null,
    currentStreak: 0,
    longestStreak: 0,
    lastActivityDate: null,
    lessonsCompleted: 0,
    quizzesPassed: 0,
    perfectQuizzes: 0,
    coursesCompleted: 0,
    pathsCompleted: 0,
    completedCourses: [],
    unlockedAchievements: [],
    totalTimeSpent: 0
  }
  const [stats, setstats] = useState(defaultstats)
  const [loading, setloading] = useState(true)
  const syncedref = useRef(false)
  const statsref = useRef(defaultstats)

  const getlevel = (xp) => {
    let current = levels[0]
    for (const level of levels) {
      if (xp >= level.xp) current = level
      else break
    }
    return current
  }

  const getnextlevel = (currentlevel) => {
    const idx = levels.findIndex(l => l.level === currentlevel)
    return idx < levels.length - 1 ? levels[idx + 1] : null
  }

  useEffect(() => {
    const loadstats = async () => {
      const saved = localStorage.getItem('ml-user-stats')
      let localstats = null
      if (saved) {
        try {
          localstats = JSON.parse(saved)
        } catch (e) {}
      }

      if (user && !syncedref.current) {
        syncedref.current = true
        const cloudstats = await syncservice.loadstats(user.id)
        if (cloudstats) {
          const merged = {
            xp: Math.max(localstats?.xp || 0, cloudstats.xp || 0),
            username: cloudstats.username || localstats?.username || '',
            fullname: cloudstats.fullname || localstats?.fullname || '',
            usernamechangedat: cloudstats.usernamechangedat || localstats?.usernamechangedat || null,
            fullnamechangedat: cloudstats.fullnamechangedat || localstats?.fullnamechangedat || null,
            lessonsCompleted: Math.max(localstats?.lessonsCompleted || 0, cloudstats.lessonsCompleted || 0),
            quizzesPassed: Math.max(localstats?.quizzesPassed || 0, cloudstats.quizzesPassed || 0),
            perfectQuizzes: Math.max(localstats?.perfectQuizzes || 0, cloudstats.perfectQuizzes || 0),
            currentStreak: Math.max(localstats?.currentStreak || 0, cloudstats.currentStreak || 0),
            longestStreak: Math.max(localstats?.longestStreak || 0, cloudstats.longestStreak || 0),
            lastActivityDate: localstats?.lastActivityDate || cloudstats.lastActivityDate,
            coursesCompleted: Math.max(localstats?.coursesCompleted || 0, cloudstats.coursesCompleted || 0),
            completedCourses: [...new Set([...(localstats?.completedCourses || []), ...(cloudstats.completedCourses || [])])],
            pathsCompleted: Math.max(localstats?.pathsCompleted || 0, cloudstats.pathsCompleted || 0),
            unlockedAchievements: [...new Set([...(localstats?.unlockedAchievements || []), ...(cloudstats.achievements || [])])]
          }
          const level = getlevel(merged.xp)
          const finalstats = { ...merged, level: level.level, title: level.title }
          statsref.current = finalstats
          setstats(finalstats)
          localStorage.setItem('ml-user-stats', JSON.stringify(finalstats))
          setloading(false)
          return
        }
      }

      if (localstats) {
        const level = getlevel(localstats.xp || 0)
        const finalstats = { ...localstats, level: level.level, title: level.title }
        statsref.current = finalstats
        setstats(finalstats)
      }
      setloading(false)
    }

    loadstats()
  }, [user])

  const savestats = useCallback((newstats) => {
    const level = getlevel(newstats.xp)
    const updated = { ...newstats, level: level.level, title: level.title }
    statsref.current = updated
    setstats(updated)
    localStorage.setItem('ml-user-stats', JSON.stringify(updated))
    if (user) {
      syncservice.savestats(user.id, updated)
    }
    return updated
  }, [user])

  const checkstreak = useCallback(() => {
    const today = new Date().toDateString()
    const lastactivity = stats.lastActivityDate

    if (!lastactivity) return

    const lastdate = new Date(lastactivity)
    const yesterday = new Date()
    yesterday.setDate(yesterday.getDate() - 1)

    if (lastdate.toDateString() === yesterday.toDateString()) {
      return
    } else if (lastdate.toDateString() !== today) {
      savestats({ ...stats, currentStreak: 0 })
    }
  }, [stats, savestats])

  const recordactivity = useCallback(() => {
    const today = new Date().toDateString()
    const current = statsref.current
    const lastactivity = current.lastActivityDate

    let newstreak = current.currentStreak

    if (!lastactivity || new Date(lastactivity).toDateString() !== today) {
      const yesterday = new Date()
      yesterday.setDate(yesterday.getDate() - 1)

      if (lastactivity && new Date(lastactivity).toDateString() === yesterday.toDateString()) {
        newstreak = current.currentStreak + 1
      } else if (!lastactivity || new Date(lastactivity).toDateString() !== today) {
        newstreak = 1
      }
    }

    const newstats = {
      ...current,
      currentStreak: newstreak,
      longestStreak: Math.max(current.longestStreak, newstreak),
      lastActivityDate: today,
      xp: current.xp + (newstreak > current.currentStreak ? xpvalues.streakday : 0)
    }

    return savestats(newstats)
  }, [savestats])

  const addxp = useCallback((amount, reason) => {
    const updated = recordactivity()
    return savestats({ ...updated, xp: updated.xp + amount })
  }, [recordactivity, savestats])

  const completelesson = useCallback((courseid, lessonid) => {
    const updated = addxp(xpvalues.lessoncomplete, 'lesson')
    return savestats({
      ...updated,
      lessonsCompleted: updated.lessonsCompleted + 1
    })
  }, [addxp, savestats])

  const passquiz = useCallback((score, total) => {
    const isperfect = score === total
    const xpgained = isperfect ? xpvalues.quizperfect : xpvalues.quizpass
    const updated = addxp(xpgained, 'quiz')

    return savestats({
      ...updated,
      quizzesPassed: updated.quizzesPassed + 1,
      perfectQuizzes: isperfect ? updated.perfectQuizzes + 1 : updated.perfectQuizzes
    })
  }, [addxp, savestats])

  const completecourse = useCallback((courseid) => {
    const updated = addxp(xpvalues.coursecomplete, 'course')
    return savestats({
      ...updated,
      coursesCompleted: updated.coursesCompleted + 1,
      completedCourses: [...(updated.completedCourses || []), courseid]
    })
  }, [addxp, savestats])

  const getunlockedachievements = useCallback(() => {
    return achievements.filter(a => a.condition(stats))
  }, [stats])

  const getnewachievements = useCallback((prevstats) => {
    const prevunlocked = achievements.filter(a => a.condition(prevstats)).map(a => a.id)
    const currentunlocked = achievements.filter(a => a.condition(stats)).map(a => a.id)
    return currentunlocked.filter(id => !prevunlocked.includes(id))
  }, [stats])

  const repairstats = useCallback((progress, quizscores) => {
    let lessonscount = 0
    let perfectcount = 0

    Object.keys(progress).forEach(courseid => {
      const courseprogress = progress[courseid]
      if (courseprogress && typeof courseprogress === 'object') {
        lessonscount += Object.values(courseprogress).filter(Boolean).length
      }
    })

    Object.values(quizscores).forEach(quiz => {
      if (quiz && quiz.total > 0 && quiz.score >= quiz.total) {
        perfectcount++
      }
    })

    const current = statsref.current
    if (current.lessonsCompleted !== lessonscount || current.perfectQuizzes !== perfectcount) {
      const repaired = {
        ...current,
        lessonsCompleted: Math.max(current.lessonsCompleted, lessonscount),
        perfectQuizzes: Math.max(current.perfectQuizzes, perfectcount)
      }
      savestats(repaired)
      return true
    }
    return false
  }, [savestats])

  const canchangeusername = useCallback(() => {
    const current = statsref.current
    if (!current.usernamechangedat) return true
    const lastchange = new Date(current.usernamechangedat)
    const now = new Date()
    const dayspassed = (now - lastchange) / (1000 * 60 * 60 * 24)
    return dayspassed >= 7
  }, [])

  const canchangefullname = useCallback(() => {
    const current = statsref.current
    if (!current.fullnamechangedat) return true
    const lastchange = new Date(current.fullnamechangedat)
    const now = new Date()
    const dayspassed = (now - lastchange) / (1000 * 60 * 60 * 24)
    return dayspassed >= 30
  }, [])

  const getusernamecoolddown = useCallback(() => {
    const current = statsref.current
    if (!current.usernamechangedat) return 0
    const lastchange = new Date(current.usernamechangedat)
    const now = new Date()
    const dayspassed = (now - lastchange) / (1000 * 60 * 60 * 24)
    return Math.max(0, Math.ceil(7 - dayspassed))
  }, [])

  const getfullnamecooldown = useCallback(() => {
    const current = statsref.current
    if (!current.fullnamechangedat) return 0
    const lastchange = new Date(current.fullnamechangedat)
    const now = new Date()
    const dayspassed = (now - lastchange) / (1000 * 60 * 60 * 24)
    return Math.max(0, Math.ceil(30 - dayspassed))
  }, [])

  const setusername = useCallback((newusername) => {
    if (!canchangeusername()) return { success: false, error: 'Username can only be changed once every 7 days' }
    const current = statsref.current
    const updated = {
      ...current,
      username: newusername.trim(),
      usernamechangedat: new Date().toISOString()
    }
    savestats(updated)
    return { success: true }
  }, [canchangeusername, savestats])

  const setfullname = useCallback((newfullname) => {
    if (!canchangefullname()) return { success: false, error: 'Full name can only be changed once every 30 days' }
    const current = statsref.current
    const updated = {
      ...current,
      fullname: newfullname.trim(),
      fullnamechangedat: new Date().toISOString()
    }
    savestats(updated)
    return { success: true }
  }, [canchangefullname, savestats])

  return {
    stats,
    loading,
    levels,
    achievements,
    xpvalues,
    getlevel,
    getnextlevel,
    addxp,
    completelesson,
    passquiz,
    completecourse,
    recordactivity,
    getunlockedachievements,
    getnewachievements,
    repairstats,
    canchangeusername,
    canchangefullname,
    getusernamecoolddown,
    getfullnamecooldown,
    setusername,
    setfullname
  }
}
