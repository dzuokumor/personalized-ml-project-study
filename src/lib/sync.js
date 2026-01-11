import { supabase } from './supabase'

export const syncservice = {
  async loadprogress(userid) {
    const { data, error } = await supabase
      .from('user_progress')
      .select('course_id, lesson_id')
      .eq('user_id', userid)
      .eq('completed', true)

    if (error) {
      console.error('Failed to load progress:', error)
      return null
    }

    const progress = {}
    data?.forEach(row => {
      if (!progress[row.course_id]) progress[row.course_id] = {}
      progress[row.course_id][row.lesson_id] = true
    })
    return progress
  },

  async saveprogress(userid, courseid, lessonid) {
    const { error } = await supabase
      .from('user_progress')
      .upsert({
        user_id: userid,
        course_id: courseid,
        lesson_id: lessonid,
        completed: true,
        completed_at: new Date().toISOString()
      }, { onConflict: 'user_id,course_id,lesson_id' })

    if (error) console.error('Failed to save progress:', error)
    return !error
  },

  async loadbookmarks(userid) {
    const { data, error } = await supabase
      .from('bookmarks')
      .select('course_id, lesson_id')
      .eq('user_id', userid)

    if (error) {
      console.error('Failed to load bookmarks:', error)
      return null
    }

    return data?.map(row => `${row.course_id}:${row.lesson_id}`) || []
  },

  async addbookmark(userid, courseid, lessonid) {
    const { error } = await supabase
      .from('bookmarks')
      .upsert({
        user_id: userid,
        course_id: courseid,
        lesson_id: lessonid
      }, { onConflict: 'user_id,course_id,lesson_id' })

    if (error) console.error('Failed to add bookmark:', error)
    return !error
  },

  async removebookmark(userid, courseid, lessonid) {
    const { error } = await supabase
      .from('bookmarks')
      .delete()
      .eq('user_id', userid)
      .eq('course_id', courseid)
      .eq('lesson_id', lessonid)

    if (error) console.error('Failed to remove bookmark:', error)
    return !error
  },

  async loadnotes(userid) {
    const { data, error } = await supabase
      .from('user_notes')
      .select('course_id, lesson_id, content')
      .eq('user_id', userid)

    if (error) {
      console.error('Failed to load notes:', error)
      return null
    }

    const notes = {}
    data?.forEach(row => {
      notes[`${row.course_id}:${row.lesson_id}`] = row.content
    })
    return notes
  },

  async savenote(userid, courseid, lessonid, content) {
    const { error } = await supabase
      .from('user_notes')
      .upsert({
        user_id: userid,
        course_id: courseid,
        lesson_id: lessonid,
        content,
        updated_at: new Date().toISOString()
      }, { onConflict: 'user_id,course_id,lesson_id' })

    if (error) console.error('Failed to save note:', error)
    return !error
  },

  async loadquizscores(userid) {
    const { data, error } = await supabase
      .from('quiz_scores')
      .select('course_id, lesson_id, score, total, created_at')
      .eq('user_id', userid)

    if (error) {
      console.error('Failed to load quiz scores:', error)
      return null
    }

    const scores = {}
    data?.forEach(row => {
      const key = `${row.course_id}:${row.lesson_id}`
      if (!scores[key] || new Date(row.created_at) > new Date(scores[key].date)) {
        scores[key] = { score: row.score, total: row.total, date: new Date(row.created_at).getTime() }
      }
    })
    return scores
  },

  async savequizscore(userid, courseid, lessonid, score, total) {
    const { error } = await supabase
      .from('quiz_scores')
      .insert({
        user_id: userid,
        course_id: courseid,
        lesson_id: lessonid,
        score,
        total
      })

    if (error) console.error('Failed to save quiz score:', error)
    return !error
  },

  async loadstats(userid) {
    const { data, error } = await supabase
      .from('user_stats')
      .select('*')
      .eq('user_id', userid)
      .maybeSingle()

    if (error) {
      console.error('Failed to load stats:', error)
      return null
    }

    if (!data) {
      await supabase.from('user_stats').insert({ user_id: userid })
      return null
    }

    return {
      xp: data.xp || 0,
      level: data.level || 1,
      username: data.username || '',
      fullname: data.fullname || '',
      usernamechangedat: data.username_changed_at || null,
      fullnamechangedat: data.fullname_changed_at || null,
      lessonsCompleted: data.lessons_completed || 0,
      quizzesPassed: data.quizzes_passed || 0,
      perfectQuizzes: data.perfect_quizzes || 0,
      currentStreak: data.current_streak || 0,
      longestStreak: data.longest_streak || 0,
      lastActivityDate: data.last_activity_date,
      achievements: data.achievements || [],
      coursesCompleted: (data.achievements || []).filter(a => a && a.startsWith && a.startsWith('course:')).length,
      completedCourses: (data.achievements || []).filter(a => a && a.startsWith && a.startsWith('course:')).map(a => a.replace('course:', '')),
      pathsCompleted: 0
    }
  },

  async savestats(userid, stats) {
    const achievements = [
      ...(stats.unlockedAchievements || []),
      ...(stats.completedCourses || []).map(c => `course:${c}`)
    ]

    const { error } = await supabase
      .from('user_stats')
      .upsert({
        user_id: userid,
        xp: stats.xp || 0,
        level: stats.level || 1,
        username: stats.username || null,
        fullname: stats.fullname || null,
        username_changed_at: stats.usernamechangedat || null,
        fullname_changed_at: stats.fullnamechangedat || null,
        lessons_completed: stats.lessonsCompleted || 0,
        quizzes_passed: stats.quizzesPassed || 0,
        perfect_quizzes: stats.perfectQuizzes || 0,
        current_streak: stats.currentStreak || 0,
        longest_streak: stats.longestStreak || 0,
        last_activity_date: stats.lastActivityDate,
        achievements: [...new Set(achievements)],
        updated_at: new Date().toISOString()
      }, { onConflict: 'user_id' })

    if (error) console.error('Failed to save stats:', error)
    return !error
  },

  async loadall(userid) {
    const [progress, bookmarks, notes, quizscores, stats] = await Promise.all([
      this.loadprogress(userid),
      this.loadbookmarks(userid),
      this.loadnotes(userid),
      this.loadquizscores(userid),
      this.loadstats(userid)
    ])

    return { progress, bookmarks, notes, quizscores, stats }
  }
}
