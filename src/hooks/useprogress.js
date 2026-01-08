import { useState, useEffect, useCallback } from 'react'
import { supabase } from '../lib/supabase'
import { useauth } from '../contexts/authcontext'

export function useprogress() {
  const { user } = useauth()
  const [progress, setprogress] = useState({})
  const [loading, setloading] = useState(true)

  useEffect(() => {
    if (!user) {
      const saved = localStorage.getItem('ml-progress')
      if (saved) {
        try {
          setprogress(JSON.parse(saved))
        } catch (e) {
          setprogress({})
        }
      }
      setloading(false)
      return
    }

    const fetchprogress = async () => {
      const { data, error } = await supabase
        .from('user_progress')
        .select('course_id, lesson_id, completed')
        .eq('user_id', user.id)

      if (error) {
        console.error('Error fetching progress:', error)
        setloading(false)
        return
      }

      const progressmap = {}
      data.forEach(item => {
        if (!progressmap[item.course_id]) {
          progressmap[item.course_id] = {}
        }
        progressmap[item.course_id][item.lesson_id] = item.completed
      })
      setprogress(progressmap)
      setloading(false)
    }

    fetchprogress()
  }, [user])

  const markcomplete = useCallback(async (courseid, lessonid) => {
    const newprogress = {
      ...progress,
      [courseid]: {
        ...progress[courseid],
        [lessonid]: true
      }
    }
    setprogress(newprogress)

    if (!user) {
      localStorage.setItem('ml-progress', JSON.stringify(newprogress))
      return
    }

    const { error } = await supabase
      .from('user_progress')
      .upsert({
        user_id: user.id,
        course_id: courseid,
        lesson_id: lessonid,
        completed: true,
        completed_at: new Date().toISOString()
      }, {
        onConflict: 'user_id,course_id,lesson_id'
      })

    if (error) {
      console.error('Error saving progress:', error)
    }
  }, [user, progress])

  const iscomplete = useCallback((courseid, lessonid) => {
    return progress[courseid]?.[lessonid] === true
  }, [progress])

  const getcourseprogress = useCallback((courseid, totallessons) => {
    if (!progress[courseid]) return 0
    const completed = Object.values(progress[courseid]).filter(Boolean).length
    return Math.round((completed / totallessons) * 100)
  }, [progress])

  return {
    progress,
    loading,
    markcomplete,
    iscomplete,
    getcourseprogress
  }
}

export function usequizscores() {
  const { user } = useauth()
  const [scores, setscores] = useState({})

  useEffect(() => {
    if (!user) {
      const saved = localStorage.getItem('ml-quiz-scores')
      if (saved) {
        try {
          setscores(JSON.parse(saved))
        } catch (e) {
          setscores({})
        }
      }
      return
    }

    const fetchscores = async () => {
      const { data, error } = await supabase
        .from('quiz_scores')
        .select('course_id, lesson_id, score, total, best_score')
        .eq('user_id', user.id)

      if (error) {
        console.error('Error fetching scores:', error)
        return
      }

      const scoresmap = {}
      data.forEach(item => {
        const key = `${item.course_id}:${item.lesson_id}`
        scoresmap[key] = {
          score: item.score,
          total: item.total,
          bestscore: item.best_score
        }
      })
      setscores(scoresmap)
    }

    fetchscores()
  }, [user])

  const savescore = useCallback(async (courseid, lessonid, score, total) => {
    const key = `${courseid}:${lessonid}`
    const existing = scores[key]
    const bestscore = existing ? Math.max(existing.bestscore || 0, score) : score

    const newscores = {
      ...scores,
      [key]: { score, total, bestscore }
    }
    setscores(newscores)

    if (!user) {
      localStorage.setItem('ml-quiz-scores', JSON.stringify(newscores))
      return
    }

    const { error } = await supabase
      .from('quiz_scores')
      .upsert({
        user_id: user.id,
        course_id: courseid,
        lesson_id: lessonid,
        score,
        total,
        best_score: bestscore
      }, {
        onConflict: 'user_id,course_id,lesson_id'
      })

    if (error) {
      console.error('Error saving score:', error)
    }
  }, [user, scores])

  const getscore = useCallback((courseid, lessonid) => {
    return scores[`${courseid}:${lessonid}`]
  }, [scores])

  return { scores, savescore, getscore }
}

export function usenotes() {
  const { user } = useauth()
  const [notes, setnotes] = useState({})

  useEffect(() => {
    if (!user) {
      const saved = localStorage.getItem('ml-notes')
      if (saved) {
        try {
          setnotes(JSON.parse(saved))
        } catch (e) {
          setnotes({})
        }
      }
      return
    }

    const fetchnotes = async () => {
      const { data, error } = await supabase
        .from('user_notes')
        .select('course_id, lesson_id, content')
        .eq('user_id', user.id)

      if (error) {
        console.error('Error fetching notes:', error)
        return
      }

      const notesmap = {}
      data.forEach(item => {
        const key = `${item.course_id}:${item.lesson_id}`
        notesmap[key] = item.content
      })
      setnotes(notesmap)
    }

    fetchnotes()
  }, [user])

  const savenote = useCallback(async (courseid, lessonid, content) => {
    const key = `${courseid}:${lessonid}`
    const newnotes = { ...notes, [key]: content }
    setnotes(newnotes)

    if (!user) {
      localStorage.setItem('ml-notes', JSON.stringify(newnotes))
      return
    }

    const { error } = await supabase
      .from('user_notes')
      .upsert({
        user_id: user.id,
        course_id: courseid,
        lesson_id: lessonid,
        content,
        updated_at: new Date().toISOString()
      }, {
        onConflict: 'user_id,course_id,lesson_id'
      })

    if (error) {
      console.error('Error saving note:', error)
    }
  }, [user, notes])

  const getnote = useCallback((courseid, lessonid) => {
    return notes[`${courseid}:${lessonid}`] || ''
  }, [notes])

  return { notes, savenote, getnote }
}

export function usebookmarks() {
  const { user } = useauth()
  const [bookmarks, setbookmarks] = useState([])

  useEffect(() => {
    if (!user) {
      const saved = localStorage.getItem('ml-bookmarks')
      if (saved) {
        try {
          setbookmarks(JSON.parse(saved))
        } catch (e) {
          setbookmarks([])
        }
      }
      return
    }

    const fetchbookmarks = async () => {
      const { data, error } = await supabase
        .from('bookmarks')
        .select('course_id, lesson_id')
        .eq('user_id', user.id)

      if (error) {
        console.error('Error fetching bookmarks:', error)
        return
      }

      setbookmarks(data.map(b => `${b.course_id}:${b.lesson_id}`))
    }

    fetchbookmarks()
  }, [user])

  const addbookmark = useCallback(async (courseid, lessonid) => {
    const key = `${courseid}:${lessonid}`
    if (bookmarks.includes(key)) return

    const newbookmarks = [...bookmarks, key]
    setbookmarks(newbookmarks)

    if (!user) {
      localStorage.setItem('ml-bookmarks', JSON.stringify(newbookmarks))
      return
    }

    const { error } = await supabase
      .from('bookmarks')
      .insert({
        user_id: user.id,
        course_id: courseid,
        lesson_id: lessonid
      })

    if (error) {
      console.error('Error adding bookmark:', error)
    }
  }, [user, bookmarks])

  const removebookmark = useCallback(async (courseid, lessonid) => {
    const key = `${courseid}:${lessonid}`
    const newbookmarks = bookmarks.filter(b => b !== key)
    setbookmarks(newbookmarks)

    if (!user) {
      localStorage.setItem('ml-bookmarks', JSON.stringify(newbookmarks))
      return
    }

    const { error } = await supabase
      .from('bookmarks')
      .delete()
      .eq('user_id', user.id)
      .eq('course_id', courseid)
      .eq('lesson_id', lessonid)

    if (error) {
      console.error('Error removing bookmark:', error)
    }
  }, [user, bookmarks])

  const isbookmarked = useCallback((courseid, lessonid) => {
    return bookmarks.includes(`${courseid}:${lessonid}`)
  }, [bookmarks])

  return { bookmarks, addbookmark, removebookmark, isbookmarked }
}
