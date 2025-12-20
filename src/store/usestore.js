import { create } from 'zustand'
import { persist } from 'zustand/middleware'

const generatecode = () => {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
  let code = ''
  for (let i = 0; i < 8; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return code
}

export const usestore = create(
  persist(
    (set, get) => ({
      progress: {},
      bookmarks: [],
      notes: {},
      quizscores: {},
      theme: 'light',
      synccode: null,
      showsyncmodal: false,

      initializesynccode: () => {
        const state = get()
        if (!state.synccode) {
          set({ synccode: generatecode() })
        }
      },

      markcomplete: (courseid, lessonid) => {
        set((state) => ({
          progress: {
            ...state.progress,
            [courseid]: {
              ...state.progress[courseid],
              [lessonid]: true
            }
          }
        }))
      },

      getcourseprogress: (courseid, totallessons) => {
        const state = get()
        const courseprogress = state.progress[courseid] || {}
        const completed = Object.values(courseprogress).filter(Boolean).length
        return Math.round((completed / totallessons) * 100)
      },

      islessonComplete: (courseid, lessonid) => {
        const state = get()
        return state.progress[courseid]?.[lessonid] || false
      },

      addbookmark: (courseid, lessonid) => {
        set((state) => {
          const key = `${courseid}:${lessonid}`
          if (state.bookmarks.includes(key)) return state
          return { bookmarks: [...state.bookmarks, key] }
        })
      },

      removebookmark: (courseid, lessonid) => {
        set((state) => ({
          bookmarks: state.bookmarks.filter(b => b !== `${courseid}:${lessonid}`)
        }))
      },

      isbookmarked: (courseid, lessonid) => {
        const state = get()
        return state.bookmarks.includes(`${courseid}:${lessonid}`)
      },

      savenote: (courseid, lessonid, note) => {
        set((state) => ({
          notes: {
            ...state.notes,
            [`${courseid}:${lessonid}`]: note
          }
        }))
      },

      getnote: (courseid, lessonid) => {
        const state = get()
        return state.notes[`${courseid}:${lessonid}`] || ''
      },

      savequizscore: (courseid, lessonid, score, total) => {
        set((state) => ({
          quizscores: {
            ...state.quizscores,
            [`${courseid}:${lessonid}`]: { score, total, date: Date.now() }
          }
        }))
      },

      getquizscore: (courseid, lessonid) => {
        const state = get()
        return state.quizscores[`${courseid}:${lessonid}`] || null
      },

      toggletheme: () => {
        set((state) => ({
          theme: state.theme === 'light' ? 'dark' : 'light'
        }))
      },

      setsyncmodalopen: (open) => {
        set({ showsyncmodal: open })
      },

      exportdata: () => {
        const state = get()
        const data = {
          progress: state.progress,
          bookmarks: state.bookmarks,
          notes: state.notes,
          quizscores: state.quizscores,
          synccode: state.synccode
        }
        return btoa(JSON.stringify(data))
      },

      importdata: (code) => {
        try {
          const data = JSON.parse(atob(code))
          set({
            progress: data.progress || {},
            bookmarks: data.bookmarks || [],
            notes: data.notes || {},
            quizscores: data.quizscores || {},
            synccode: data.synccode || get().synccode
          })
          return true
        } catch {
          return false
        }
      },

      resetprogress: () => {
        set({ progress: {}, quizscores: {} })
      }
    }),
    {
      name: 'ml-study-storage'
    }
  )
)
