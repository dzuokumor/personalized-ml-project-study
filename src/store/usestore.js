import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { syncservice } from '../lib/sync'

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
      userid: null,
      synced: false,

      setuserid: (userid) => {
        set({ userid })
      },

      loadfromcloud: async (userid) => {
        if (!userid) return
        set({ userid })
        const data = await syncservice.loadall(userid)
        if (data.progress) {
          set((state) => ({
            progress: { ...state.progress, ...data.progress },
            synced: true
          }))
        }
        if (data.bookmarks) {
          set((state) => {
            const merged = [...new Set([...state.bookmarks, ...data.bookmarks])]
            return { bookmarks: merged }
          })
        }
        if (data.notes) {
          set((state) => ({ notes: { ...state.notes, ...data.notes } }))
        }
        if (data.quizscores) {
          set((state) => ({ quizscores: { ...state.quizscores, ...data.quizscores } }))
        }
      },

      initializesynccode: () => {
        const state = get()
        if (!state.synccode) {
          set({ synccode: generatecode() })
        }
      },

      markcomplete: (courseid, lessonid) => {
        const { userid } = get()
        set((state) => ({
          progress: {
            ...state.progress,
            [courseid]: {
              ...state.progress[courseid],
              [lessonid]: true
            }
          }
        }))
        if (userid) {
          syncservice.saveprogress(userid, courseid, lessonid)
        }
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
        const { userid } = get()
        set((state) => {
          const key = `${courseid}:${lessonid}`
          if (state.bookmarks.includes(key)) return state
          return { bookmarks: [...state.bookmarks, key] }
        })
        if (userid) {
          syncservice.addbookmark(userid, courseid, lessonid)
        }
      },

      removebookmark: (courseid, lessonid) => {
        const { userid } = get()
        set((state) => ({
          bookmarks: state.bookmarks.filter(b => b !== `${courseid}:${lessonid}`)
        }))
        if (userid) {
          syncservice.removebookmark(userid, courseid, lessonid)
        }
      },

      isbookmarked: (courseid, lessonid) => {
        const state = get()
        return state.bookmarks.includes(`${courseid}:${lessonid}`)
      },

      savenote: (courseid, lessonid, note) => {
        const { userid } = get()
        set((state) => ({
          notes: {
            ...state.notes,
            [`${courseid}:${lessonid}`]: note
          }
        }))
        if (userid) {
          syncservice.savenote(userid, courseid, lessonid, note)
        }
      },

      getnote: (courseid, lessonid) => {
        const state = get()
        return state.notes[`${courseid}:${lessonid}`] || ''
      },

      savequizscore: (courseid, lessonid, score, total) => {
        const { userid } = get()
        set((state) => ({
          quizscores: {
            ...state.quizscores,
            [`${courseid}:${lessonid}`]: { score, total, date: Date.now() }
          }
        }))
        if (userid) {
          syncservice.savequizscore(userid, courseid, lessonid, score, total)
        }
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
