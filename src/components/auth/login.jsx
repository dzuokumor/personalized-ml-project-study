import { useState } from 'react'
import { createPortal } from 'react-dom'
import { useauth } from '../../contexts/authcontext'

export default function login({ onclose }) {
  const [mode, setmode] = useState('signin')
  const [email, setemail] = useState('')
  const [password, setpassword] = useState('')
  const [error, seterror] = useState('')
  const [loading, setloading] = useState(false)
  const [message, setmessage] = useState('')

  const { signin, signup, signinwithgoogle, signinwithgithub } = useauth()

  const handlesubmit = async (e) => {
    e.preventDefault()
    seterror('')
    setmessage('')
    setloading(true)

    try {
      if (mode === 'signin') {
        const { error } = await signin(email, password)
        if (error) throw error
        onclose?.()
      } else {
        const { error } = await signup(email, password)
        if (error) throw error
        setmessage('Check your email for the confirmation link!')
      }
    } catch (err) {
      seterror(err.message)
    } finally {
      setloading(false)
    }
  }

  const handlegoogle = async () => {
    seterror('')
    const { data, error } = await signinwithgoogle()
    if (error) {
      console.error('Google OAuth error:', error)
      seterror(error.message)
    } else if (!data?.url) {
      seterror('Google OAuth not configured. Enable it in Supabase Dashboard → Authentication → Providers.')
    }
  }

  const handlegithub = async () => {
    seterror('')
    const { data, error } = await signinwithgithub()
    if (error) {
      console.error('GitHub OAuth error:', error)
      seterror(error.message)
    } else if (!data?.url) {
      seterror('GitHub OAuth not configured. Enable it in Supabase Dashboard → Authentication → Providers.')
    }
  }

  return createPortal(
    <div
      className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
      style={{ zIndex: 9999 }}
      onClick={(e) => e.target === e.currentTarget && onclose?.()}
    >
      <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full p-5 sm:p-8 relative max-h-[90vh] overflow-y-auto">
        <button
          onClick={onclose}
          className="absolute top-3 right-3 sm:top-4 sm:right-4 text-slate-400 hover:text-slate-600"
        >
          <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <h2 className="text-xl sm:text-2xl font-bold text-slate-900 mb-1 sm:mb-2 pr-8">
          {mode === 'signin' ? 'Welcome back' : 'Create account'}
        </h2>
        <p className="text-sm sm:text-base text-slate-500 mb-4 sm:mb-6">
          {mode === 'signin' ? 'Sign in to track your progress' : 'Start your ML learning journey'}
        </p>

        <div className="flex gap-2 sm:gap-3 mb-4 sm:mb-6">
          <button
            onClick={handlegoogle}
            className="flex-1 flex items-center justify-center gap-2 px-3 sm:px-4 py-2.5 sm:py-3 border border-slate-200 rounded-xl hover:bg-slate-50 transition-colors text-sm sm:text-base"
          >
            <svg className="w-4 h-4 sm:w-5 sm:h-5 shrink-0" viewBox="0 0 24 24">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
            </svg>
            Google
          </button>
          <button
            onClick={handlegithub}
            className="flex-1 flex items-center justify-center gap-2 px-3 sm:px-4 py-2.5 sm:py-3 border border-slate-200 rounded-xl hover:bg-slate-50 transition-colors text-sm sm:text-base"
          >
            <svg className="w-4 h-4 sm:w-5 sm:h-5 shrink-0" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
            </svg>
            GitHub
          </button>
        </div>

        <div className="relative mb-4 sm:mb-6">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-slate-200"></div>
          </div>
          <div className="relative flex justify-center text-xs sm:text-sm">
            <span className="px-2 bg-white text-slate-500">or continue with email</span>
          </div>
        </div>

        <form onSubmit={handlesubmit} className="space-y-3 sm:space-y-4">
          <div>
            <label className="block text-xs sm:text-sm font-medium text-slate-700 mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setemail(e.target.value)}
              className="w-full px-3 sm:px-4 py-2.5 sm:py-3 border border-slate-200 rounded-xl text-sm sm:text-base focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              placeholder="you@example.com"
              required
            />
          </div>
          <div>
            <label className="block text-xs sm:text-sm font-medium text-slate-700 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setpassword(e.target.value)}
              className="w-full px-3 sm:px-4 py-2.5 sm:py-3 border border-slate-200 rounded-xl text-sm sm:text-base focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              placeholder="••••••••"
              required
              minLength={6}
            />
          </div>

          {error && (
            <div className="p-2.5 sm:p-3 bg-red-50 border border-red-200 rounded-xl text-red-600 text-xs sm:text-sm">
              {error}
            </div>
          )}

          {message && (
            <div className="p-2.5 sm:p-3 bg-emerald-50 border border-emerald-200 rounded-xl text-emerald-600 text-xs sm:text-sm">
              {message}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 sm:py-3 bg-emerald-600 text-white text-sm sm:text-base font-medium rounded-xl hover:bg-emerald-700 transition-colors disabled:opacity-50"
          >
            {loading ? 'Loading...' : mode === 'signin' ? 'Sign in' : 'Create account'}
          </button>
        </form>

        <p className="mt-4 sm:mt-6 text-center text-xs sm:text-sm text-slate-500">
          {mode === 'signin' ? "Don't have an account? " : 'Already have an account? '}
          <button
            onClick={() => setmode(mode === 'signin' ? 'signup' : 'signin')}
            className="text-emerald-600 font-medium hover:underline"
          >
            {mode === 'signin' ? 'Sign up' : 'Sign in'}
          </button>
        </p>
      </div>
    </div>,
    document.body
  )
}
