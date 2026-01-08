import { createContext, useContext, useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'

const authcontext = createContext({})

export const useauth = () => {
  const context = useContext(authcontext)
  if (!context) {
    throw new Error('useauth must be used within authprovider')
  }
  return context
}

export const Authprovider = ({ children }) => {
  const [user, setuser] = useState(null)
  const [loading, setloading] = useState(true)
  const [avatar, setavatar] = useState(null)

  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setuser(session?.user ?? null)
      setavatar(session?.user?.user_metadata?.avatar_url ?? null)
      setloading(false)
    })

    supabase.auth.getSession().then(({ data: { session } }) => {
      setuser(session?.user ?? null)
      setavatar(session?.user?.user_metadata?.avatar_url ?? null)
      setloading(false)
    })

    return () => subscription.unsubscribe()
  }, [])

  const signup = async (email, password) => {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
    })
    return { data, error }
  }

  const signin = async (email, password) => {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    return { data, error }
  }

  const signinwithgoogle = async () => {
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: window.location.origin,
      },
    })
    return { data, error }
  }

  const signinwithgithub = async () => {
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: 'github',
      options: {
        redirectTo: window.location.origin,
      },
    })
    return { data, error }
  }

  const signout = async () => {
    const { error } = await supabase.auth.signOut()
    return { error }
  }

  const updateavatar = async (avatarurl) => {
    const { data, error } = await supabase.auth.updateUser({
      data: { avatar_url: avatarurl }
    })
    if (!error) {
      setavatar(avatarurl)
    }
    return { data, error }
  }

  const value = {
    user,
    loading,
    avatar,
    signup,
    signin,
    signinwithgoogle,
    signinwithgithub,
    signout,
    updateavatar,
  }

  return (
    <authcontext.Provider value={value}>
      {children}
    </authcontext.Provider>
  )
}
