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
  const [githubtoken, setgithubtoken] = useState(null)
  const [githubconnected, setgithubconnected] = useState(false)

  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setuser(session?.user ?? null)
      setavatar(session?.user?.user_metadata?.custom_avatar_url ?? session?.user?.user_metadata?.avatar_url ?? null)
      if (session?.provider_token && session?.user?.app_metadata?.provider === 'github') {
        setgithubtoken(session.provider_token)
        setgithubconnected(true)
      }
      setloading(false)
    })

    supabase.auth.getSession().then(({ data: { session } }) => {
      setuser(session?.user ?? null)
      setavatar(session?.user?.user_metadata?.custom_avatar_url ?? session?.user?.user_metadata?.avatar_url ?? null)
      const isgithubuser = session?.user?.app_metadata?.provider === 'github' ||
                          session?.user?.identities?.some(i => i.provider === 'github')
      setgithubconnected(isgithubuser)
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
        scopes: 'repo',
      },
    })
    return { data, error }
  }

  const connectgithubforrepos = async () => {
    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: 'github',
      options: {
        redirectTo: `${window.location.origin}/profile?tab=github`,
        scopes: 'repo',
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
      data: { custom_avatar_url: avatarurl }
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
    githubtoken,
    githubconnected,
    signup,
    signin,
    signinwithgoogle,
    signinwithgithub,
    connectgithubforrepos,
    signout,
    updateavatar,
  }

  return (
    <authcontext.Provider value={value}>
      {children}
    </authcontext.Provider>
  )
}
