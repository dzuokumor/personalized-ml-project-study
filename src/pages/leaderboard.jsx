import { useState, useEffect, Suspense } from 'react'
import { useauth } from '../contexts/authcontext'
import { usestats } from '../hooks/usestats'
import { supabase } from '../lib/supabase'
import { getoptimizedurl } from '../services/cloudinary'
import Neuroncrown from '../components/leaderboard/neuroncrown'

export default function leaderboard() {
  const { user, avatar } = useauth()
  const { stats } = usestats()
  const [leaders, setleaders] = useState([])
  const [loading, setloading] = useState(true)
  const [userrank, setuserrank] = useState(null)

  useEffect(() => {
    const fetchleaderboard = async () => {
      const { data, error } = await supabase
        .from('user_stats')
        .select('user_id, username, fullname, xp, level, avatar_url')
        .order('xp', { ascending: false })
        .limit(100)

      if (error) {
        console.error('Error fetching leaderboard:', error)
        setloading(false)
        return
      }

      setleaders(data || [])

      if (user && data) {
        const rank = data.findIndex(l => l.user_id === user.id)
        if (rank !== -1) {
          setuserrank({
            rank: rank + 1,
            xptogap: rank > 0 ? data[rank - 1].xp - data[rank].xp : 0
          })
        } else {
          const { data: userdata } = await supabase
            .from('user_stats')
            .select('xp')
            .eq('user_id', user.id)
            .single()

          if (userdata) {
            const highercount = data.filter(l => l.xp > userdata.xp).length
            setuserrank({
              rank: highercount + 1,
              xptogap: highercount < data.length ? data[highercount]?.xp - userdata.xp : 0
            })
          }
        }
      }

      setloading(false)
    }

    fetchleaderboard()
  }, [user])

  const top3 = leaders.slice(0, 3)
  const rest = leaders.slice(3, 10)
  const showall = leaders.length < 3

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="relative bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-2xl p-6 sm:p-8 mb-8 overflow-hidden">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <svg className="w-full h-full opacity-20" viewBox="0 0 400 200" preserveAspectRatio="xMidYMid slice">
            <defs>
              <linearGradient id="neuralLeaderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
                <stop offset="50%" stopColor="#06b6d4" stopOpacity="0.4" />
                <stop offset="100%" stopColor="#10b981" stopOpacity="0.8" />
              </linearGradient>
            </defs>
            <g className="animate-pulse" style={{ animationDuration: '4s' }}>
              <path d="M0,100 Q50,50 100,80 T200,60 T300,90 T400,70" stroke="url(#neuralLeaderGrad)" strokeWidth="1.5" fill="none" />
              <path d="M0,150 Q80,100 160,130 T320,110 T400,140" stroke="url(#neuralLeaderGrad)" strokeWidth="1" fill="none" opacity="0.6" />
            </g>
            <g className="animate-pulse" style={{ animationDuration: '3s' }}>
              <circle cx="50" cy="80" r="4" fill="#10b981" />
              <circle cx="150" cy="60" r="3" fill="#06b6d4" />
              <circle cx="250" cy="90" r="4" fill="#10b981" />
              <circle cx="350" cy="70" r="3" fill="#06b6d4" />
            </g>
          </svg>
        </div>

        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-xl flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
              </svg>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">Leaderboard</h1>
              <p className="text-sm text-slate-400 font-mono">Neural Network Rankings</p>
            </div>
          </div>
        </div>
      </div>

      {showall && leaders.length > 0 && (
        <div className="mb-8">
          <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-xl overflow-hidden border border-slate-700/50">
            <div className="px-4 py-3 border-b border-slate-700/50 bg-slate-800/50">
              <span className="text-sm text-slate-400 font-mono">Top Learners</span>
            </div>
            <div className="divide-y divide-slate-700/30">
              {leaders.map((leader, idx) => {
                const isuser = user && leader.user_id === user.id
                return (
                  <div
                    key={leader.user_id}
                    className={`flex items-center justify-between px-4 py-3 ${isuser ? 'bg-emerald-900/20' : 'hover:bg-slate-800/50'}`}
                  >
                    <div className="flex items-center gap-4">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-mono font-bold text-sm ${idx === 0 ? 'bg-amber-400 text-white' : idx === 1 ? 'bg-slate-400 text-white' : idx === 2 ? 'bg-amber-600 text-white' : 'bg-slate-700 text-slate-300'}`}>
                        {idx + 1}
                      </div>
                      <div className="w-10 h-10 rounded-lg bg-slate-700 overflow-hidden border border-slate-600">
                        {leader.avatar_url ? (
                          <img src={getoptimizedurl(leader.avatar_url, { width: 80, height: 80 })} alt="" className="w-full h-full object-cover" />
                        ) : (
                          <div className="w-full h-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-slate-300 font-semibold text-sm">
                            {leader.username?.[0]?.toUpperCase() || leader.fullname?.[0]?.toUpperCase() || '?'}
                          </div>
                        )}
                      </div>
                      <div>
                        <p className={`font-medium ${isuser ? 'text-emerald-400' : 'text-white'}`}>{leader.username || leader.fullname || 'Anonymous'}</p>
                        <p className="text-xs text-slate-500">Level {leader.level || 1}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className={`font-mono font-semibold ${isuser ? 'text-emerald-400' : 'text-cyan-400'}`}>{(leader.xp || 0).toLocaleString()}</p>
                      <p className="text-[10px] text-slate-500 uppercase tracking-wider">XP</p>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}

      {top3.length >= 3 && (
        <div className="relative bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-2xl p-6 sm:p-8 mb-8 overflow-hidden border border-slate-700/50">
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <svg className="w-full h-full opacity-10" viewBox="0 0 400 300" preserveAspectRatio="xMidYMid slice">
              <defs>
                <linearGradient id="podiumGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#10b981" />
                  <stop offset="100%" stopColor="#06b6d4" />
                </linearGradient>
              </defs>
              <g className="animate-pulse" style={{ animationDuration: '5s' }}>
                <line x1="20" y1="20" x2="100" y2="60" stroke="url(#podiumGrad)" strokeWidth="0.5" />
                <line x1="100" y1="60" x2="200" y2="40" stroke="url(#podiumGrad)" strokeWidth="0.5" />
                <line x1="200" y1="40" x2="300" y2="70" stroke="url(#podiumGrad)" strokeWidth="0.5" />
                <line x1="300" y1="70" x2="380" y2="30" stroke="url(#podiumGrad)" strokeWidth="0.5" />
                <circle cx="20" cy="20" r="3" fill="#10b981" />
                <circle cx="100" cy="60" r="4" fill="#06b6d4" />
                <circle cx="200" cy="40" r="5" fill="#10b981" />
                <circle cx="300" cy="70" r="4" fill="#06b6d4" />
                <circle cx="380" cy="30" r="3" fill="#10b981" />
              </g>
            </svg>
          </div>

          <div className="relative flex items-end justify-center gap-2 sm:gap-4 pt-4">
            {[1, 0, 2].map((idx, pos) => {
              const leader = top3[idx]
              if (!leader) return null

              const podiumheight = idx === 0 ? 'h-32 sm:h-40' : idx === 1 ? 'h-24 sm:h-28' : 'h-20 sm:h-24'
              const podiumcolor = idx === 0
                ? 'from-amber-400 via-yellow-500 to-amber-600'
                : idx === 1
                  ? 'from-slate-300 via-slate-400 to-slate-500'
                  : 'from-amber-600 via-amber-700 to-amber-800'
              const bordercolor = idx === 0 ? 'border-amber-400' : idx === 1 ? 'border-slate-400' : 'border-amber-600'
              const glowcolor = idx === 0 ? 'shadow-amber-500/30' : idx === 1 ? 'shadow-slate-400/20' : 'shadow-amber-700/20'

              return (
                <div key={leader.user_id} className="flex flex-col items-center">
                  {idx === 0 && (
                    <div className="mb-2">
                      <Suspense fallback={<div className="w-20 h-20 animate-pulse bg-slate-800 rounded-full" />}>
                        <Neuroncrown />
                      </Suspense>
                    </div>
                  )}

                  <div className={`relative mb-3 ${idx !== 0 ? 'mt-12 sm:mt-16' : ''}`}>
                    <div className={`w-16 h-16 sm:w-20 sm:h-20 rounded-2xl border-3 ${bordercolor} bg-slate-800 overflow-hidden shadow-lg ${glowcolor} transform hover:scale-105 transition-transform`}>
                      {leader.avatar_url ? (
                        <img src={getoptimizedurl(leader.avatar_url, { width: 160, height: 160 })} alt="" className="w-full h-full object-cover" />
                      ) : (
                        <div className="w-full h-full bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center text-white font-bold text-xl sm:text-2xl">
                          {leader.username?.[0]?.toUpperCase() || '?'}
                        </div>
                      )}
                    </div>
                    <div className={`absolute -bottom-1 -right-1 w-7 h-7 sm:w-8 sm:h-8 bg-gradient-to-br ${podiumcolor} rounded-full flex items-center justify-center text-white font-bold text-sm shadow-lg border-2 border-slate-900`}>
                      {idx + 1}
                    </div>
                  </div>

                  <p className="text-sm sm:text-base font-bold text-white mb-1 text-center truncate max-w-[80px] sm:max-w-[100px]">
                    {leader.username || leader.fullname || 'Anonymous'}
                  </p>
                  <p className="text-xs text-slate-400 mb-3 font-mono">Level {leader.level || 1}</p>

                  <div className={`w-20 sm:w-28 ${podiumheight} bg-gradient-to-b ${podiumcolor} rounded-t-xl flex flex-col items-center justify-start pt-3 sm:pt-4 shadow-xl relative overflow-hidden`}>
                    <div className="absolute inset-0 bg-gradient-to-r from-white/10 via-transparent to-black/10" />
                    <div className="absolute top-0 left-0 right-0 h-1 bg-white/30" />

                    <span className="text-white/90 font-bold text-2xl sm:text-3xl relative z-10">#{idx + 1}</span>
                    <span className="text-white/70 font-mono text-xs sm:text-sm mt-1 relative z-10">{(leader.xp || 0).toLocaleString()}</span>
                    <span className="text-white/50 text-[10px] uppercase tracking-wider relative z-10">XP</span>

                    <div className="absolute bottom-0 left-0 right-0">
                      <svg className="w-full h-8 opacity-20" viewBox="0 0 100 30">
                        <path d="M0,15 Q25,5 50,15 T100,15 L100,30 L0,30 Z" fill="currentColor" className="text-black" />
                      </svg>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          <div className="absolute bottom-0 left-0 right-0 h-4 bg-gradient-to-t from-slate-950 to-transparent" />
        </div>
      )}

      {userrank && userrank.rank > 3 && (
        <div className="bg-gradient-to-r from-emerald-900/80 to-cyan-900/80 rounded-xl p-4 mb-6 border border-emerald-500/30 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-slate-800 border-2 border-emerald-500/50 overflow-hidden shadow-lg shadow-emerald-500/20">
                {avatar ? (
                  <img src={getoptimizedurl(avatar, { width: 96, height: 96 })} alt="" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full bg-gradient-to-br from-emerald-600 to-cyan-600 flex items-center justify-center text-white font-bold">
                    {stats.username?.[0]?.toUpperCase() || '?'}
                  </div>
                )}
              </div>
              <div>
                <p className="text-white font-semibold">{stats.username || 'You'}</p>
                <p className="text-emerald-400 font-mono text-sm">{stats.xp.toLocaleString()} XP</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-3xl font-bold text-white font-mono">#{userrank.rank}</p>
              {userrank.xptogap > 0 && (
                <p className="text-xs text-slate-400">{userrank.xptogap.toLocaleString()} XP to next rank</p>
              )}
            </div>
          </div>
        </div>
      )}

      {rest.length > 0 && (
        <div className="relative bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-xl overflow-hidden border border-slate-700/50">
          <div className="px-4 py-3 border-b border-slate-700/50 bg-slate-800/50">
            <div className="flex items-center justify-between text-xs text-slate-400 font-mono uppercase tracking-wider">
              <span>Rank</span>
              <span>Learner</span>
              <span>XP</span>
            </div>
          </div>

          <div className="divide-y divide-slate-700/30">
            {rest.map((leader, idx) => {
              const rank = idx + 4
              const isuser = user && leader.user_id === user.id
              return (
                <div
                  key={leader.user_id}
                  className={`flex items-center justify-between px-4 py-3 transition-colors ${isuser ? 'bg-emerald-900/20' : 'hover:bg-slate-800/50'}`}
                >
                  <div className="flex items-center gap-4">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-mono font-bold text-sm ${isuser ? 'bg-emerald-500 text-white' : 'bg-slate-700 text-slate-300'}`}>
                      {rank}
                    </div>
                    <div className="w-10 h-10 rounded-lg bg-slate-700 overflow-hidden border border-slate-600">
                      {leader.avatar_url ? (
                        <img src={getoptimizedurl(leader.avatar_url, { width: 80, height: 80 })} alt="" className="w-full h-full object-cover" />
                      ) : (
                        <div className="w-full h-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-slate-300 font-semibold text-sm">
                          {leader.username?.[0]?.toUpperCase() || '?'}
                        </div>
                      )}
                    </div>
                    <div>
                      <p className={`font-medium ${isuser ? 'text-emerald-400' : 'text-white'}`}>{leader.username || leader.fullname || 'Anonymous'}</p>
                      {leader.fullname && leader.username && (
                        <p className="text-xs text-slate-500">{leader.fullname}</p>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    <p className={`font-mono font-semibold ${isuser ? 'text-emerald-400' : 'text-cyan-400'}`}>{(leader.xp || 0).toLocaleString()}</p>
                    <p className="text-[10px] text-slate-500 uppercase tracking-wider">XP</p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {leaders.length === 0 && (
        <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-xl p-12 text-center border border-slate-700/50">
          <div className="w-20 h-20 bg-slate-800 rounded-2xl mx-auto mb-4 flex items-center justify-center border border-slate-700">
            <svg className="w-10 h-10 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
          </div>
          <p className="text-slate-400 text-lg">No learners on the leaderboard yet</p>
          <p className="text-sm text-slate-500 mt-2">Start learning to climb the ranks!</p>
        </div>
      )}
    </div>
  )
}
