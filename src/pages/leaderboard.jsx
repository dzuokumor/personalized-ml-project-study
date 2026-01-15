import { useState, useEffect } from 'react'
import { useauth } from '../contexts/authcontext'
import { usestats } from '../hooks/usestats'
import { supabase } from '../lib/supabase'
import { getoptimizedurl } from '../services/cloudinary'

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

      console.log('Leaderboard fetch result:', { data, error })

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

  const podiumorder = [1, 0, 2]
  const podiumheights = ['h-28', 'h-36', 'h-24']
  const podiumcolors = ['from-slate-400 to-slate-500', 'from-amber-400 to-yellow-500', 'from-amber-600 to-amber-700']
  const badgecolors = ['bg-slate-400', 'bg-amber-400', 'bg-amber-600']

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
        <div className="absolute inset-0 opacity-30">
          <svg className="w-full h-full" viewBox="0 0 400 200" preserveAspectRatio="none">
            <defs>
              <linearGradient id="neuralLeaderGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.6" />
                <stop offset="50%" stopColor="#06b6d4" stopOpacity="0.3" />
                <stop offset="100%" stopColor="#10b981" stopOpacity="0.6" />
              </linearGradient>
            </defs>
            <path d="M0,100 Q50,50 100,80 T200,60 T300,90 T400,70" stroke="url(#neuralLeaderGrad)" strokeWidth="1" fill="none" />
            <path d="M0,150 Q80,100 160,130 T320,110 T400,140" stroke="url(#neuralLeaderGrad)" strokeWidth="0.8" fill="none" opacity="0.6" />
            <circle cx="50" cy="80" r="3" fill="#10b981" opacity="0.8" />
            <circle cx="150" cy="60" r="2" fill="#06b6d4" opacity="0.6" />
            <circle cx="250" cy="90" r="3" fill="#10b981" opacity="0.7" />
            <circle cx="350" cy="70" r="2" fill="#06b6d4" opacity="0.5" />
            <circle cx="100" cy="130" r="2" fill="#10b981" opacity="0.5" />
            <circle cx="200" cy="110" r="3" fill="#06b6d4" opacity="0.6" />
            <circle cx="300" cy="120" r="2" fill="#10b981" opacity="0.4" />
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
        <div className="mb-8">
          <div className="flex items-end justify-center gap-4 sm:gap-6">
            {podiumorder.map((idx, pos) => {
              const leader = top3[idx]
              if (!leader) return null
              return (
                <div key={leader.user_id} className="flex flex-col items-center">
                  <div className="relative mb-3">
                    <div className={`w-16 h-16 sm:w-20 sm:h-20 rounded-xl border-2 ${idx === 0 ? 'border-amber-400' : idx === 1 ? 'border-slate-400' : 'border-amber-600'} bg-slate-800 overflow-hidden`}>
                      {leader.avatar_url ? (
                        <img src={getoptimizedurl(leader.avatar_url, { width: 160, height: 160 })} alt="" className="w-full h-full object-cover" />
                      ) : (
                        <div className="w-full h-full bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center text-white font-bold text-lg sm:text-xl">
                          {leader.username?.[0]?.toUpperCase() || '?'}
                        </div>
                      )}
                    </div>
                    <div className={`absolute -top-2 -right-2 w-6 h-6 sm:w-7 sm:h-7 ${badgecolors[idx]} rounded-full flex items-center justify-center text-white font-bold text-xs sm:text-sm shadow-lg`}>
                      {idx + 1}
                    </div>
                  </div>
                  <p className="text-sm sm:text-base font-semibold text-slate-900 mb-1 text-center truncate max-w-[80px] sm:max-w-[100px]">{leader.username}</p>
                  <p className="text-xs sm:text-sm text-emerald-600 font-mono font-semibold mb-3">{leader.xp.toLocaleString()} XP</p>
                  <div className={`w-20 sm:w-24 ${podiumheights[pos]} bg-gradient-to-t ${podiumcolors[idx]} rounded-t-lg flex items-start justify-center pt-3`}>
                    <span className="text-white font-bold text-lg sm:text-xl">#{idx + 1}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {userrank && userrank.rank > 3 && (
        <div className="bg-gradient-to-r from-emerald-900 to-cyan-900 rounded-xl p-4 mb-6 border border-emerald-500/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-xl bg-slate-800 border border-emerald-500/50 overflow-hidden">
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
              <p className="text-2xl font-bold text-white font-mono">#{userrank.rank}</p>
              {userrank.xptogap > 0 && (
                <p className="text-xs text-slate-400">{userrank.xptogap.toLocaleString()} XP to next rank</p>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-xl overflow-hidden border border-slate-700/50">
        <div className="absolute inset-0 opacity-10 pointer-events-none">
          <svg className="w-full h-full" viewBox="0 0 200 400" preserveAspectRatio="none">
            <path d="M20,0 L20,400" stroke="#10b981" strokeWidth="0.5" strokeDasharray="4,8" />
            <path d="M60,0 L60,400" stroke="#06b6d4" strokeWidth="0.3" strokeDasharray="2,12" />
            <path d="M140,0 L140,400" stroke="#10b981" strokeWidth="0.3" strokeDasharray="2,12" />
            <path d="M180,0 L180,400" stroke="#06b6d4" strokeWidth="0.5" strokeDasharray="4,8" />
          </svg>
        </div>

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
                    <p className={`font-medium ${isuser ? 'text-emerald-400' : 'text-white'}`}>{leader.username}</p>
                    {leader.fullname && (
                      <p className="text-xs text-slate-500">{leader.fullname}</p>
                    )}
                  </div>
                </div>
                <div className="text-right">
                  <p className={`font-mono font-semibold ${isuser ? 'text-emerald-400' : 'text-cyan-400'}`}>{leader.xp.toLocaleString()}</p>
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider">XP</p>
                </div>
              </div>
            )
          })}
        </div>

        {leaders.length === 0 && (
          <div className="px-4 py-12 text-center">
            <div className="w-16 h-16 bg-slate-800 rounded-xl mx-auto mb-4 flex items-center justify-center">
              <svg className="w-8 h-8 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
            </div>
            <p className="text-slate-400">No learners on the leaderboard yet</p>
            <p className="text-sm text-slate-500 mt-1">Start learning to climb the ranks!</p>
          </div>
        )}
      </div>
    </div>
  )
}
