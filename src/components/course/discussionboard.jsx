import { useState, useEffect } from 'react'
import { useauth } from '../../contexts/authcontext'
import { supabase } from '../../lib/supabase'
import { getoptimizedurl } from '../../services/cloudinary'
import { usestats } from '../../hooks/usestats'

function neuralbackground() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <svg className="w-full h-full opacity-20" viewBox="0 0 600 400" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="discLineGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.4" />
          </linearGradient>
          <linearGradient id="discLineGrad2" x1="100%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#10b981" stopOpacity="0.3" />
          </linearGradient>
        </defs>

        <g className="animate-pulse" style={{ animationDuration: '4s' }}>
          <line x1="20" y1="40" x2="100" y2="80" stroke="url(#discLineGrad1)" strokeWidth="1" opacity="0.6" />
          <line x1="20" y1="40" x2="100" y2="160" stroke="url(#discLineGrad1)" strokeWidth="1" opacity="0.5" />
          <line x1="20" y1="120" x2="100" y2="80" stroke="url(#discLineGrad1)" strokeWidth="1" opacity="0.4" />
          <line x1="20" y1="120" x2="100" y2="160" stroke="url(#discLineGrad1)" strokeWidth="1" opacity="0.6" />
          <line x1="20" y1="200" x2="100" y2="160" stroke="url(#discLineGrad1)" strokeWidth="1" opacity="0.5" />
        </g>

        <g className="animate-pulse" style={{ animationDuration: '3s', animationDelay: '0.5s' }}>
          <line x1="100" y1="80" x2="200" y2="120" stroke="url(#discLineGrad2)" strokeWidth="1" opacity="0.5" />
          <line x1="100" y1="160" x2="200" y2="120" stroke="url(#discLineGrad2)" strokeWidth="1" opacity="0.6" />
          <line x1="100" y1="160" x2="200" y2="200" stroke="url(#discLineGrad2)" strokeWidth="1" opacity="0.4" />
        </g>

        <g className="animate-pulse" style={{ animationDuration: '3.5s', animationDelay: '1s' }}>
          <line x1="200" y1="120" x2="300" y2="140" stroke="url(#discLineGrad1)" strokeWidth="1" opacity="0.6" />
          <line x1="200" y1="200" x2="300" y2="140" stroke="url(#discLineGrad1)" strokeWidth="1" opacity="0.5" />
          <line x1="200" y1="200" x2="300" y2="240" stroke="url(#discLineGrad1)" strokeWidth="1" opacity="0.4" />
        </g>

        <g className="animate-pulse" style={{ animationDuration: '4.5s', animationDelay: '0.3s' }}>
          <line x1="400" y1="60" x2="480" y2="100" stroke="url(#discLineGrad2)" strokeWidth="1" opacity="0.5" />
          <line x1="400" y1="140" x2="480" y2="100" stroke="url(#discLineGrad2)" strokeWidth="1" opacity="0.6" />
          <line x1="480" y1="100" x2="560" y2="80" stroke="url(#discLineGrad1)" strokeWidth="1" opacity="0.4" />
          <line x1="480" y1="100" x2="560" y2="160" stroke="url(#discLineGrad1)" strokeWidth="1" opacity="0.5" />
        </g>

        <g className="animate-pulse" style={{ animationDuration: '5s', animationDelay: '0.8s' }}>
          <line x1="420" y1="280" x2="500" y2="320" stroke="url(#discLineGrad1)" strokeWidth="1" opacity="0.6" />
          <line x1="420" y1="360" x2="500" y2="320" stroke="url(#discLineGrad1)" strokeWidth="1" opacity="0.5" />
          <line x1="500" y1="320" x2="580" y2="300" stroke="url(#discLineGrad2)" strokeWidth="1" opacity="0.4" />
        </g>

        <g>
          <circle cx="20" cy="40" r="5" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2s' }} />
          <circle cx="20" cy="40" r="2" fill="#10b981" />
          <circle cx="20" cy="120" r="5" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.5s' }} />
          <circle cx="20" cy="120" r="2" fill="#10b981" />
          <circle cx="20" cy="200" r="5" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '3s' }} />
          <circle cx="20" cy="200" r="2" fill="#10b981" />
        </g>

        <g>
          <circle cx="100" cy="80" r="6" fill="#1e293b" stroke="#06b6d4" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.2s' }} />
          <circle cx="100" cy="80" r="2.5" fill="#06b6d4" />
          <circle cx="100" cy="160" r="6" fill="#1e293b" stroke="#06b6d4" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.7s' }} />
          <circle cx="100" cy="160" r="2.5" fill="#06b6d4" />
        </g>

        <g>
          <circle cx="200" cy="120" r="7" fill="#1e293b" stroke="#10b981" strokeWidth="2" className="animate-pulse" style={{ animationDuration: '2.4s' }} />
          <circle cx="200" cy="120" r="3" fill="#10b981" />
          <circle cx="200" cy="200" r="7" fill="#1e293b" stroke="#10b981" strokeWidth="2" className="animate-pulse" style={{ animationDuration: '2.9s' }} />
          <circle cx="200" cy="200" r="3" fill="#10b981" />
        </g>

        <g>
          <circle cx="300" cy="140" r="6" fill="#1e293b" stroke="#06b6d4" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.6s' }} />
          <circle cx="300" cy="140" r="2.5" fill="#06b6d4" />
          <circle cx="300" cy="240" r="6" fill="#1e293b" stroke="#06b6d4" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '3.1s' }} />
          <circle cx="300" cy="240" r="2.5" fill="#06b6d4" />
        </g>

        <g>
          <circle cx="400" cy="60" r="5" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.3s' }} />
          <circle cx="400" cy="60" r="2" fill="#10b981" />
          <circle cx="400" cy="140" r="5" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.8s' }} />
          <circle cx="400" cy="140" r="2" fill="#10b981" />
          <circle cx="480" cy="100" r="6" fill="#1e293b" stroke="#06b6d4" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '3.2s' }} />
          <circle cx="480" cy="100" r="2.5" fill="#06b6d4" />
          <circle cx="560" cy="80" r="5" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.5s' }} />
          <circle cx="560" cy="80" r="2" fill="#10b981" />
          <circle cx="560" cy="160" r="5" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '3s' }} />
          <circle cx="560" cy="160" r="2" fill="#10b981" />
        </g>

        <g>
          <circle cx="420" cy="280" r="5" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.7s' }} />
          <circle cx="420" cy="280" r="2" fill="#10b981" />
          <circle cx="420" cy="360" r="5" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '3.2s' }} />
          <circle cx="420" cy="360" r="2" fill="#10b981" />
          <circle cx="500" cy="320" r="6" fill="#1e293b" stroke="#06b6d4" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.9s' }} />
          <circle cx="500" cy="320" r="2.5" fill="#06b6d4" />
          <circle cx="580" cy="300" r="5" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '3.4s' }} />
          <circle cx="580" cy="300" r="2" fill="#10b981" />
        </g>
      </svg>
    </div>
  )
}

export default function discussionboard({ courseid, lessonid, lessontitle }) {
  const { user, avatar } = useauth()
  const { stats } = usestats()
  const [posts, setposts] = useState([])
  const [newpost, setnewpost] = useState('')
  const [replyto, setreplyto] = useState(null)
  const [replytext, setreplytext] = useState('')
  const [editing, setediting] = useState(null)
  const [edittext, setedittext] = useState('')
  const [loading, setloading] = useState(true)
  const [posting, setposting] = useState(false)
  const [expandedreplies, setexpandedreplies] = useState({})

  useEffect(() => {
    fetchposts()
  }, [courseid, lessonid])

  const fetchposts = async () => {
    const { data, error } = await supabase
      .from('discussions')
      .select(`
        id,
        content,
        created_at,
        updated_at,
        user_id,
        parent_id,
        user_stats(username, fullname, avatar_url)
      `)
      .eq('course_id', courseid)
      .eq('lesson_id', lessonid)
      .order('created_at', { ascending: false })

    if (error) {
      console.error('Error fetching discussions:', error)
      setloading(false)
      return
    }

    const toplevel = data?.filter(p => !p.parent_id) || []
    const replies = data?.filter(p => p.parent_id) || []

    const postswithreplies = toplevel.map(post => ({
      ...post,
      replies: replies.filter(r => r.parent_id === post.id).sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    }))

    setposts(postswithreplies)
    setloading(false)
  }

  const handlepost = async () => {
    if (!newpost.trim() || !user || posting) return

    setposting(true)
    const { error } = await supabase
      .from('discussions')
      .insert({
        course_id: courseid,
        lesson_id: lessonid,
        user_id: user.id,
        content: newpost.trim(),
        parent_id: null
      })

    if (error) {
      console.error('Error posting:', error)
    } else {
      setnewpost('')
      await fetchposts()
    }
    setposting(false)
  }

  const handlereply = async (postid) => {
    if (!replytext.trim() || !user || posting) return

    setposting(true)
    const { error } = await supabase
      .from('discussions')
      .insert({
        course_id: courseid,
        lesson_id: lessonid,
        user_id: user.id,
        content: replytext.trim(),
        parent_id: postid
      })

    if (error) {
      console.error('Error replying:', error)
    } else {
      setreplytext('')
      setreplyto(null)
      await fetchposts()
    }
    setposting(false)
  }

  const togglereplies = (postid) => {
    setexpandedreplies(prev => ({
      ...prev,
      [postid]: !prev[postid]
    }))
  }

  const startedit = (postid, content) => {
    setediting(postid)
    setedittext(content)
  }

  const canceledit = () => {
    setediting(null)
    setedittext('')
  }

  const handleedit = async (postid) => {
    if (!edittext.trim() || !user || posting) return

    setposting(true)
    const { error } = await supabase
      .from('discussions')
      .update({ content: edittext.trim(), updated_at: new Date().toISOString() })
      .eq('id', postid)
      .eq('user_id', user.id)

    if (error) {
      console.error('Error editing:', error)
    } else {
      setediting(null)
      setedittext('')
      await fetchposts()
    }
    setposting(false)
  }

  const formatdate = (datestr) => {
    const date = new Date(datestr)
    const now = new Date()
    const diff = now - date

    if (diff < 60000) return 'just now'
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`
    if (diff < 604800000) return `${Math.floor(diff / 86400000)}d ago`
    return date.toLocaleDateString()
  }

  return (
    <div className="relative bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-2xl overflow-hidden border border-slate-700/50">
      {neuralbackground()}

      <div className="relative px-4 sm:px-6 py-4 border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-slate-800 rounded-xl flex items-center justify-center border border-slate-700">
            <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a1.994 1.994 0 01-1.414-.586m0 0L11 14h4a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2v4l.586-.586z" />
            </svg>
          </div>
          <div>
            <h3 className="font-semibold text-white">Discussion</h3>
            <p className="text-xs text-slate-400 font-mono">{posts.length} posts</p>
          </div>
        </div>
      </div>

      <div className="relative p-4 sm:p-6">
        {user ? (
          <div className="mb-6">
            <div className="flex gap-3">
              <div className="w-10 h-10 rounded-lg bg-slate-700 overflow-hidden flex-shrink-0 border border-slate-600">
                {avatar ? (
                  <img src={getoptimizedurl(avatar, { width: 80, height: 80 })} alt="" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full bg-gradient-to-br from-emerald-600 to-cyan-600 flex items-center justify-center text-white font-bold text-sm">
                    {stats.username?.[0]?.toUpperCase() || '?'}
                  </div>
                )}
              </div>
              <div className="flex-1">
                <textarea
                  value={newpost}
                  onChange={(e) => setnewpost(e.target.value)}
                  placeholder="Share a question or insight about this lesson..."
                  className="w-full px-4 py-3 bg-slate-800/80 border border-slate-700/50 rounded-xl resize-none focus:outline-none focus:border-emerald-500/50 text-sm text-white placeholder-slate-500 min-h-[80px] backdrop-blur-sm"
                />
                <div className="flex justify-end mt-2">
                  <button
                    onClick={handlepost}
                    disabled={!newpost.trim() || posting}
                    className="px-4 py-2 bg-gradient-to-br from-emerald-500 to-emerald-600 text-white rounded-lg text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:from-emerald-400 hover:to-emerald-500 transition-all flex items-center gap-2"
                  >
                    {posting ? (
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    ) : (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                      </svg>
                    )}
                    Post
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="mb-6 p-4 bg-slate-800/50 rounded-xl border border-slate-700/50 text-center backdrop-blur-sm">
            <p className="text-slate-400 text-sm">Sign in to join the discussion</p>
          </div>
        )}

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-6 h-6 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : posts.length === 0 ? (
          <div className="text-center py-8">
            <div className="w-14 h-14 bg-slate-800/70 rounded-xl mx-auto mb-4 flex items-center justify-center border border-slate-700/50 backdrop-blur-sm">
              <svg className="w-7 h-7 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <p className="text-slate-400 text-sm">No discussions yet</p>
            <p className="text-slate-500 text-xs mt-1">Be the first to start a conversation!</p>
          </div>
        ) : (
          <div className="space-y-4">
            {posts.map(post => (
              <div key={post.id} className="bg-slate-800/40 rounded-xl border border-slate-700/30 overflow-hidden backdrop-blur-sm">
                <div className="p-4">
                  <div className="flex gap-3">
                    <div className="w-9 h-9 rounded-lg bg-slate-700 overflow-hidden flex-shrink-0 border border-slate-600">
                      {post.user_stats?.avatar_url ? (
                        <img src={getoptimizedurl(post.user_stats.avatar_url, { width: 72, height: 72 })} alt="" className="w-full h-full object-cover" />
                      ) : (
                        <div className="w-full h-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-slate-300 font-semibold text-sm">
                          {post.user_stats?.username?.[0]?.toUpperCase() || '?'}
                        </div>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-medium text-white text-sm">{post.user_stats?.username || 'Anonymous'}</span>
                        <span className="text-slate-500 text-xs font-mono">{formatdate(post.created_at)}</span>
                        {post.updated_at && post.updated_at !== post.created_at && (
                          <span className="text-slate-500 text-[10px] italic">(edited)</span>
                        )}
                      </div>
                      {editing === post.id ? (
                        <div className="mt-2">
                          <textarea
                            value={edittext}
                            onChange={(e) => setedittext(e.target.value)}
                            className="w-full px-3 py-2 bg-slate-800/80 border border-slate-700/50 rounded-lg resize-none focus:outline-none focus:border-emerald-500/50 text-sm text-white placeholder-slate-500 min-h-[80px]"
                          />
                          <div className="flex gap-2 mt-2">
                            <button
                              onClick={() => handleedit(post.id)}
                              disabled={!edittext.trim() || posting}
                              className="px-3 py-1.5 bg-emerald-600 text-white rounded-lg text-xs font-medium disabled:opacity-40 hover:bg-emerald-500 transition-colors"
                            >
                              Save
                            </button>
                            <button
                              onClick={canceledit}
                              className="px-3 py-1.5 bg-slate-700 text-slate-300 rounded-lg text-xs hover:bg-slate-600 transition-colors"
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : (
                        <p className="text-slate-300 text-sm whitespace-pre-wrap">{post.content}</p>
                      )}
                      <div className="flex items-center gap-4 mt-3">
                        {user && (
                          <button
                            onClick={() => setreplyto(replyto === post.id ? null : post.id)}
                            className="text-xs text-slate-400 hover:text-emerald-400 transition-colors flex items-center gap-1"
                          >
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
                            </svg>
                            Reply
                          </button>
                        )}
                        {user && post.user_id === user.id && editing !== post.id && (
                          <button
                            onClick={() => startedit(post.id, post.content)}
                            className="text-xs text-slate-400 hover:text-amber-400 transition-colors flex items-center gap-1"
                          >
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                            </svg>
                            Edit
                          </button>
                        )}
                        {post.replies.length > 0 && (
                          <button
                            onClick={() => togglereplies(post.id)}
                            className="text-xs text-slate-400 hover:text-cyan-400 transition-colors flex items-center gap-1"
                          >
                            <svg className={`w-3.5 h-3.5 transition-transform ${expandedreplies[post.id] ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                            {post.replies.length} {post.replies.length === 1 ? 'reply' : 'replies'}
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {replyto === post.id && (
                  <div className="px-4 pb-4">
                    <div className="ml-12 flex gap-2">
                      <textarea
                        value={replytext}
                        onChange={(e) => setreplytext(e.target.value)}
                        placeholder="Write a reply..."
                        className="flex-1 px-3 py-2 bg-slate-800/80 border border-slate-700/50 rounded-lg resize-none focus:outline-none focus:border-emerald-500/50 text-sm text-white placeholder-slate-500 min-h-[60px]"
                      />
                      <div className="flex flex-col gap-1">
                        <button
                          onClick={() => handlereply(post.id)}
                          disabled={!replytext.trim() || posting}
                          className="px-3 py-2 bg-emerald-600 text-white rounded-lg text-xs font-medium disabled:opacity-40 hover:bg-emerald-500 transition-colors"
                        >
                          Send
                        </button>
                        <button
                          onClick={() => { setreplyto(null); setreplytext('') }}
                          className="px-3 py-2 bg-slate-700 text-slate-300 rounded-lg text-xs hover:bg-slate-600 transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {expandedreplies[post.id] && post.replies.length > 0 && (
                  <div className="border-t border-slate-700/30 bg-slate-800/20">
                    {post.replies.map(reply => (
                      <div key={reply.id} className="px-4 py-3 border-b border-slate-700/20 last:border-b-0">
                        <div className="ml-8 flex gap-3">
                          <div className="w-7 h-7 rounded-md bg-slate-700 overflow-hidden flex-shrink-0 border border-slate-600">
                            {reply.user_stats?.avatar_url ? (
                              <img src={getoptimizedurl(reply.user_stats.avatar_url, { width: 56, height: 56 })} alt="" className="w-full h-full object-cover" />
                            ) : (
                              <div className="w-full h-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-slate-300 font-semibold text-xs">
                                {reply.user_stats?.username?.[0]?.toUpperCase() || '?'}
                              </div>
                            )}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-0.5">
                              <span className="font-medium text-slate-300 text-xs">{reply.user_stats?.username || 'Anonymous'}</span>
                              <span className="text-slate-500 text-[10px] font-mono">{formatdate(reply.created_at)}</span>
                              {reply.updated_at && reply.updated_at !== reply.created_at && (
                                <span className="text-slate-500 text-[10px] italic">(edited)</span>
                              )}
                              {user && reply.user_id === user.id && editing !== reply.id && (
                                <button
                                  onClick={() => startedit(reply.id, reply.content)}
                                  className="text-[10px] text-slate-500 hover:text-amber-400 transition-colors"
                                >
                                  Edit
                                </button>
                              )}
                            </div>
                            {editing === reply.id ? (
                              <div className="mt-1">
                                <textarea
                                  value={edittext}
                                  onChange={(e) => setedittext(e.target.value)}
                                  className="w-full px-2 py-1.5 bg-slate-800/80 border border-slate-700/50 rounded-lg resize-none focus:outline-none focus:border-emerald-500/50 text-xs text-white placeholder-slate-500 min-h-[50px]"
                                />
                                <div className="flex gap-2 mt-1">
                                  <button
                                    onClick={() => handleedit(reply.id)}
                                    disabled={!edittext.trim() || posting}
                                    className="px-2 py-1 bg-emerald-600 text-white rounded text-[10px] font-medium disabled:opacity-40 hover:bg-emerald-500 transition-colors"
                                  >
                                    Save
                                  </button>
                                  <button
                                    onClick={canceledit}
                                    className="px-2 py-1 bg-slate-700 text-slate-300 rounded text-[10px] hover:bg-slate-600 transition-colors"
                                  >
                                    Cancel
                                  </button>
                                </div>
                              </div>
                            ) : (
                              <p className="text-slate-400 text-sm whitespace-pre-wrap">{reply.content}</p>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
