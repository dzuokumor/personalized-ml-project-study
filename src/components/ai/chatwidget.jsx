import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

function neuralbackground() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <svg className="w-full h-full opacity-30" viewBox="0 0 400 500" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="lineGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.4" />
          </linearGradient>
          <linearGradient id="lineGrad2" x1="100%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#10b981" stopOpacity="0.3" />
          </linearGradient>
        </defs>

        <g className="animate-pulse" style={{ animationDuration: '4s' }}>
          <line x1="30" y1="50" x2="120" y2="100" stroke="url(#lineGrad1)" strokeWidth="1" opacity="0.6" />
          <line x1="30" y1="50" x2="120" y2="180" stroke="url(#lineGrad1)" strokeWidth="1" opacity="0.5" />
          <line x1="30" y1="150" x2="120" y2="100" stroke="url(#lineGrad1)" strokeWidth="1" opacity="0.4" />
          <line x1="30" y1="150" x2="120" y2="180" stroke="url(#lineGrad1)" strokeWidth="1" opacity="0.6" />
          <line x1="30" y1="250" x2="120" y2="180" stroke="url(#lineGrad1)" strokeWidth="1" opacity="0.5" />
          <line x1="30" y1="250" x2="120" y2="280" stroke="url(#lineGrad1)" strokeWidth="1" opacity="0.4" />
        </g>

        <g className="animate-pulse" style={{ animationDuration: '3s', animationDelay: '0.5s' }}>
          <line x1="120" y1="100" x2="220" y2="140" stroke="url(#lineGrad2)" strokeWidth="1" opacity="0.5" />
          <line x1="120" y1="180" x2="220" y2="140" stroke="url(#lineGrad2)" strokeWidth="1" opacity="0.6" />
          <line x1="120" y1="180" x2="220" y2="240" stroke="url(#lineGrad2)" strokeWidth="1" opacity="0.4" />
          <line x1="120" y1="280" x2="220" y2="240" stroke="url(#lineGrad2)" strokeWidth="1" opacity="0.5" />
          <line x1="120" y1="280" x2="220" y2="340" stroke="url(#lineGrad2)" strokeWidth="1" opacity="0.6" />
        </g>

        <g className="animate-pulse" style={{ animationDuration: '3.5s', animationDelay: '1s' }}>
          <line x1="220" y1="140" x2="320" y2="180" stroke="url(#lineGrad1)" strokeWidth="1" opacity="0.6" />
          <line x1="220" y1="240" x2="320" y2="180" stroke="url(#lineGrad1)" strokeWidth="1" opacity="0.5" />
          <line x1="220" y1="240" x2="320" y2="300" stroke="url(#lineGrad1)" strokeWidth="1" opacity="0.4" />
          <line x1="220" y1="340" x2="320" y2="300" stroke="url(#lineGrad1)" strokeWidth="1" opacity="0.6" />
        </g>

        <g className="animate-pulse" style={{ animationDuration: '5s' }}>
          <line x1="320" y1="180" x2="380" y2="220" stroke="url(#lineGrad2)" strokeWidth="1" opacity="0.5" />
          <line x1="320" y1="300" x2="380" y2="220" stroke="url(#lineGrad2)" strokeWidth="1" opacity="0.4" />
          <line x1="320" y1="300" x2="380" y2="380" stroke="url(#lineGrad2)" strokeWidth="1" opacity="0.6" />
        </g>

        <g>
          <circle cx="30" cy="50" r="6" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2s' }} />
          <circle cx="30" cy="50" r="2.5" fill="#10b981" />

          <circle cx="30" cy="150" r="6" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.5s' }} />
          <circle cx="30" cy="150" r="2.5" fill="#10b981" />

          <circle cx="30" cy="250" r="6" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '3s' }} />
          <circle cx="30" cy="250" r="2.5" fill="#10b981" />
        </g>

        <g>
          <circle cx="120" cy="100" r="7" fill="#1e293b" stroke="#06b6d4" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.2s' }} />
          <circle cx="120" cy="100" r="3" fill="#06b6d4" />

          <circle cx="120" cy="180" r="7" fill="#1e293b" stroke="#06b6d4" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.7s' }} />
          <circle cx="120" cy="180" r="3" fill="#06b6d4" />

          <circle cx="120" cy="280" r="7" fill="#1e293b" stroke="#06b6d4" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '3.2s' }} />
          <circle cx="120" cy="280" r="3" fill="#06b6d4" />
        </g>

        <g>
          <circle cx="220" cy="140" r="8" fill="#1e293b" stroke="#10b981" strokeWidth="2" className="animate-pulse" style={{ animationDuration: '2.4s' }} />
          <circle cx="220" cy="140" r="3.5" fill="#10b981" />

          <circle cx="220" cy="240" r="8" fill="#1e293b" stroke="#10b981" strokeWidth="2" className="animate-pulse" style={{ animationDuration: '2.9s' }} />
          <circle cx="220" cy="240" r="3.5" fill="#10b981" />

          <circle cx="220" cy="340" r="8" fill="#1e293b" stroke="#10b981" strokeWidth="2" className="animate-pulse" style={{ animationDuration: '3.4s' }} />
          <circle cx="220" cy="340" r="3.5" fill="#10b981" />
        </g>

        <g>
          <circle cx="320" cy="180" r="7" fill="#1e293b" stroke="#06b6d4" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.6s' }} />
          <circle cx="320" cy="180" r="3" fill="#06b6d4" />

          <circle cx="320" cy="300" r="7" fill="#1e293b" stroke="#06b6d4" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '3.1s' }} />
          <circle cx="320" cy="300" r="3" fill="#06b6d4" />
        </g>

        <g>
          <circle cx="380" cy="220" r="6" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '2.8s' }} />
          <circle cx="380" cy="220" r="2.5" fill="#10b981" />

          <circle cx="380" cy="380" r="6" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" className="animate-pulse" style={{ animationDuration: '3.3s' }} />
          <circle cx="380" cy="380" r="2.5" fill="#10b981" />
        </g>
      </svg>
    </div>
  )
}

function neuronicon({ className = "w-6 h-6" }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none">
      <g stroke="#10b981" strokeWidth="1" opacity="0.8">
        <line x1="4" y1="6" x2="12" y2="8" />
        <line x1="4" y1="12" x2="12" y2="8" />
        <line x1="4" y1="12" x2="12" y2="16" />
        <line x1="4" y1="18" x2="12" y2="16" />
        <line x1="12" y1="8" x2="20" y2="10" />
        <line x1="12" y1="16" x2="20" y2="10" />
        <line x1="12" y1="16" x2="20" y2="14" />
      </g>
      <circle cx="4" cy="6" r="2" fill="#0f172a" stroke="#10b981" strokeWidth="1" />
      <circle cx="4" cy="12" r="2" fill="#0f172a" stroke="#10b981" strokeWidth="1" />
      <circle cx="4" cy="18" r="2" fill="#0f172a" stroke="#10b981" strokeWidth="1" />
      <circle cx="12" cy="8" r="2.5" fill="#0f172a" stroke="#06b6d4" strokeWidth="1" />
      <circle cx="12" cy="16" r="2.5" fill="#0f172a" stroke="#06b6d4" strokeWidth="1" />
      <circle cx="20" cy="10" r="2" fill="#0f172a" stroke="#10b981" strokeWidth="1" />
      <circle cx="20" cy="14" r="2" fill="#0f172a" stroke="#10b981" strokeWidth="1" />
      <circle cx="4" cy="6" r="0.8" fill="#10b981" />
      <circle cx="4" cy="12" r="0.8" fill="#10b981" />
      <circle cx="4" cy="18" r="0.8" fill="#10b981" />
      <circle cx="12" cy="8" r="1" fill="#06b6d4" />
      <circle cx="12" cy="16" r="1" fill="#06b6d4" />
      <circle cx="20" cy="10" r="0.8" fill="#10b981" />
      <circle cx="20" cy="14" r="0.8" fill="#10b981" />
    </svg>
  )
}

export default function chatwidget({ context = '' }) {
  const [isopen, setisopen] = useState(false)
  const [messages, setmessages] = useState([])
  const [input, setinput] = useState('')
  const [loading, setloading] = useState(false)
  const messagesendref = useRef(null)

  useEffect(() => {
    messagesendref.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendmessage = async () => {
    if (!input.trim() || loading) return

    const usermessage = { role: 'user', content: input }
    setmessages(prev => [...prev, usermessage])
    setinput('')
    setloading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, usermessage],
          context
        })
      })

      const data = await response.json()

      if (data.error) {
        setmessages(prev => [...prev, {
          role: 'assistant',
          content: `Error: ${data.error}`
        }])
      } else {
        setmessages(prev => [...prev, {
          role: 'assistant',
          content: data.content
        }])
      }
    } catch (err) {
      setmessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, there was an error connecting to the AI. Please try again.'
      }])
    } finally {
      setloading(false)
    }
  }

  const handlekeydown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendmessage()
    }
  }

  return (
    <>
      <button
        onClick={() => setisopen(!isopen)}
        className="fixed bottom-4 right-4 sm:bottom-6 sm:right-6 w-12 h-12 sm:w-14 sm:h-14 bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl hover:scale-105 transition-all flex items-center justify-center z-50 border border-slate-700 shadow-lg"
      >
        {isopen ? (
          <svg className="w-6 h-6 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <neuronicon className="w-7 h-7" />
        )}
      </button>

      {isopen && (
        <div className="fixed bottom-20 sm:bottom-24 right-2 sm:right-6 left-2 sm:left-auto sm:w-96 h-[70vh] sm:h-[500px] bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-2xl flex flex-col z-50 overflow-hidden shadow-2xl border border-slate-700/50">
          <neuralbackground />

          <div className="relative px-4 py-3 border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 bg-slate-800 rounded-lg flex items-center justify-center border border-slate-700">
                  <neuronicon className="w-6 h-6" />
                </div>
                <div>
                  <span className="font-semibold text-white text-sm">Neural Tutor</span>
                  <div className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
                    <span className="text-[10px] text-slate-400 font-mono uppercase tracking-wider">Online</span>
                  </div>
                </div>
              </div>
              <button
                onClick={() => setmessages([])}
                className="px-2.5 py-1 text-xs text-slate-400 hover:text-white hover:bg-slate-700/50 rounded-lg transition-colors font-mono"
              >
                Clear
              </button>
            </div>
          </div>

          <div className="relative flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 && (
              <div className="text-center mt-8">
                <div className="w-16 h-16 mx-auto mb-4 bg-slate-800/70 rounded-xl flex items-center justify-center border border-slate-700/50 backdrop-blur-sm">
                  <neuronicon className="w-10 h-10" />
                </div>
                <p className="text-sm text-slate-300 font-medium mb-1">Ask me anything about ML</p>
                <p className="text-xs text-slate-500">Concepts, debugging, or questions</p>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[85%] px-4 py-2.5 rounded-2xl backdrop-blur-sm ${
                    msg.role === 'user'
                      ? 'bg-gradient-to-br from-emerald-600/90 to-emerald-700/90 text-white rounded-br-md'
                      : 'bg-slate-800/70 text-slate-200 rounded-bl-md border border-slate-700/50'
                  }`}
                >
                  {msg.role === 'assistant' ? (
                    <div className="prose prose-sm prose-invert max-w-none">
                      <ReactMarkdown
                        components={{
                          code: ({ node, inline, className, children, ...props }) => {
                            if (inline) {
                              return (
                                <code className="bg-slate-700 px-1.5 py-0.5 rounded text-sm font-mono text-cyan-300" {...props}>
                                  {children}
                                </code>
                              )
                            }
                            return (
                              <pre className="bg-slate-900 text-slate-100 p-3 rounded-lg overflow-x-auto my-2 border border-slate-700">
                                <code className="text-sm font-mono" {...props}>
                                  {children}
                                </code>
                              </pre>
                            )
                          },
                          p: ({ children }) => <p className="mb-2 last:mb-0 text-sm leading-relaxed">{children}</p>
                        }}
                      >
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <p className="text-sm">{msg.content}</p>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="bg-slate-800/70 px-4 py-3 rounded-2xl rounded-bl-md border border-slate-700/50 backdrop-blur-sm">
                  <div className="flex gap-1.5">
                    <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesendref} />
          </div>

          <div className="relative p-4 border-t border-slate-700/50 bg-slate-900/80 backdrop-blur-sm">
            <div className="flex gap-2">
              <textarea
                value={input}
                onChange={(e) => setinput(e.target.value)}
                onKeyDown={handlekeydown}
                placeholder="Ask about ML concepts..."
                className="flex-1 px-4 py-2.5 bg-slate-800/80 border border-slate-700/50 rounded-xl resize-none focus:outline-none focus:border-emerald-500/50 text-sm text-white placeholder-slate-500"
                rows={1}
              />
              <button
                onClick={sendmessage}
                disabled={loading || !input.trim()}
                className="px-4 py-2.5 bg-gradient-to-br from-emerald-500 to-emerald-600 text-white rounded-xl disabled:opacity-40 disabled:cursor-not-allowed hover:from-emerald-400 hover:to-emerald-500 transition-all"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
