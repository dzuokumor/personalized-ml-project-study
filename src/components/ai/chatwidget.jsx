import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'

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
        className="fixed bottom-4 right-4 sm:bottom-6 sm:right-6 w-12 h-12 sm:w-14 sm:h-14 bg-gradient-to-br from-slate-800 to-slate-900 text-emerald-400 rounded-xl hover:scale-105 transition-all flex items-center justify-center z-50 border border-slate-700 shadow-lg"
      >
        {isopen ? (
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        )}
      </button>

      {isopen && (
        <div className="fixed bottom-20 sm:bottom-24 right-2 sm:right-6 left-2 sm:left-auto sm:w-96 h-[70vh] sm:h-[500px] bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-2xl flex flex-col z-50 overflow-hidden shadow-2xl border border-slate-700/50">
          <div className="absolute inset-0 opacity-20 pointer-events-none">
            <svg className="w-full h-full" viewBox="0 0 400 500" preserveAspectRatio="none">
              <defs>
                <linearGradient id="chatNeuralGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#10b981" stopOpacity="0.5" />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.3" />
                </linearGradient>
              </defs>
              <path d="M0,50 Q100,30 200,60 T400,40" stroke="url(#chatNeuralGrad)" strokeWidth="0.8" fill="none" />
              <path d="M0,150 Q150,120 250,160 T400,130" stroke="url(#chatNeuralGrad)" strokeWidth="0.6" fill="none" opacity="0.5" />
              <path d="M0,350 Q80,320 180,360 T400,340" stroke="url(#chatNeuralGrad)" strokeWidth="0.6" fill="none" opacity="0.4" />
              <path d="M0,450 Q120,420 220,460 T400,440" stroke="url(#chatNeuralGrad)" strokeWidth="0.8" fill="none" opacity="0.3" />
              <circle cx="50" cy="50" r="2" fill="#10b981" opacity="0.6" />
              <circle cx="200" cy="60" r="1.5" fill="#06b6d4" opacity="0.5" />
              <circle cx="350" cy="40" r="2" fill="#10b981" opacity="0.4" />
              <circle cx="100" cy="150" r="1.5" fill="#06b6d4" opacity="0.4" />
              <circle cx="300" cy="130" r="2" fill="#10b981" opacity="0.3" />
            </svg>
          </div>

          <div className="relative px-4 py-3 border-b border-slate-700/50 bg-slate-800/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-lg flex items-center justify-center">
                  <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <div>
                  <span className="font-semibold text-white text-sm">Neural Tutor</span>
                  <div className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />
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
                <div className="w-16 h-16 mx-auto mb-4 bg-slate-800/50 rounded-xl flex items-center justify-center border border-slate-700/50">
                  <svg className="w-8 h-8 text-emerald-500/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
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
                  className={`max-w-[85%] px-4 py-2.5 rounded-2xl ${
                    msg.role === 'user'
                      ? 'bg-gradient-to-br from-emerald-600 to-emerald-700 text-white rounded-br-md'
                      : 'bg-slate-800/80 text-slate-200 rounded-bl-md border border-slate-700/50'
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
                <div className="bg-slate-800/80 px-4 py-3 rounded-2xl rounded-bl-md border border-slate-700/50">
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

          <div className="relative p-4 border-t border-slate-700/50 bg-slate-800/30">
            <div className="flex gap-2">
              <textarea
                value={input}
                onChange={(e) => setinput(e.target.value)}
                onKeyDown={handlekeydown}
                placeholder="Ask about ML concepts..."
                className="flex-1 px-4 py-2.5 bg-slate-800/80 border border-slate-700/50 rounded-xl resize-none focus:outline-none focus:border-emerald-500/50 text-sm text-white placeholder-slate-500 font-mono"
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
