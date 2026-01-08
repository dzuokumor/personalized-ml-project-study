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
        className="fixed bottom-6 right-6 w-14 h-14 btn-primary text-white rounded-full hover:scale-105 transition-all flex items-center justify-center z-50 glow-emerald"
      >
        {isopen ? (
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
        )}
      </button>

      {isopen && (
        <div className="fixed bottom-24 right-6 w-96 h-[500px] glass-card rounded-2xl flex flex-col z-50 overflow-hidden shadow-2xl">
          <div className="px-4 py-3 bg-gradient-to-r from-emerald-600 to-teal-600 text-white flex items-center justify-between">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              <span className="font-medium">ML Tutor</span>
            </div>
            <button
              onClick={() => setmessages([])}
              className="text-emerald-200 hover:text-white text-sm"
            >
              Clear
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 && (
              <div className="text-center text-slate-400 mt-8">
                <svg className="w-12 h-12 mx-auto mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                <p className="text-sm">Ask me anything about ML!</p>
                <p className="text-xs mt-1">I can help explain concepts, debug code, or answer questions.</p>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[85%] px-4 py-2 rounded-2xl ${
                    msg.role === 'user'
                      ? 'bg-emerald-600 text-white rounded-br-md'
                      : 'bg-slate-100 text-slate-800 rounded-bl-md'
                  }`}
                >
                  {msg.role === 'assistant' ? (
                    <div className="prose prose-sm prose-slate max-w-none">
                      <ReactMarkdown
                        components={{
                          code: ({ node, inline, className, children, ...props }) => {
                            if (inline) {
                              return (
                                <code className="bg-slate-200 px-1 py-0.5 rounded text-sm font-mono" {...props}>
                                  {children}
                                </code>
                              )
                            }
                            return (
                              <pre className="bg-slate-800 text-slate-100 p-3 rounded-lg overflow-x-auto my-2">
                                <code className="text-sm font-mono" {...props}>
                                  {children}
                                </code>
                              </pre>
                            )
                          },
                          p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>
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
                <div className="bg-slate-100 px-4 py-3 rounded-2xl rounded-bl-md">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                    <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                    <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesendref} />
          </div>

          <div className="p-4 border-t border-slate-200">
            <div className="flex gap-2">
              <textarea
                value={input}
                onChange={(e) => setinput(e.target.value)}
                onKeyDown={handlekeydown}
                placeholder="Ask about ML concepts..."
                className="flex-1 px-4 py-2 search-input rounded-xl resize-none focus:outline-none text-sm"
                rows={1}
              />
              <button
                onClick={sendmessage}
                disabled={loading || !input.trim()}
                className="px-4 py-2 btn-primary text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed"
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
