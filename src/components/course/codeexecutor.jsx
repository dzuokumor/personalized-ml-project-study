import { useState } from 'react'
import Editor from '@monaco-editor/react'

export default function codeexecutor({ initialcode = '', expectedoutput = null }) {
  const [code, setcode] = useState(initialcode)
  const [output, setoutput] = useState('')
  const [error, seterror] = useState('')
  const [running, setrunning] = useState(false)
  const [passed, setpassed] = useState(null)

  const runcode = async () => {
    setrunning(true)
    setoutput('')
    seterror('')
    setpassed(null)

    try {
      const response = await fetch('/api/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      })

      const data = await response.json()

      if (data.error) {
        seterror(data.error)
      } else {
        setoutput(data.output || '(No output)')

        if (expectedoutput !== null) {
          const normalizedexpected = expectedoutput.trim()
          const normalizedoutput = (data.output || '').trim()
          setpassed(normalizedexpected === normalizedoutput)
        }
      }
    } catch (err) {
      seterror('Failed to connect to the code execution service. Please try again.')
    } finally {
      setrunning(false)
    }
  }

  const resetcode = () => {
    setcode(initialcode)
    setoutput('')
    seterror('')
    setpassed(null)
  }

  return (
    <div className="border border-slate-200 rounded-xl overflow-hidden bg-white">
      <div className="flex items-center justify-between px-4 py-2 bg-slate-800 text-white">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <span className="w-3 h-3 rounded-full bg-red-500"></span>
            <span className="w-3 h-3 rounded-full bg-yellow-500"></span>
            <span className="w-3 h-3 rounded-full bg-green-500"></span>
          </div>
          <span className="text-sm text-slate-400 ml-2">Python Editor</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={resetcode}
            className="px-3 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded transition-colors"
          >
            Reset
          </button>
          <button
            onClick={runcode}
            disabled={running}
            className="px-4 py-1 text-xs bg-emerald-600 hover:bg-emerald-700 rounded flex items-center gap-1 transition-colors disabled:opacity-50"
          >
            {running ? (
              <>
                <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Running...
              </>
            ) : (
              <>
                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z"/>
                </svg>
                Run Code
              </>
            )}
          </button>
        </div>
      </div>

      <div className="h-64">
        <Editor
          height="100%"
          defaultLanguage="python"
          value={code}
          onChange={(value) => setcode(value || '')}
          theme="vs-dark"
          options={{
            minimap: { enabled: false },
            fontSize: 14,
            fontFamily: "'JetBrains Mono', monospace",
            lineNumbers: 'on',
            scrollBeyondLastLine: false,
            padding: { top: 12 },
            automaticLayout: true
          }}
        />
      </div>

      {(output || error) && (
        <div className="border-t border-slate-700">
          <div className="px-4 py-2 bg-slate-900 text-xs text-slate-400 flex items-center justify-between">
            <span>Output</span>
            {passed !== null && (
              <span className={`px-2 py-0.5 rounded ${passed ? 'bg-green-600 text-white' : 'bg-red-600 text-white'}`}>
                {passed ? 'Correct!' : 'Try again'}
              </span>
            )}
          </div>
          <div className="bg-slate-900 p-4 font-mono text-sm max-h-48 overflow-auto">
            {error ? (
              <pre className="text-red-400 whitespace-pre-wrap">{error}</pre>
            ) : (
              <pre className="text-green-400 whitespace-pre-wrap">{output}</pre>
            )}
          </div>
        </div>
      )}

      {expectedoutput !== null && (
        <div className="px-4 py-2 bg-slate-100 border-t border-slate-200 text-xs text-slate-500">
          Expected output: <code className="bg-slate-200 px-1 py-0.5 rounded">{expectedoutput}</code>
        </div>
      )}
    </div>
  )
}
