import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

export default function CodeBlock({ code, language = 'python', filename = null, fromproject = null }) {
  const customstyle = {
    ...vscDarkPlus,
    'pre[class*="language-"]': {
      ...vscDarkPlus['pre[class*="language-"]'],
      background: '#1e1e1e',
      margin: 0,
      padding: '1rem',
      fontSize: '0.875rem',
      lineHeight: '1.6',
      fontFamily: "'JetBrains Mono', monospace"
    },
    'code[class*="language-"]': {
      ...vscDarkPlus['code[class*="language-"]'],
      fontFamily: "'JetBrains Mono', monospace"
    }
  }

  return (
    <div className="my-6 rounded-lg overflow-hidden border border-slate-200">
      {(filename || fromproject) && (
        <div className="bg-slate-800 px-4 py-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
            </div>
            {filename && (
              <span className="text-slate-400 text-xs font-mono ml-3">{filename}</span>
            )}
          </div>
          {fromproject && (
            <span className="text-xs text-slate-500">from: {fromproject}</span>
          )}
        </div>
      )}
      <SyntaxHighlighter
        language={language}
        style={customstyle}
        showLineNumbers
        wrapLines
      >
        {code.trim()}
      </SyntaxHighlighter>
    </div>
  )
}
