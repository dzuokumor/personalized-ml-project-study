import CodeBlock from './codeblock'

export default function lessoncontent({ content }) {
  return (
    <div className="prose prose-slate max-w-none w-full overflow-x-hidden">
      {content.map((block, idx) => {
        switch (block.type) {
          case 'heading':
            return (
              <h2 key={idx} className="text-2xl font-semibold text-slate-900 mt-8 mb-4 first:mt-0">
                {block.text || block.content}
              </h2>
            )

          case 'subheading':
            return (
              <h3 key={idx} className="text-xl font-medium text-slate-800 mt-6 mb-3">
                {block.text || block.content}
              </h3>
            )

          case 'text':
          case 'paragraph':
            return (
              <div
                key={idx}
                className="text-slate-600 leading-relaxed mb-4 whitespace-pre-line"
                dangerouslySetInnerHTML={{
                  __html: (block.text || block.content || '')
                    .replace(/\*\*(.*?)\*\*/g, '<strong class="text-slate-800">$1</strong>')
                    .replace(/`([^`]+)`/g, '<code class="bg-slate-100 px-1.5 py-0.5 rounded text-sm font-mono text-slate-700">$1</code>')
                    .replace(/\n\n/g, '</p><p class="mt-4">')
                }}
              />
            )

          case 'list':
            return (
              <ul key={idx} className="list-disc list-inside space-y-2 mb-4 text-slate-600">
                {(block.items || []).map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            )

          case 'code':
            return (
              <CodeBlock
                key={idx}
                code={block.code || block.content}
                language={block.language || 'python'}
                filename={block.filename}
                fromproject={block.fromproject}
              />
            )

          case 'callout':
            const styles = {
              info: 'bg-emerald-50 border-emerald-200 text-emerald-800',
              warning: 'bg-amber-50 border-amber-200 text-amber-800',
              tip: 'bg-green-50 border-green-200 text-green-800',
              note: 'bg-slate-50 border-slate-200 text-slate-800'
            }
            return (
              <div key={idx} className={`p-4 rounded-lg border mb-4 ${styles[block.variant] || styles.note}`}>
                <p className="font-medium mb-1 capitalize">{block.variant || 'note'}</p>
                <p className="text-sm">{block.text || block.content}</p>
              </div>
            )

          case 'keypoints':
            return (
              <div key={idx} className="bg-slate-100 rounded-xl p-6 my-6">
                <h4 className="font-semibold text-slate-900 mb-3">Key Takeaways</h4>
                <ul className="space-y-2">
                  {(block.points || []).map((point, i) => (
                    <li key={i} className="flex items-start gap-2 text-slate-700">
                      <svg className="w-5 h-5 text-emerald-600 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )

          case 'formula':
            return (
              <div key={idx} className="bg-slate-800 text-white p-4 rounded-lg my-4 font-mono text-center overflow-x-auto">
                <pre className="text-sm md:text-base">{block.formula || block.content}</pre>
              </div>
            )

          case 'image':
            return (
              <figure key={idx} className="my-6">
                <img src={block.src} alt={block.alt} className="rounded-lg border border-slate-200" />
                {block.caption && (
                  <figcaption className="text-sm text-slate-500 text-center mt-2">{block.caption}</figcaption>
                )}
              </figure>
            )

          case 'diagram':
            return (
              <figure key={idx} className="my-8 bg-white border border-slate-200 rounded-xl p-6 overflow-x-auto">
                <div
                  className="flex justify-center"
                  dangerouslySetInnerHTML={{ __html: block.svg }}
                />
                {block.caption && (
                  <figcaption className="text-sm text-slate-500 text-center mt-4">{block.caption}</figcaption>
                )}
              </figure>
            )

          case 'visualization':
            return (
              <div key={idx} className="my-8 bg-gradient-to-br from-slate-50 to-slate-100 border border-slate-200 rounded-xl p-6">
                {block.title && (
                  <h4 className="font-semibold text-slate-800 mb-4 text-center">{block.title}</h4>
                )}
                <div
                  className="flex justify-center items-center"
                  dangerouslySetInnerHTML={{ __html: block.svg }}
                />
                {block.caption && (
                  <p className="text-sm text-slate-500 text-center mt-4">{block.caption}</p>
                )}
              </div>
            )

          case 'table':
            return (
              <div key={idx} className="my-6 overflow-x-auto">
                <table className="min-w-full border-collapse border border-slate-200 rounded-lg overflow-hidden">
                  <thead className="bg-slate-50">
                    <tr>
                      {(block.headers || []).map((header, i) => (
                        <th key={i} className="border border-slate-200 px-4 py-2 text-left text-sm font-semibold text-slate-700">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(block.rows || []).map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                        {row.map((cell, j) => (
                          <td key={j} className="border border-slate-200 px-4 py-2 text-sm text-slate-600">
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {block.caption && (
                  <p className="text-sm text-slate-500 text-center mt-2">{block.caption}</p>
                )}
              </div>
            )

          default:
            return null
        }
      })}
    </div>
  )
}
