import CodeBlock from './codeblock'

export default function LessonContent({ content }) {
  return (
    <div className="prose prose-slate max-w-none">
      {content.map((block, idx) => {
        switch (block.type) {
          case 'heading':
            return (
              <h2 key={idx} className="text-2xl font-semibold text-slate-900 mt-8 mb-4 first:mt-0">
                {block.text}
              </h2>
            )

          case 'subheading':
            return (
              <h3 key={idx} className="text-xl font-medium text-slate-800 mt-6 mb-3">
                {block.text}
              </h3>
            )

          case 'paragraph':
            return (
              <p key={idx} className="text-slate-600 leading-relaxed mb-4">
                {block.text}
              </p>
            )

          case 'list':
            return (
              <ul key={idx} className="list-disc list-inside space-y-2 mb-4 text-slate-600">
                {block.items.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            )

          case 'code':
            return (
              <CodeBlock
                key={idx}
                code={block.code}
                language={block.language || 'python'}
                filename={block.filename}
                fromproject={block.fromproject}
              />
            )

          case 'callout':
            const styles = {
              info: 'bg-blue-50 border-blue-200 text-blue-800',
              warning: 'bg-amber-50 border-amber-200 text-amber-800',
              tip: 'bg-green-50 border-green-200 text-green-800',
              note: 'bg-slate-50 border-slate-200 text-slate-800'
            }
            return (
              <div key={idx} className={`p-4 rounded-lg border mb-4 ${styles[block.variant] || styles.note}`}>
                <p className="font-medium mb-1 capitalize">{block.variant || 'note'}</p>
                <p className="text-sm">{block.text}</p>
              </div>
            )

          case 'keypoints':
            return (
              <div key={idx} className="bg-slate-100 rounded-xl p-6 my-6">
                <h4 className="font-semibold text-slate-900 mb-3">Key Takeaways</h4>
                <ul className="space-y-2">
                  {block.points.map((point, i) => (
                    <li key={i} className="flex items-start gap-2 text-slate-700">
                      <svg className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
              <div key={idx} className="bg-slate-800 text-white p-4 rounded-lg my-4 font-mono text-center">
                {block.formula}
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

          default:
            return null
        }
      })}
    </div>
  )
}
