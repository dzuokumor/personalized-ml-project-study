import { useState } from 'react'
import { glossary } from '../data/glossary'

export default function Glossary() {
  const [search, setsearch] = useState('')
  const [selectedcategory, setselectedcategory] = useState('all')

  const categories = ['all', ...new Set(glossary.map(item => item.category))]

  const filtered = glossary.filter(item => {
    const matchessearch = item.term.toLowerCase().includes(search.toLowerCase()) ||
      item.definition.toLowerCase().includes(search.toLowerCase())
    const matchescategory = selectedcategory === 'all' || item.category === selectedcategory
    return matchessearch && matchescategory
  })

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Glossary</h1>
        <p className="text-slate-600">Key machine learning terms and concepts from my projects.</p>
      </div>

      <div className="flex flex-col md:flex-row gap-4 mb-8">
        <div className="flex-1">
          <input
            type="text"
            value={search}
            onChange={(e) => setsearch(e.target.value)}
            placeholder="Search terms..."
            className="w-full px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
          />
        </div>
        <div className="flex gap-2 flex-wrap">
          {categories.map(category => (
            <button
              key={category}
              onClick={() => setselectedcategory(category)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors capitalize ${
                selectedcategory === category
                  ? 'bg-emerald-600 text-white'
                  : 'bg-white border border-slate-200 text-slate-600 hover:bg-slate-50'
              }`}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-4">
        {filtered.map((item, idx) => (
          <div key={idx} className="bg-white border border-slate-200 rounded-xl p-6">
            <div className="flex items-start justify-between mb-2">
              <h3 className="text-lg font-semibold text-slate-900">{item.term}</h3>
              <span className="text-xs bg-slate-100 text-slate-600 px-2 py-1 rounded-full capitalize">
                {item.category}
              </span>
            </div>
            <p className="text-slate-600 mb-3">{item.definition}</p>
            {item.usedin && (
              <p className="text-sm text-slate-500">
                Used in: <span className="text-emerald-700">{item.usedin.join(', ')}</span>
              </p>
            )}
          </div>
        ))}

        {filtered.length === 0 && (
          <div className="text-center py-12 text-slate-500">
            No terms found matching the search.
          </div>
        )}
      </div>
    </div>
  )
}
