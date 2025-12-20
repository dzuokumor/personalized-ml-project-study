import { useState } from 'react'

export default function Quiz({ questions, onComplete }) {
  const [currentidx, setcurrentidx] = useState(0)
  const [selected, setselected] = useState(null)
  const [showresult, setshowresult] = useState(false)
  const [score, setscore] = useState(0)
  const [completed, setcompleted] = useState(false)

  const current = questions[currentidx]

  const handleselect = (idx) => {
    if (showresult) return
    setselected(idx)
  }

  const handlesubmit = () => {
    if (selected === null) return

    setshowresult(true)
    if (selected === current.correct) {
      setscore(prev => prev + 1)
    }
  }

  const handlenext = () => {
    if (currentidx < questions.length - 1) {
      setcurrentidx(prev => prev + 1)
      setselected(null)
      setshowresult(false)
    } else {
      setcompleted(true)
      if (onComplete) onComplete(score + (selected === current.correct ? 1 : 0))
    }
  }

  if (completed) {
    const finalscore = score
    const percentage = Math.round((finalscore / questions.length) * 100)

    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
        <div className={`w-20 h-20 rounded-full mx-auto mb-4 flex items-center justify-center ${
          percentage >= 70 ? 'bg-green-100' : 'bg-amber-100'
        }`}>
          <span className={`text-2xl font-bold ${
            percentage >= 70 ? 'text-green-600' : 'text-amber-600'
          }`}>{percentage}%</span>
        </div>
        <h3 className="text-xl font-semibold text-slate-900 mb-2">
          {percentage >= 70 ? 'Well done!' : 'Keep practicing!'}
        </h3>
        <p className="text-slate-600">
          You got {finalscore} out of {questions.length} questions correct.
        </p>
      </div>
    )
  }

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6">
      <div className="flex items-center justify-between mb-6">
        <span className="text-sm text-slate-500">
          Question {currentidx + 1} of {questions.length}
        </span>
        <div className="flex gap-1">
          {questions.map((_, idx) => (
            <div
              key={idx}
              className={`w-2 h-2 rounded-full ${
                idx < currentidx ? 'bg-emerald-600' :
                idx === currentidx ? 'bg-emerald-400' : 'bg-slate-200'
              }`}
            />
          ))}
        </div>
      </div>

      <h3 className="text-lg font-medium text-slate-900 mb-4">{current.question}</h3>

      <div className="space-y-3 mb-6">
        {current.options.map((option, idx) => {
          let styles = 'border-slate-200 hover:border-slate-300'

          if (showresult) {
            if (idx === current.correct) {
              styles = 'border-green-500 bg-green-50'
            } else if (idx === selected && idx !== current.correct) {
              styles = 'border-red-500 bg-red-50'
            }
          } else if (selected === idx) {
            styles = 'border-emerald-500 bg-emerald-50'
          }

          return (
            <button
              key={idx}
              onClick={() => handleselect(idx)}
              className={`w-full text-left p-4 rounded-lg border-2 transition-colors ${styles}`}
            >
              <span className="text-slate-700">{option}</span>
            </button>
          )
        })}
      </div>

      {showresult && current.explanation && (
        <div className="mb-6 p-4 bg-slate-50 rounded-lg">
          <p className="text-sm text-slate-600">{current.explanation}</p>
        </div>
      )}

      <div className="flex justify-end">
        {!showresult ? (
          <button
            onClick={handlesubmit}
            disabled={selected === null}
            className="px-6 py-2 bg-emerald-600 text-white rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-emerald-700 transition-colors"
          >
            Check Answer
          </button>
        ) : (
          <button
            onClick={handlenext}
            className="px-6 py-2 bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 transition-colors"
          >
            {currentidx < questions.length - 1 ? 'Next Question' : 'See Results'}
          </button>
        )}
      </div>
    </div>
  )
}
