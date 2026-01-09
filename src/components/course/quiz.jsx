import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useauth } from '../../contexts/authcontext'

const optionlabels = ['A', 'B', 'C', 'D', 'E', 'F']

export default function Quiz({ questions, onComplete, coursetitle = 'Assessment' }) {
  const { githubconnected, connectgithubforrepos } = useauth()
  const [currentidx, setcurrentidx] = useState(0)
  const [selected, setselected] = useState(null)
  const [showresult, setshowresult] = useState(false)
  const [answers, setanswers] = useState([])
  const [completed, setcompleted] = useState(false)
  const [timeelapsed, settimeelapsed] = useState(0)
  const [starttime] = useState(Date.now())

  const current = questions[currentidx]
  const score = answers.filter((a, i) => a === questions[i].correct).length

  useEffect(() => {
    if (completed) return
    const interval = setInterval(() => {
      settimeelapsed(Math.floor((Date.now() - starttime) / 1000))
    }, 1000)
    return () => clearInterval(interval)
  }, [completed, starttime])

  const formattime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const handleselect = (idx) => {
    if (showresult) return
    setselected(idx)
  }

  const handlesubmit = () => {
    if (selected === null) return
    setshowresult(true)
    setanswers([...answers, selected])
  }

  const handlenext = () => {
    if (currentidx < questions.length - 1) {
      setcurrentidx(prev => prev + 1)
      setselected(null)
      setshowresult(false)
    } else {
      setcompleted(true)
      const finalscore = answers.length > 0
        ? answers.filter((a, i) => a === questions[i].correct).length + (selected === current.correct ? 1 : 0)
        : (selected === current.correct ? 1 : 0)
      if (onComplete) onComplete(finalscore)
    }
  }

  if (completed) {
    const finalscore = score + (answers.length < questions.length && selected === current.correct ? 1 : 0)
    const percentage = Math.round((finalscore / questions.length) * 100)
    const passed = percentage >= 70
    const avgtime = Math.round(timeelapsed / questions.length)

    return (
      <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm">
        <div className={`px-8 py-6 ${passed ? 'bg-gradient-to-r from-emerald-500 to-emerald-600' : 'bg-gradient-to-r from-amber-500 to-orange-500'}`}>
          <div className="flex items-center justify-between text-white">
            <div>
              <p className="text-sm opacity-90 mb-1">Assessment Complete</p>
              <h2 className="text-2xl font-bold">{coursetitle}</h2>
            </div>
            <div className="text-right">
              <p className="text-sm opacity-90 mb-1">Final Score</p>
              <p className="text-3xl font-bold">{percentage}%</p>
            </div>
          </div>
        </div>

        <div className="p-8">
          <div className="flex items-center justify-center mb-8">
            <div className={`w-32 h-32 rounded-full flex items-center justify-center ${
              passed ? 'bg-emerald-50 border-4 border-emerald-200' : 'bg-amber-50 border-4 border-amber-200'
            }`}>
              {passed ? (
                <svg className="w-16 h-16 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              ) : (
                <svg className="w-16 h-16 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
            </div>
          </div>

          <div className="text-center mb-8">
            <h3 className="text-2xl font-bold text-slate-900 mb-2">
              {passed ? 'Congratulations!' : 'Keep Learning!'}
            </h3>
            <p className="text-slate-600 max-w-md mx-auto">
              {passed
                ? 'You have demonstrated strong understanding of this material. Great work!'
                : 'Review the material and try again. You need 70% to pass this assessment.'}
            </p>
          </div>

          <div className="grid grid-cols-4 gap-4 mb-8">
            <div className="bg-slate-50 rounded-xl p-4 text-center">
              <p className="text-2xl font-bold text-slate-900">{finalscore}/{questions.length}</p>
              <p className="text-xs text-slate-500 mt-1">Correct Answers</p>
            </div>
            <div className="bg-slate-50 rounded-xl p-4 text-center">
              <p className="text-2xl font-bold text-slate-900">{percentage}%</p>
              <p className="text-xs text-slate-500 mt-1">Score</p>
            </div>
            <div className="bg-slate-50 rounded-xl p-4 text-center">
              <p className="text-2xl font-bold text-slate-900">{formattime(timeelapsed)}</p>
              <p className="text-xs text-slate-500 mt-1">Total Time</p>
            </div>
            <div className="bg-slate-50 rounded-xl p-4 text-center">
              <p className="text-2xl font-bold text-slate-900">{avgtime}s</p>
              <p className="text-xs text-slate-500 mt-1">Avg per Question</p>
            </div>
          </div>

          <div className="border-t border-slate-200 pt-6">
            <h4 className="text-sm font-semibold text-slate-700 mb-4">Question Breakdown</h4>
            <div className="space-y-2">
              {questions.map((q, idx) => {
                const useranswer = answers[idx]
                const iscorrect = useranswer === q.correct
                return (
                  <div key={idx} className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      iscorrect ? 'bg-emerald-100 text-emerald-600' : 'bg-red-100 text-red-600'
                    }`}>
                      {iscorrect ? (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-slate-700 truncate">Q{idx + 1}: {q.question}</p>
                      {!iscorrect && (
                        <p className="text-xs text-slate-500 mt-0.5">
                          Correct: {optionlabels[q.correct]}. {q.options[q.correct]}
                        </p>
                      )}
                    </div>
                    <span className={`text-xs font-medium px-2 py-1 rounded ${
                      iscorrect ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'
                    }`}>
                      {iscorrect ? 'Correct' : 'Incorrect'}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

          {passed && (
            <div className="border-t border-slate-200 pt-6 mt-6">
              <div className="flex items-center justify-between p-4 bg-slate-900 rounded-xl">
                <div className="flex items-center gap-3">
                  <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                  </svg>
                  <div>
                    <p className="text-white font-medium text-sm">Share your achievement</p>
                    <p className="text-slate-400 text-xs">Add badges to your GitHub profile</p>
                  </div>
                </div>
                {githubconnected ? (
                  <Link
                    to="/profile?tab=github"
                    className="px-4 py-2 bg-white text-slate-900 rounded-lg text-sm font-medium hover:bg-slate-100 transition-colors"
                  >
                    Share on GitHub
                  </Link>
                ) : (
                  <button
                    onClick={connectgithubforrepos}
                    className="px-4 py-2 bg-white text-slate-900 rounded-lg text-sm font-medium hover:bg-slate-100 transition-colors"
                  >
                    Connect GitHub
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm">
      <div className="bg-gradient-to-r from-slate-800 to-slate-900 px-6 py-4">
        <div className="flex items-center justify-between text-white">
          <div>
            <p className="text-xs text-slate-400 mb-0.5">Assessment</p>
            <h2 className="font-semibold">{coursetitle}</h2>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-right">
              <p className="text-xs text-slate-400 mb-0.5">Time</p>
              <p className="font-mono text-lg">{formattime(timeelapsed)}</p>
            </div>
            <div className="text-right">
              <p className="text-xs text-slate-400 mb-0.5">Progress</p>
              <p className="font-mono text-lg">{currentidx + 1}/{questions.length}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="h-1 bg-slate-200">
        <div
          className="h-full bg-emerald-500 transition-all duration-300"
          style={{ width: `${((currentidx + (showresult ? 1 : 0)) / questions.length) * 100}%` }}
        />
      </div>

      <div className="p-8">
        <div className="flex items-start gap-4 mb-6">
          <div className="w-10 h-10 bg-slate-100 rounded-xl flex items-center justify-center flex-shrink-0">
            <span className="font-bold text-slate-700">{currentidx + 1}</span>
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                Question {currentidx + 1} of {questions.length}
              </span>
              {current.difficulty && (
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  current.difficulty === 'easy' ? 'bg-green-100 text-green-700' :
                  current.difficulty === 'medium' ? 'bg-amber-100 text-amber-700' :
                  'bg-red-100 text-red-700'
                }`}>
                  {current.difficulty}
                </span>
              )}
            </div>
            <h3 className="text-xl font-semibold text-slate-900 leading-relaxed">{current.question}</h3>
          </div>
        </div>

        <div className="space-y-3 mb-8">
          {current.options.map((option, idx) => {
            let styles = 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
            let labelstyles = 'bg-slate-100 text-slate-600'
            let iconstyles = null

            if (showresult) {
              if (idx === current.correct) {
                styles = 'border-emerald-500 bg-emerald-50'
                labelstyles = 'bg-emerald-500 text-white'
                iconstyles = 'correct'
              } else if (idx === selected && idx !== current.correct) {
                styles = 'border-red-500 bg-red-50'
                labelstyles = 'bg-red-500 text-white'
                iconstyles = 'incorrect'
              }
            } else if (selected === idx) {
              styles = 'border-emerald-500 bg-emerald-50'
              labelstyles = 'bg-emerald-500 text-white'
            }

            return (
              <button
                key={idx}
                onClick={() => handleselect(idx)}
                disabled={showresult}
                className={`w-full text-left p-4 rounded-xl border-2 transition-all duration-200 flex items-center gap-4 ${styles} ${
                  showresult ? 'cursor-default' : 'cursor-pointer'
                }`}
              >
                <div className={`w-9 h-9 rounded-lg flex items-center justify-center font-bold text-sm transition-colors ${labelstyles}`}>
                  {optionlabels[idx]}
                </div>
                <span className="flex-1 text-slate-700 font-medium">{option}</span>
                {iconstyles === 'correct' && (
                  <svg className="w-6 h-6 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                )}
                {iconstyles === 'incorrect' && (
                  <svg className="w-6 h-6 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                )}
              </button>
            )
          })}
        </div>

        {showresult && current.explanation && (
          <div className="mb-8 p-5 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-100">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center flex-shrink-0">
                <svg className="w-4 h-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-semibold text-blue-900 mb-1">Explanation</p>
                <p className="text-sm text-blue-800 leading-relaxed">{current.explanation}</p>
              </div>
            </div>
          </div>
        )}

        <div className="flex items-center justify-between pt-4 border-t border-slate-200">
          <div className="flex items-center gap-2">
            {questions.map((_, idx) => (
              <div
                key={idx}
                className={`w-3 h-3 rounded-full transition-colors ${
                  idx < currentidx ? 'bg-emerald-500' :
                  idx === currentidx ? 'bg-emerald-400 ring-4 ring-emerald-100' :
                  'bg-slate-200'
                }`}
              />
            ))}
          </div>

          <div className="flex items-center gap-3">
            {!showresult ? (
              <button
                onClick={handlesubmit}
                disabled={selected === null}
                className="px-8 py-3 bg-emerald-600 text-white rounded-xl font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-emerald-700 transition-colors flex items-center gap-2"
              >
                Submit Answer
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            ) : (
              <button
                onClick={handlenext}
                className="px-8 py-3 bg-emerald-600 text-white rounded-xl font-semibold hover:bg-emerald-700 transition-colors flex items-center gap-2"
              >
                {currentidx < questions.length - 1 ? 'Next Question' : 'View Results'}
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
