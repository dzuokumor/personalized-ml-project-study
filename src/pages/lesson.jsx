import { useParams, Link, useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { getcoursebyid } from '../data/courses'
import { usestore } from '../store/usestore'
import { useauth } from '../contexts/authcontext'
import { usestats } from '../hooks/usestats'
import { useprogress } from '../hooks/useprogress'
import LessonContent from '../components/course/lessoncontent'
import Quiz from '../components/course/quiz'
import Codeexecutor from '../components/course/codeexecutor'
import Login from '../components/auth/login'
import Discussionboard from '../components/course/discussionboard'

export default function lesson() {
  const { courseid, lessonid } = useParams()
  const navigate = useNavigate()
  const [showquiz, setshowquiz] = useState(false)
  const [isretake, setisretake] = useState(false)
  const [note, setnote] = useState('')
  const [showlogin, setshowlogin] = useState(false)

  const { user, loading } = useauth()
  const { completelesson, passquiz, completecourse } = usestats()
  const { markcomplete: markprogresscomplete, iscomplete: isprogresscomplete } = useprogress()

  const markcomplete = usestore((state) => state.markcomplete)
  const islessonComplete = usestore((state) => state.islessonComplete)
  const progress = usestore((state) => state.progress)
  const isbookmarked = usestore((state) => state.isbookmarked)
  const addbookmark = usestore((state) => state.addbookmark)
  const removebookmark = usestore((state) => state.removebookmark)
  const savenote = usestore((state) => state.savenote)
  const getnote = usestore((state) => state.getnote)
  const savequizscore = usestore((state) => state.savequizscore)
  const getquizscore = usestore((state) => state.getquizscore)

  const course = getcoursebyid(courseid)
  const lesson = course?.lessons.find(l => l.id === lessonid)
  const lessonidx = course?.lessons.findIndex(l => l.id === lessonid)

  const existingnote = getnote(courseid, lessonid)

  useEffect(() => {
    setshowquiz(false)
    setisretake(false)
    setnote('')
  }, [courseid, lessonid])

  if (!course || !lesson) {
    return (
      <div className="text-center py-12">
        <h2 className="text-xl text-slate-600">Lesson not found</h2>
        <Link to="/" className="text-emerald-600 hover:underline mt-4 inline-block">
          Back to home
        </Link>
      </div>
    )
  }

  if (!loading && !user) {
    return (
      <div className="max-w-2xl mx-auto text-center py-16">
        <div className="w-24 h-24 bg-gradient-to-br from-emerald-100 to-emerald-200 rounded-full mx-auto mb-6 flex items-center justify-center">
          <svg className="w-12 h-12 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/>
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-slate-900 mb-3">Sign in to Access Lessons</h2>
        <p className="text-slate-500 mb-8 max-w-md mx-auto">
          Create a free account to unlock all lessons, track your progress, and earn achievements.
        </p>
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <button
            onClick={() => setshowlogin(true)}
            className="px-8 py-3 bg-emerald-600 text-white font-medium rounded-xl hover:bg-emerald-700 transition-colors"
          >
            Sign in to Continue
          </button>
          <Link
            to="/"
            className="px-8 py-3 bg-slate-100 text-slate-700 font-medium rounded-xl hover:bg-slate-200 transition-colors"
          >
            Back to Home
          </Link>
        </div>
        {showlogin && <Login onclose={() => setshowlogin(false)} />}
      </div>
    )
  }

  const prevlesson = lessonidx > 0 ? course.lessons[lessonidx - 1] : null
  const nextlesson = lessonidx < course.lessons.length - 1 ? course.lessons[lessonidx + 1] : null
  const iscomplete = islessonComplete(courseid, lessonid)
  const bookmarked = isbookmarked(courseid, lessonid)
  const existingquizscore = getquizscore(courseid, lessonid)

  const checkcoursecompletion = () => {
    if (!course) return
    const courseprogress = progress[courseid] || {}
    const completedcount = Object.values(courseprogress).filter(Boolean).length + 1
    if (completedcount >= course.lessons.length) {
      completecourse(courseid)
    }
  }

  const handlemarkcomplete = () => {
    if (!islessonComplete(courseid, lessonid)) {
      markcomplete(courseid, lessonid)
      markprogresscomplete(courseid, lessonid)
      completelesson(courseid, lessonid)
      checkcoursecompletion()
    }
    if (nextlesson) {
      navigate(`/course/${courseid}/lesson/${nextlesson.id}`)
    }
  }

  const handlequizcomplete = (score, isretakequiz) => {
    savequizscore(courseid, lessonid, score, lesson.quiz.length)
    if (!islessonComplete(courseid, lessonid)) {
      markcomplete(courseid, lessonid)
      markprogresscomplete(courseid, lessonid)
      completelesson(courseid, lessonid)
      checkcoursecompletion()
    }
    if (!isretakequiz && score >= Math.ceil(lesson.quiz.length * 0.7)) {
      passquiz(score, lesson.quiz.length)
    }
  }

  const handlebookmark = () => {
    if (bookmarked) {
      removebookmark(courseid, lessonid)
    } else {
      addbookmark(courseid, lessonid)
    }
  }

  const handlesavenote = () => {
    savenote(courseid, lessonid, note || existingnote)
  }

  return (
    <div className="max-w-4xl">
      <div className="flex items-center justify-between gap-2 mb-4 sm:mb-6">
        <div className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm text-slate-500 min-w-0 overflow-hidden">
          <Link to="/" className="hover:text-slate-700 shrink-0">Home</Link>
          <span className="shrink-0">/</span>
          <Link to={`/course/${courseid}`} className="hover:text-slate-700 truncate">{course.title}</Link>
          <span className="shrink-0 hidden sm:inline">/</span>
          <span className="text-slate-900 truncate hidden sm:inline">{lesson.title}</span>
        </div>
        <button
          onClick={handlebookmark}
          className={`p-2 rounded-lg transition-colors shrink-0 ${
            bookmarked ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-500 hover:bg-slate-200'
          }`}
          title={bookmarked ? 'Remove bookmark' : 'Add bookmark'}
        >
          <svg className="w-5 h-5" fill={bookmarked ? 'currentColor' : 'none'} viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
          </svg>
        </button>
      </div>

      <div className="glass-card rounded-xl p-4 sm:p-8 mb-4 sm:mb-6">
        {lesson.concepts && lesson.concepts.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 mb-3 sm:mb-4">
            {lesson.concepts.map((concept, idx) => (
              <span key={idx} className="text-xs bg-emerald-100 text-emerald-700 px-2 py-1 rounded-full">
                {concept}
              </span>
            ))}
          </div>
        )}

        <h1 className="text-xl sm:text-3xl font-bold text-slate-900 mb-2">{lesson.title}</h1>
        <p className="text-sm sm:text-base text-slate-500">{lesson.duration}</p>
      </div>

      {!showquiz ? (
        <>
          <div className="glass-card rounded-xl p-4 sm:p-8 mb-4 sm:mb-6">
            <LessonContent content={lesson.content} />
          </div>

          <div className="glass-card rounded-xl p-4 sm:p-6 mb-4 sm:mb-6">
            <h3 className="text-base sm:text-lg font-semibold text-slate-900 mb-3">Notes</h3>
            <textarea
              defaultValue={existingnote}
              onChange={(e) => setnote(e.target.value)}
              placeholder="Add notes for this lesson..."
              className="w-full h-32 p-3 border border-slate-200 rounded-lg text-sm resize-none focus:outline-none focus:border-emerald-500"
            />
            <button
              onClick={handlesavenote}
              className="mt-2 px-4 py-2 bg-slate-100 text-slate-700 rounded-lg text-sm font-medium hover:bg-slate-200 transition-colors"
            >
              Save Note
            </button>
          </div>

          <div className="glass-card rounded-xl p-4 sm:p-6 mb-4 sm:mb-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 border-2 border-blue-200 bg-blue-50/50 rounded-xl flex items-center justify-center shrink-0">
                <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <h3 className="text-base sm:text-lg font-semibold text-slate-900">Python Practice</h3>
                  <span className="text-[10px] px-2 py-0.5 bg-slate-100 text-slate-500 rounded-full flex items-center gap-1">
                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                    </svg>
                    Builds portfolio
                  </span>
                </div>
                <p className="text-xs sm:text-sm text-slate-500">Code you run here is saved and can be published to GitHub</p>
              </div>
            </div>
            <Codeexecutor
              initialcode={lesson.startercode || '# Practice Python code here\n\n'}
              courseid={courseid}
              lessonid={lessonid}
              lessontitle={lesson.title}
            />
          </div>

          {lesson.quiz && lesson.quiz.length > 0 && (
            <div className="glass-card rounded-xl p-4 sm:p-8 mb-4 sm:mb-6">
              {existingquizscore ? (
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                        existingquizscore.score >= Math.ceil(lesson.quiz.length * 0.7)
                          ? 'bg-emerald-100'
                          : 'bg-amber-100'
                      }`}>
                        {existingquizscore.score >= Math.ceil(lesson.quiz.length * 0.7) ? (
                          <svg className="w-6 h-6 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        ) : (
                          <svg className="w-6 h-6 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        )}
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-slate-900">Quiz Completed</h3>
                        <p className="text-slate-500 text-sm">
                          Score: {existingquizscore.score}/{existingquizscore.total} ({Math.round((existingquizscore.score / existingquizscore.total) * 100)}%)
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => { setshowquiz(true); setisretake(true) }}
                      className="px-5 py-2.5 bg-slate-100 text-slate-700 rounded-xl font-medium hover:bg-slate-200 transition-colors"
                    >
                      Retake Quiz
                    </button>
                  </div>
                  <p className="text-xs text-slate-400">
                    Note: Retaking will update your score but won't add to your XP or quiz stats.
                  </p>
                </div>
              ) : (
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-slate-900">Test Knowledge</h3>
                    <p className="text-slate-500 text-sm">{lesson.quiz.length} questions</p>
                  </div>
                  <button
                    onClick={() => setshowquiz(true)}
                    className="px-5 py-2.5 btn-primary text-white rounded-xl font-medium"
                  >
                    Start Quiz
                  </button>
                </div>
              )}
            </div>
          )}
        </>
      ) : (
        <div className="mb-6">
          <button
            onClick={() => { setshowquiz(false); setisretake(false) }}
            className="text-sm text-slate-500 hover:text-slate-700 mb-4 inline-flex items-center gap-1"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to lesson
          </button>
          {isretake && (
            <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
              <p className="text-sm text-amber-800">
                <strong>Retake Mode:</strong> Your score will be updated but XP and quiz stats won't change.
              </p>
            </div>
          )}
          <Quiz
            questions={lesson.quiz}
            onComplete={(score) => handlequizcomplete(score, isretake)}
            coursetitle={lesson.title}
            isretake={isretake}
          />
        </div>
      )}

      <div className="flex items-center justify-between mb-6">
        <div>
          {prevlesson && (
            <Link
              to={`/course/${courseid}/lesson/${prevlesson.id}`}
              className="inline-flex items-center gap-2 text-slate-600 hover:text-slate-900"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              <span>{prevlesson.title}</span>
            </Link>
          )}
        </div>

        <div className="flex items-center gap-4">
          {nextlesson ? (
            <button
              onClick={handlemarkcomplete}
              className="inline-flex items-center gap-2 px-6 py-2 bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 transition-colors"
            >
              <span>{iscomplete ? 'Next' : 'Complete & Next'}</span>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          ) : (
            !iscomplete && (
              <button
                onClick={handlemarkcomplete}
                className="px-6 py-2 bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 transition-colors"
              >
                Mark Complete
              </button>
            )
          )}
        </div>
      </div>

      <Discussionboard courseid={courseid} lessonid={lessonid} />
    </div>
  )
}
