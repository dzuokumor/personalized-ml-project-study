import { useParams, Link, useNavigate } from 'react-router-dom'
import { useState, useEffect } from 'react'
import { courses } from '../data/courses'
import { usestore } from '../store/usestore'
import LessonContent from '../components/course/lessoncontent'
import Quiz from '../components/course/quiz'

export default function Lesson() {
  const { courseid, lessonid } = useParams()
  const navigate = useNavigate()
  const [showquiz, setshowquiz] = useState(false)
  const [note, setnote] = useState('')

  const markcomplete = usestore((state) => state.markcomplete)
  const islessonComplete = usestore((state) => state.islessonComplete)
  const isbookmarked = usestore((state) => state.isbookmarked)
  const addbookmark = usestore((state) => state.addbookmark)
  const removebookmark = usestore((state) => state.removebookmark)
  const savenote = usestore((state) => state.savenote)
  const getnote = usestore((state) => state.getnote)
  const savequizscore = usestore((state) => state.savequizscore)

  const course = courses.find(c => c.id === courseid)
  const lesson = course?.lessons.find(l => l.id === lessonid)
  const lessonidx = course?.lessons.findIndex(l => l.id === lessonid)

  const existingnote = getnote(courseid, lessonid)

  useEffect(() => {
    setshowquiz(false)
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

  const prevlesson = lessonidx > 0 ? course.lessons[lessonidx - 1] : null
  const nextlesson = lessonidx < course.lessons.length - 1 ? course.lessons[lessonidx + 1] : null
  const iscomplete = islessonComplete(courseid, lessonid)
  const bookmarked = isbookmarked(courseid, lessonid)

  const handlemarkcomplete = () => {
    markcomplete(courseid, lessonid)
    if (nextlesson) {
      navigate(`/course/${courseid}/lesson/${nextlesson.id}`)
    }
  }

  const handlequizcomplete = (score) => {
    savequizscore(courseid, lessonid, score, lesson.quiz.length)
    markcomplete(courseid, lessonid)
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
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Link to="/" className="hover:text-slate-700">Home</Link>
          <span>/</span>
          <Link to={`/course/${courseid}`} className="hover:text-slate-700">{course.title}</Link>
          <span>/</span>
          <span className="text-slate-900">{lesson.title}</span>
        </div>
        <button
          onClick={handlebookmark}
          className={`p-2 rounded-lg transition-colors ${
            bookmarked ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-500 hover:bg-slate-200'
          }`}
          title={bookmarked ? 'Remove bookmark' : 'Add bookmark'}
        >
          <svg className="w-5 h-5" fill={bookmarked ? 'currentColor' : 'none'} viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
          </svg>
        </button>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-8 mb-6">
        <div className="flex items-center gap-3 mb-4">
          {lesson.concepts.map((concept, idx) => (
            <span key={idx} className="text-xs bg-emerald-100 text-emerald-700 px-2 py-1 rounded-full">
              {concept}
            </span>
          ))}
        </div>

        <h1 className="text-3xl font-bold text-slate-900 mb-2">{lesson.title}</h1>
        <p className="text-slate-500">{lesson.duration}</p>
      </div>

      {!showquiz ? (
        <>
          <div className="bg-white border border-slate-200 rounded-xl p-8 mb-6">
            <LessonContent content={lesson.content} />
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-6 mb-6">
            <h3 className="text-lg font-semibold text-slate-900 mb-3">Notes</h3>
            <textarea
              defaultValue={existingnote}
              onChange={(e) => setnote(e.target.value)}
              placeholder="Add your notes for this lesson..."
              className="w-full h-32 p-3 border border-slate-200 rounded-lg text-sm resize-none focus:outline-none focus:border-emerald-500"
            />
            <button
              onClick={handlesavenote}
              className="mt-2 px-4 py-2 bg-slate-100 text-slate-700 rounded-lg text-sm font-medium hover:bg-slate-200 transition-colors"
            >
              Save Note
            </button>
          </div>

          {lesson.quiz && lesson.quiz.length > 0 && (
            <div className="bg-white border border-slate-200 rounded-xl p-8 mb-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-slate-900">Test Your Knowledge</h3>
                  <p className="text-slate-500 text-sm">{lesson.quiz.length} questions</p>
                </div>
                <button
                  onClick={() => setshowquiz(true)}
                  className="px-4 py-2 bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 transition-colors"
                >
                  Start Quiz
                </button>
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="mb-6">
          <button
            onClick={() => setshowquiz(false)}
            className="text-sm text-slate-500 hover:text-slate-700 mb-4 inline-flex items-center gap-1"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to lesson
          </button>
          <Quiz questions={lesson.quiz} onComplete={handlequizcomplete} />
        </div>
      )}

      <div className="flex items-center justify-between">
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
          {!iscomplete && (
            <button
              onClick={handlemarkcomplete}
              className="px-6 py-2 bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 transition-colors"
            >
              {nextlesson ? 'Complete & Continue' : 'Mark Complete'}
            </button>
          )}

          {nextlesson && iscomplete && (
            <Link
              to={`/course/${courseid}/lesson/${nextlesson.id}`}
              className="inline-flex items-center gap-2 px-6 py-2 bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 transition-colors"
            >
              <span>Next: {nextlesson.title}</span>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          )}
        </div>
      </div>
    </div>
  )
}
