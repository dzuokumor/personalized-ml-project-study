import { useState } from 'react'
import { useauth } from '../../contexts/authcontext'
import { createrepo, generateprojectreadme } from '../../services/github'

export default function publishmodal({ course, onclose }) {
  const { githubtoken, githubconnected, connectgithubforrepos } = useauth()
  const [reponame, setreponame] = useState(course?.title?.toLowerCase().replace(/[^a-z0-9]+/g, '-') || 'ml-project')
  const [description, setdescription] = useState(course?.description || '')
  const [isprivate, setisprivate] = useState(false)
  const [publishing, setpublishing] = useState(false)
  const [success, setsuccess] = useState(null)
  const [error, seterror] = useState(null)

  const handleconnect = async () => {
    await connectgithubforrepos()
  }

  const handlepublish = async () => {
    if (!githubtoken) {
      seterror('Please connect GitHub first')
      return
    }

    setpublishing(true)
    seterror(null)

    const samplecode = course?.lessons?.[0]?.startercode || '# ML Project\nprint("Hello ML!")'
    const readme = generateprojectreadme({
      coursename: course?.title || 'ML Course',
      lessonname: course?.title || 'ML Project',
      description: course?.description || '',
      code: samplecode
    })

    const result = await createrepo({
      token: githubtoken,
      reponame,
      description,
      code: samplecode,
      readme,
      isprivate
    })

    setpublishing(false)

    if (result.error) {
      seterror(result.error)
    } else if (result.url) {
      setsuccess(result.url)
    } else {
      seterror('Failed to create repository')
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl max-w-lg w-full shadow-xl max-h-[90vh] overflow-y-auto">
        <div className="p-4 sm:p-6 border-b border-slate-200">
          <div className="flex items-center justify-between">
            <h2 className="text-lg sm:text-xl font-bold text-slate-900 pr-8">Publish to GitHub</h2>
            <button
              onClick={onclose}
              className="p-1.5 sm:p-2 text-slate-400 hover:text-slate-600 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <div className="p-4 sm:p-6">
          {success ? (
            <div className="text-center py-2 sm:py-4">
              <div className="w-14 h-14 sm:w-16 sm:h-16 bg-emerald-100 rounded-full mx-auto mb-3 sm:mb-4 flex items-center justify-center">
                <svg className="w-7 h-7 sm:w-8 sm:h-8 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h3 className="text-base sm:text-lg font-semibold text-slate-900 mb-2">Repository Created!</h3>
              <p className="text-sm sm:text-base text-slate-500 mb-4">Your project has been published to GitHub.</p>
              <a
                href={success}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-5 sm:px-6 py-2.5 sm:py-3 bg-slate-900 text-white text-sm sm:text-base rounded-xl font-medium hover:bg-slate-800 transition-colors"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                View on GitHub
              </a>
            </div>
          ) : !githubtoken ? (
            <div className="text-center py-2 sm:py-4">
              <div className="w-14 h-14 sm:w-16 sm:h-16 bg-slate-100 rounded-full mx-auto mb-3 sm:mb-4 flex items-center justify-center">
                <svg className="w-7 h-7 sm:w-8 sm:h-8 text-slate-600" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
              </div>
              <h3 className="text-base sm:text-lg font-semibold text-slate-900 mb-2">Connect GitHub</h3>
              <p className="text-sm sm:text-base text-slate-500 mb-4">Link your GitHub account to publish projects as repositories.</p>
              <button
                onClick={handleconnect}
                className="inline-flex items-center gap-2 px-5 sm:px-6 py-2.5 sm:py-3 bg-slate-900 text-white text-sm sm:text-base rounded-xl font-medium hover:bg-slate-800 transition-colors"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                Connect GitHub
              </button>
            </div>
          ) : (
            <div className="space-y-3 sm:space-y-4">
              <div className="flex items-center gap-2 sm:gap-3 p-2.5 sm:p-3 bg-emerald-50 rounded-lg">
                <svg className="w-4 h-4 sm:w-5 sm:h-5 text-emerald-600 shrink-0" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
                </svg>
                <span className="text-xs sm:text-sm text-emerald-700">GitHub connected</span>
              </div>

              <div>
                <label className="block text-xs sm:text-sm font-medium text-slate-700 mb-1">Repository Name</label>
                <input
                  type="text"
                  value={reponame}
                  onChange={(e) => setreponame(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '-'))}
                  className="w-full px-3 sm:px-4 py-2 sm:py-2.5 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                  placeholder="my-ml-project"
                />
              </div>

              <div>
                <label className="block text-xs sm:text-sm font-medium text-slate-700 mb-1">Description</label>
                <textarea
                  value={description}
                  onChange={(e) => setdescription(e.target.value)}
                  rows={2}
                  className="w-full px-3 sm:px-4 py-2 sm:py-2.5 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none"
                  placeholder="A brief description of your project"
                />
              </div>

              <div className="flex items-center gap-2 sm:gap-3">
                <input
                  type="checkbox"
                  id="private"
                  checked={isprivate}
                  onChange={(e) => setisprivate(e.target.checked)}
                  className="w-4 h-4 text-emerald-600 border-slate-300 rounded focus:ring-emerald-500"
                />
                <label htmlFor="private" className="text-xs sm:text-sm text-slate-600">Make repository private</label>
              </div>

              {error && (
                <div className="p-2.5 sm:p-3 bg-red-50 text-red-600 rounded-lg text-xs sm:text-sm">
                  {error}
                </div>
              )}

              <button
                onClick={handlepublish}
                disabled={publishing || !reponame}
                className="w-full py-2.5 sm:py-3 bg-emerald-600 text-white text-sm sm:text-base rounded-xl font-medium hover:bg-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {publishing ? (
                  <>
                    <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                    </svg>
                    Publishing...
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                    Publish to GitHub
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
