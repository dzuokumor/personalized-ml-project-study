import { useRef } from 'react'
import { useauth } from '../../contexts/authcontext'
import Logo from '../ui/logo'

export default function certificate({ completiondate, totalcourses, totalhours, onclose }) {
  const certificateref = useRef(null)
  const { user } = useauth()

  const username = user?.user_metadata?.full_name || user?.email?.split('@')[0] || 'Student'
  const formatteddate = new Date(completiondate || Date.now()).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })

  const certificateid = `NRNCERT-${Date.now().toString(36).toUpperCase()}-${Math.random().toString(36).substring(2, 6).toUpperCase()}`

  const downloadcertificate = async () => {
    const element = certificateref.current
    if (!element) return

    try {
      const html2canvas = (await import('html2canvas')).default
      const canvas = await html2canvas(element, {
        scale: 2,
        backgroundColor: '#ffffff',
        useCORS: true
      })

      const link = document.createElement('a')
      link.download = `neuron-ml-certificate-${username.toLowerCase().replace(/\s+/g, '-')}.png`
      link.href = canvas.toDataURL('image/png')
      link.click()
    } catch (err) {
      console.error('Failed to generate certificate:', err)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4 overflow-y-auto">
      <div className="relative max-w-4xl w-full">
        <button
          onClick={onclose}
          className="absolute -top-12 right-0 text-white hover:text-slate-300 transition-colors"
        >
          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <div
          ref={certificateref}
          className="bg-white overflow-hidden"
          style={{ aspectRatio: '1.414', minHeight: '600px' }}
        >
          <div className="relative h-full p-10 flex flex-col">
            <div className="absolute inset-0 overflow-hidden">
              <div className="absolute -top-20 -left-20 w-80 h-80 bg-gradient-to-br from-emerald-100 to-transparent rounded-full opacity-60" />
              <div className="absolute -bottom-20 -right-20 w-80 h-80 bg-gradient-to-tl from-emerald-100 to-transparent rounded-full opacity-60" />
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] opacity-[0.03]">
                <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
                  <g stroke="#10b981" strokeWidth="1">
                    <line x1="12" y1="16" x2="32" y2="10" />
                    <line x1="12" y1="16" x2="32" y2="32" />
                    <line x1="12" y1="16" x2="32" y2="54" />
                    <line x1="12" y1="32" x2="32" y2="10" />
                    <line x1="12" y1="32" x2="32" y2="32" />
                    <line x1="12" y1="32" x2="32" y2="54" />
                    <line x1="12" y1="48" x2="32" y2="10" />
                    <line x1="12" y1="48" x2="32" y2="32" />
                    <line x1="12" y1="48" x2="32" y2="54" />
                    <line x1="32" y1="10" x2="52" y2="24" />
                    <line x1="32" y1="32" x2="52" y2="24" />
                    <line x1="32" y1="54" x2="52" y2="24" />
                    <line x1="32" y1="10" x2="52" y2="40" />
                    <line x1="32" y1="32" x2="52" y2="40" />
                    <line x1="32" y1="54" x2="52" y2="40" />
                  </g>
                  <circle cx="12" cy="16" r="4" fill="#10b981" />
                  <circle cx="12" cy="32" r="4" fill="#10b981" />
                  <circle cx="12" cy="48" r="4" fill="#10b981" />
                  <circle cx="32" cy="10" r="5" fill="#10b981" />
                  <circle cx="32" cy="32" r="5" fill="#10b981" />
                  <circle cx="32" cy="54" r="5" fill="#10b981" />
                  <circle cx="52" cy="24" r="4" fill="#10b981" />
                  <circle cx="52" cy="40" r="4" fill="#10b981" />
                </svg>
              </div>
            </div>

            <div className="absolute inset-6 border-2 border-emerald-200 rounded-lg pointer-events-none" />
            <div className="absolute inset-8 border border-emerald-100 rounded-lg pointer-events-none" />

            <div className="relative flex-1 flex flex-col items-center justify-center text-center">
              <div className="flex items-center gap-4 mb-8">
                <Logo size={56} />
                <div className="text-left">
                  <h2 className="text-3xl font-bold text-slate-900">Neuron</h2>
                  <p className="text-sm text-slate-500 tracking-wide">Machine Learning Academy</p>
                </div>
              </div>

              <div className="mb-4">
                <p className="text-sm uppercase tracking-[0.4em] text-emerald-600 font-semibold">Certificate of Achievement</p>
              </div>

              <div className="w-32 h-1 bg-gradient-to-r from-transparent via-emerald-500 to-transparent mb-8" />

              <p className="text-slate-500 mb-3 text-lg">This certifies that</p>

              <h1 className="text-5xl font-bold text-slate-900 mb-4" style={{ fontFamily: 'Georgia, serif' }}>{username}</h1>

              <p className="text-slate-500 mb-6 text-lg">has successfully completed the</p>

              <div className="bg-gradient-to-r from-emerald-500 to-teal-500 px-10 py-5 rounded-2xl mb-8 shadow-lg shadow-emerald-200">
                <h2 className="text-2xl font-bold text-white tracking-wide">Complete Machine Learning Curriculum</h2>
                <p className="text-emerald-100 text-sm mt-1">From Foundations to Production</p>
              </div>

              <p className="text-slate-600 max-w-lg mb-8">
                Demonstrating comprehensive proficiency in machine learning concepts, deep learning architectures,
                natural language processing, and production ML systems.
              </p>

              <div className="grid grid-cols-3 gap-12 mb-10">
                <div className="text-center">
                  <div className="w-14 h-14 bg-blue-50 rounded-2xl flex items-center justify-center mx-auto mb-3 border border-blue-100">
                    <svg className="w-7 h-7 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                    </svg>
                  </div>
                  <p className="text-2xl font-bold text-slate-900">{totalcourses}</p>
                  <p className="text-xs text-slate-500">Courses Completed</p>
                </div>
                <div className="text-center">
                  <div className="w-14 h-14 bg-amber-50 rounded-2xl flex items-center justify-center mx-auto mb-3 border border-amber-100">
                    <svg className="w-7 h-7 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <p className="text-2xl font-bold text-slate-900">{totalhours}+</p>
                  <p className="text-xs text-slate-500">Hours of Learning</p>
                </div>
                <div className="text-center">
                  <div className="w-14 h-14 bg-emerald-50 rounded-2xl flex items-center justify-center mx-auto mb-3 border border-emerald-100">
                    <svg className="w-7 h-7 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <p className="text-2xl font-bold text-slate-900">{formatteddate}</p>
                  <p className="text-xs text-slate-500">Date of Completion</p>
                </div>
              </div>

              <div className="flex items-end justify-center gap-16">
                <div className="text-center">
                  <div className="w-40 border-b-2 border-slate-300 mb-2" />
                  <p className="text-xs text-slate-500">Platform Director</p>
                </div>
                <div className="flex flex-col items-center -mb-2">
                  <div className="w-24 h-24 bg-gradient-to-br from-amber-100 to-amber-200 rounded-full flex items-center justify-center border-4 border-amber-300 shadow-xl">
                    <Logo size={48} />
                  </div>
                </div>
                <div className="text-center">
                  <div className="w-40 border-b-2 border-slate-300 mb-2" />
                  <p className="text-xs text-slate-500">Lead Instructor</p>
                </div>
              </div>
            </div>

            <div className="relative flex items-center justify-between text-xs text-slate-400 pt-6 border-t border-slate-100">
              <p>Certificate ID: {certificateid}</p>
              <div className="flex items-center gap-1">
                <Logo size={16} />
                <span className="font-medium text-slate-500">Neuron</span>
              </div>
              <p>Issued: {formatteddate}</p>
            </div>
          </div>
        </div>

        <div className="flex justify-center gap-4 mt-6">
          <button
            onClick={downloadcertificate}
            className="px-8 py-3 bg-emerald-600 text-white rounded-xl font-semibold hover:bg-emerald-700 transition-colors flex items-center gap-2 shadow-lg"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download Certificate
          </button>
          <button
            onClick={onclose}
            className="px-8 py-3 bg-white text-slate-700 rounded-xl font-semibold hover:bg-slate-100 transition-colors border border-slate-200"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
