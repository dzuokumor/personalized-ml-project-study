import { useState, useEffect } from 'react'
import { usestore } from '../../store/usestore'

export default function SyncModal() {
  const [importcode, setimportcode] = useState('')
  const [copied, setcopied] = useState(false)
  const [importstatus, setimportstatus] = useState(null)
  const [activeTab, setactivetab] = useState('export')

  const showsyncmodal = usestore((state) => state.showsyncmodal)
  const setsyncmodalopen = usestore((state) => state.setsyncmodalopen)
  const synccode = usestore((state) => state.synccode)
  const initializesynccode = usestore((state) => state.initializesynccode)
  const exportdata = usestore((state) => state.exportdata)
  const importdata = usestore((state) => state.importdata)

  useEffect(() => {
    initializesynccode()
  }, [initializesynccode])

  if (!showsyncmodal) return null

  const handlecopy = () => {
    const code = exportdata()
    navigator.clipboard.writeText(code)
    setcopied(true)
    setTimeout(() => setcopied(false), 2000)
  }

  const handleimport = () => {
    if (!importcode.trim()) return

    const success = importdata(importcode.trim())
    if (success) {
      setimportstatus('success')
      setTimeout(() => {
        setsyncmodalopen(false)
        setimportstatus(null)
        setimportcode('')
      }, 1500)
    } else {
      setimportstatus('error')
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-black/50"
        onClick={() => setsyncmodalopen(false)}
      />

      <div className="relative bg-white rounded-xl shadow-xl w-full max-w-md">
        <div className="flex items-center justify-between p-4 border-b border-slate-200">
          <h2 className="text-lg font-semibold text-slate-900">Sync Progress</h2>
          <button
            onClick={() => setsyncmodalopen(false)}
            className="p-1 rounded-lg hover:bg-slate-100 text-slate-500"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-4">
          <div className="flex gap-2 mb-4">
            <button
              onClick={() => setactivetab('export')}
              className={`flex-1 py-2 px-4 text-sm font-medium rounded-lg transition-colors ${
                activeTab === 'export'
                  ? 'bg-emerald-100 text-emerald-700'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              Export
            </button>
            <button
              onClick={() => setactivetab('import')}
              className={`flex-1 py-2 px-4 text-sm font-medium rounded-lg transition-colors ${
                activeTab === 'import'
                  ? 'bg-emerald-100 text-emerald-700'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              Import
            </button>
          </div>

          {activeTab === 'export' ? (
            <div>
              <p className="text-sm text-slate-600 mb-4">
                Copy this code to transfer progress to another device.
              </p>

              <div className="mb-4 p-3 bg-slate-50 rounded-lg">
                <p className="text-xs text-slate-500 mb-1">Sync code</p>
                <p className="text-lg font-mono font-semibold text-slate-900 tracking-wider">
                  {synccode}
                </p>
              </div>

              <button
                onClick={handlecopy}
                className="w-full py-2 px-4 bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 transition-colors flex items-center justify-center gap-2"
              >
                {copied ? (
                  <>
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    Copied
                  </>
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    Copy Progress Code
                  </>
                )}
              </button>
            </div>
          ) : (
            <div>
              <p className="text-sm text-slate-600 mb-4">
                Paste a progress code from another device to sync progress here.
              </p>

              <textarea
                value={importcode}
                onChange={(e) => {
                  setimportcode(e.target.value)
                  setimportstatus(null)
                }}
                placeholder="Paste progress code here..."
                className="w-full h-24 p-3 border border-slate-200 rounded-lg text-sm font-mono resize-none focus:outline-none focus:border-emerald-500 mb-4"
              />

              {importstatus === 'success' && (
                <p className="text-sm text-emerald-600 mb-4 flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Progress imported successfully
                </p>
              )}

              {importstatus === 'error' && (
                <p className="text-sm text-red-600 mb-4 flex items-center gap-2">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  Invalid code. Please check and try again.
                </p>
              )}

              <button
                onClick={handleimport}
                disabled={!importcode.trim()}
                className="w-full py-2 px-4 bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Import Progress
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
