import { useState, useRef } from 'react'
import { useauth } from '../../contexts/authcontext'
import { uploadimage, getoptimizedurl } from '../../services/cloudinary'

export default function avatarupload({ size = 'large' }) {
  const { user, avatar, updateavatar } = useauth()
  const [uploading, setuploading] = useState(false)
  const fileinputref = useRef(null)

  const handleclick = () => {
    fileinputref.current?.click()
  }

  const handlefilechange = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.type.startsWith('image/')) {
      alert('Please select an image file')
      return
    }

    if (file.size > 5 * 1024 * 1024) {
      alert('Image must be less than 5MB')
      return
    }

    setuploading(true)
    try {
      const { url } = await uploadimage(file)
      await updateavatar(url)
    } catch (error) {
      console.error('Upload failed:', error)
      alert('Failed to upload image. Please try again.')
    } finally {
      setuploading(false)
    }
  }

  const avatarurl = avatar ? getoptimizedurl(avatar, { width: 200, height: 200 }) : null
  const islarge = size === 'large'

  return (
    <div className="relative group">
      <input
        type="file"
        ref={fileinputref}
        onChange={handlefilechange}
        accept="image/*"
        className="hidden"
      />
      <button
        onClick={handleclick}
        disabled={uploading}
        className={`${islarge ? 'w-24 h-24' : 'w-20 h-20'} rounded-2xl flex items-center justify-center text-4xl font-bold backdrop-blur-sm border overflow-hidden transition-all hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed ${
          islarge
            ? 'border-white/30 bg-white/20 hover:border-white/50'
            : 'border-slate-200 bg-slate-100'
        }`}
      >
        {uploading ? (
          <svg className={`${islarge ? 'w-8 h-8' : 'w-6 h-6'} animate-spin ${islarge ? '' : 'text-slate-400'}`} fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
          </svg>
        ) : avatarurl ? (
          <img src={avatarurl} alt="Profile" className="w-full h-full object-cover" />
        ) : (
          <span className={islarge ? '' : 'text-slate-400 text-2xl'}>{user?.email?.charAt(0).toUpperCase()}</span>
        )}
      </button>
      <div className={`absolute inset-0 bg-black/40 ${islarge ? 'rounded-2xl' : 'rounded-xl'} opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none`}>
        <svg className={`${islarge ? 'w-6 h-6' : 'w-5 h-5'} ${islarge ? '' : 'text-white'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"/>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"/>
        </svg>
      </div>
    </div>
  )
}
