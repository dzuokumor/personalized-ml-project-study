import { useEffect, useRef } from 'react'
import { useauth } from '../../contexts/authcontext'
import { usestore } from '../../store/usestore'

export default function syncprovider({ children }) {
  const { user } = useauth()
  const loadfromcloud = usestore((state) => state.loadfromcloud)
  const syncedref = useRef(false)

  useEffect(() => {
    if (user && !syncedref.current) {
      syncedref.current = true
      loadfromcloud(user.id)
    }
    if (!user) {
      syncedref.current = false
    }
  }, [user, loadfromcloud])

  return children
}
