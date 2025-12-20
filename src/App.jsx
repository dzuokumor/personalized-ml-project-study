import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { useEffect } from 'react'
import Layout from './components/layout/layout'
import Home from './pages/home'
import Course from './pages/course'
import Lesson from './pages/lesson'
import Glossary from './pages/glossary'
import SyncModal from './components/ui/syncmodal'
import { usestore } from './store/usestore'

export default function App() {
  const initializesynccode = usestore((state) => state.initializesynccode)
  const showsyncmodal = usestore((state) => state.showsyncmodal)

  useEffect(() => {
    initializesynccode()
  }, [initializesynccode])

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="course/:courseid" element={<Course />} />
          <Route path="course/:courseid/lesson/:lessonid" element={<Lesson />} />
          <Route path="glossary" element={<Glossary />} />
        </Route>
      </Routes>
      {showsyncmodal && <SyncModal />}
    </BrowserRouter>
  )
}
