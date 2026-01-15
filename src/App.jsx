import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/layout/layout'
import Home from './pages/home'
import Course from './pages/course'
import Lesson from './pages/lesson'
import Glossary from './pages/glossary'
import Profile from './pages/profile'
import Achievements from './pages/achievements'
import Leaderboard from './pages/leaderboard'
import Chatwidget from './components/ai/chatwidget'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="course/:courseid" element={<Course />} />
          <Route path="course/:courseid/lesson/:lessonid" element={<Lesson />} />
          <Route path="glossary" element={<Glossary />} />
          <Route path="profile" element={<Profile />} />
          <Route path="achievements" element={<Achievements />} />
          <Route path="leaderboard" element={<Leaderboard />} />
        </Route>
      </Routes>
      <Chatwidget />
    </BrowserRouter>
  )
}
