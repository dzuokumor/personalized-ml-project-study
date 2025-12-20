import { Outlet } from 'react-router-dom'
import { useState } from 'react'
import Sidebar from './sidebar'
import Header from './header'

export default function Layout() {
  const [sidebaropen, setsidebaropen] = useState(false)

  return (
    <div className="min-h-screen bg-slate-50 flex">
      <div
        className={`fixed inset-0 bg-black/50 z-40 lg:hidden transition-opacity ${
          sidebaropen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        onClick={() => setsidebaropen(false)}
      />

      <Sidebar isopen={sidebaropen} onclose={() => setsidebaropen(false)} />

      <div className="flex-1 flex flex-col lg:ml-64">
        <Header onmenuclick={() => setsidebaropen(true)} />
        <main className="flex-1 p-4 md:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
