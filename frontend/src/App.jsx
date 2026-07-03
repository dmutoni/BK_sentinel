import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { api } from './api/client'
import Sidebar from './components/Sidebar'
import Login from './pages/Login'
import Overview from './pages/Overview'
import AccountLookup from './pages/AccountLookup'
import Watchlist from './pages/Watchlist'
import Portfolio from './pages/Portfolio'

const PAGE_META = {
  '/':          { title: 'Portfolio Dashboard',    desc: 'Overview of the full loan portfolio' },
  '/lookup':    { title: 'Account Lookup',         desc: 'Search any loan account' },
  '/watchlist': { title: 'Risk Watchlist',         desc: 'Accounts predicted to deteriorate' },
  '/portfolio': { title: 'Portfolio Analysis',     desc: 'Markov chain forecasts and absorption analysis' },
}

function AuthedApp({ user, onLogout }) {
  const [months,    setMonths]    = useState([])
  const [segments,  setSegments]  = useState([])
  const [month,     setMonth]     = useState('')
  const [seg,       setSeg]       = useState('All Segments')
  const [snapshot,  setSnapshot]  = useState(null)
  const [pathname,  setPathname]  = useState('/')

  useEffect(() => {
    api.months().then(m => { setMonths(m); if (m.length) setMonth(m[m.length - 1]) })
    api.segments().then(setSegments)
  }, [])

  useEffect(() => {
    if (!month) return
    api.snapshot(month, seg).then(setSnapshot).catch(() => {})
  }, [month, seg])

  const meta = PAGE_META[pathname] || {}
  const props = { month, seg }

  return (
    <BrowserRouter>
      <div className="app">
        <Sidebar
          user={user} onLogout={onLogout}
          months={months} segments={segments}
          month={month} setMonth={setMonth}
          seg={seg} setSeg={setSeg}
          snapshot={snapshot}
        />
        <div className="content-area">
          {/* top bar */}
          <div className="topbar">
            <div>
              <div className="topbar-title">{meta.title || 'BK Sentinel'}</div>
              <div className="topbar-sub">{meta.desc || ''}</div>
            </div>
            <div className="topbar-month">📅 {month}</div>
          </div>

          <div className="page-body">
            <Routes>
              <Route path="/"          element={<Overview  {...props} />} />
              <Route path="/lookup"    element={<AccountLookup />} />
              <Route path="/watchlist" element={<Watchlist {...props} />} />
              <Route path="/portfolio" element={<Portfolio  {...props} />} />
              <Route path="*"          element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </div>
      </div>
    </BrowserRouter>
  )
}

export default function App() {
  const [user,    setUser]    = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const t = localStorage.getItem('bk_token')
    if (t) {
      api.me().then(u => { setUser(u); setLoading(false) }).catch(() => { localStorage.removeItem('bk_token'); setLoading(false) })
    } else {
      setLoading(false)
    }
  }, [])

  function handleLogin(data) { setUser(data) }
  function handleLogout() {
    api.logout().catch(() => {})
    localStorage.removeItem('bk_token')
    setUser(null)
  }

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', fontFamily: 'Poppins, sans-serif', color: '#0e5fbf' }}>
      Loading...
    </div>
  )

  if (!user) return <Login onLogin={handleLogin} />
  return <AuthedApp user={user} onLogout={handleLogout} />
}
