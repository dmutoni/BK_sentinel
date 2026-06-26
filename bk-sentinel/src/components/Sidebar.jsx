import { useNavigate, useLocation } from 'react-router-dom'

const STATES = ['Low', 'Medium', 'High', 'Default']
const S_COLOR = { Low: '#05b96a', Medium: '#f5a623', High: '#e53e3e', Default: '#7c3aed' }

const NAV = [
  { path: '/',          icon: '◉', label: 'Overview' },
  { path: '/lookup',    icon: '⊙', label: 'Account Lookup' },
  { path: '/watchlist', icon: '⚑', label: 'Risk Watchlist' },
  { path: '/portfolio', icon: '◈', label: 'Portfolio Analysis' },
]

export default function Sidebar({ user, onLogout, months, segments, month, setMonth, seg, setSeg, snapshot }) {
  const navigate  = useNavigate()
  const { pathname } = useLocation()

  return (
    <div className="sidebar">
      {/* logo */}
      <div className="sb-logo">
        <div className="sb-logo-mark">
          <div className="sb-logo-icon">🏦</div>
          <div>
            <div className="sb-logo-text">BK Sentinel</div>
            <div className="sb-logo-sub">Credit Risk System</div>
          </div>
        </div>
      </div>

      {/* nav */}
      <div className="sb-section-label">Navigation</div>
      <nav className="sb-nav">
        {NAV.map(n => (
          <button
            key={n.path}
            className={`sb-link${pathname === n.path ? ' active' : ''}`}
            onClick={() => navigate(n.path)}
          >
            <span className="sb-link-icon">{n.icon}</span>
            {n.label}
          </button>
        ))}
      </nav>

      <div className="sb-divider" />

      {/* filters */}
      <div className="sb-filters">
        <div className="sb-filter-label">Filters</div>
        <select className="sb-select" value={month} onChange={e => setMonth(e.target.value)}>
          {months.map(m => <option key={m}>{m}</option>)}
        </select>
        <select className="sb-select" value={seg} onChange={e => setSeg(e.target.value)}>
          {segments.map(s => <option key={s}>{s}</option>)}
        </select>
      </div>

      {/* snapshot */}
      {snapshot && (
        <div className="sb-snap">
          <div className="sb-filter-label">Portfolio — {month}</div>
          {STATES.map(s => (
            <div key={s} className="sb-snap-row">
              <span className="sb-snap-state" style={{ color: S_COLOR[s] }}>{s}</span>
              <span className="sb-snap-val">
                {(snapshot.counts[s] || 0).toLocaleString()}
                <span className="sb-snap-pct">{snapshot.pct[s] || 0}%</span>
              </span>
            </div>
          ))}
        </div>
      )}

      {/* user */}
      <div className="sb-user">
        <div className="sb-avatar">👤</div>
        <div>
          <div className="sb-user-name">{user?.name}</div>
          <div className="sb-user-role">{user?.role}</div>
        </div>
        <button className="sb-logout" onClick={onLogout} title="Sign out">⏻</button>
      </div>
    </div>
  )
}
