// ── Badge ──────────────────────────────────
export function Badge({ state }) {
  return <span className={`badge badge-${state}`}>{state}</span>
}

// ── KPI Card ───────────────────────────────
export function KPICard({ label, value, sub, color, dot }) {
  return (
    <div className="kpi" style={{ borderTopColor: color || '#0e5fbf' }}>
      <div className="kpi-label">
        {dot && <span className="kpi-dot" style={{ background: color }} />}
        {label}
      </div>
      <div className="kpi-value" style={{ color: color || '#0f1f3d' }}>{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  )
}

// ── Spinner ────────────────────────────────
export function Spinner() {
  return <div className="spin" />
}

// ── Loading ────────────────────────────────
export function Loading({ text = 'Loading...' }) {
  return (
    <div className="loading-row">
      <Spinner />
      <span>{text}</span>
    </div>
  )
}

// ── Info Box ───────────────────────────────
export function InfoBox({ children }) {
  return <div className="info-box">ℹ {children}</div>
}

// ── Empty State ────────────────────────────
export function Empty({ icon = '🔍', title, desc }) {
  return (
    <div className="empty">
      <div className="empty-icon">{icon}</div>
      <div className="empty-title">{title}</div>
      <div className="empty-desc">{desc}</div>
    </div>
  )
}

// ── Heatmap ────────────────────────────────
export function Heatmap({ states, matrix }) {
  if (!matrix?.length) return null

  function cellStyle(v) {
    const r = Math.round(232 - v * 150)
    const g = Math.round(241 - v * 50)
    const b = 255
    const lum = 0.299 * r + 0.587 * g + 0.114 * b
    return {
      background: `rgb(${r},${g},${b})`,
      color: lum > 170 ? '#0f1f3d' : '#fff',
    }
  }

  return (
    <div className="hm-wrap">
      <table className="hm-table">
        <thead>
          <tr>
            <th>From \ To</th>
            {states.map(s => <th key={s}>{s}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, ri) => (
            <tr key={ri}>
              <td className="hm-row-label">{states[ri]}</td>
              {row.map((v, ci) => (
                <td key={ci} style={cellStyle(v)}>{v.toFixed(3)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Progress Bar ───────────────────────────
export function ProgressBar({ label, value, max, color, sub }) {
  const pct = Math.min((value / max) * 100, 100)
  return (
    <div className="prog-row">
      <div className="prog-header">
        <span className="prog-label">{label}</span>
        <span className="prog-val" style={{ color }}>{value} months</span>
      </div>
      <div className="prog-track">
        <div className="prog-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      {sub && <div className="prog-sub">{sub}</div>}
    </div>
  )
}

// ── Pagination ─────────────────────────────
export function Pagination({ page, total, totalPages, onPrev, onNext }) {
  return (
    <div className="pagination">
      <button className="btn btn-ghost btn-sm" disabled={page <= 1} onClick={onPrev}>
        ◀ Previous
      </button>
      <button className="btn btn-ghost btn-sm" disabled={page >= totalPages} onClick={onNext}>
        Next ▶
      </button>
      <div className="page-info">
        Page {page} of {totalPages} · <strong>{total.toLocaleString()}</strong> accounts
      </div>
    </div>
  )
}

// ── Trans Arrow ────────────────────────────
export function TransArrow({ from, to }) {
  return (
    <div className="trans-arrow">
      <Badge state={from} />
      <span>→</span>
      <Badge state={to} />
    </div>
  )
}
