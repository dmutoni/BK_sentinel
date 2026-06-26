import { useState, useEffect } from 'react'
import { api } from '../api/client'
import { KPICard, Loading, Heatmap, ProgressBar } from '../components/UI'

const STATES  = ['Low', 'Medium', 'High', 'Default']
const S_COLOR = { Low: '#05b96a', Medium: '#f5a623', High: '#e53e3e', Default: '#7c3aed' }

export default function Overview({ month, seg }) {
  const [snapshot,   setSnapshot]   = useState(null)
  const [trend,      setTrend]      = useState(null)
  const [matrix,     setMatrix]     = useState(null)
  const [absorption, setAbsorption] = useState(null)
  const [loading,    setLoading]    = useState(true)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      api.snapshot(month, seg),
      api.trend(),
      api.matrix(),
      api.absorption().catch(() => null),
    ]).then(([s, t, m, a]) => {
      setSnapshot(s); setTrend(t); setMatrix(m); setAbsorption(a)
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [month, seg])

  if (loading) return <Loading text="Loading portfolio data..." />

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Portfolio Dashboard</div>
        <div className="page-desc">Overview for {month} · {seg}</div>
      </div>

      {/* KPIs */}
      {snapshot && (
        <div className="kpi-grid">
          {STATES.map(s => (
            <KPICard key={s}
              label={`${s} Risk`}
              value={(snapshot.counts[s] || 0).toLocaleString()}
              sub={`${snapshot.pct[s] || 0}% of portfolio`}
              color={S_COLOR[s]}
              dot
            />
          ))}
        </div>
      )}

      <div className="g2">
        {/* transition matrix */}
        {matrix && (
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Monthly Transition Matrix</div>
                <div className="card-sub">Empirical probabilities from 15 monthly pairs</div>
              </div>
            </div>
            <Heatmap states={matrix.states} matrix={matrix.matrix} />
            <div style={{ display: 'flex', gap: 20, marginTop: 14, paddingTop: 14, borderTop: '1px solid var(--border)', flexWrap: 'wrap' }}>
              {[
                { l: 'High → Default',   v: `${(matrix.key_rates.high_to_default * 100).toFixed(1)}%`, c: S_COLOR.High },
                { l: 'Medium recovery',  v: `${(matrix.key_rates.medium_recovery  * 100).toFixed(1)}%`, c: S_COLOR.Medium },
                { l: 'Default persists', v: `${(matrix.key_rates.default_persist  * 100).toFixed(1)}%`, c: S_COLOR.Default },
              ].map(k => (
                <div key={k.l} style={{ fontSize: 12, color: 'var(--text-2)' }}>
                  {k.l}: <strong style={{ color: k.c }}>{k.v}</strong>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* absorption */}
        {absorption && (
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">Layer 3 — Time to Default</div>
                <div className="card-sub">Kemeny & Snell (1960) absorbing Markov chain</div>
              </div>
            </div>
            {absorption.summary.map(r => (
              <ProgressBar
                key={r.state}
                label={`${r.state} risk`}
                value={r.months}
                max={65}
                color={S_COLOR[r.state]}
                sub={`${r.years} years · ${(r.probability * 100).toFixed(0)}% long-run default probability`}
              />
            ))}
          </div>
        )}
      </div>

      {/* trend table */}
      {trend && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">Risk State Trend — Last 8 Months</div>
          </div>
          <div className="tbl-wrap">
            <table>
              <thead>
                <tr>
                  <th>Month</th>
                  {STATES.map(s => <th key={s}>{s}</th>)}
                  <th>Total</th>
                </tr>
              </thead>
              <tbody>
                {trend.months.slice(-8).map(m => {
                  const i = trend.months.indexOf(m)
                  const tot = STATES.reduce((a, s) => a + (trend.series[s]?.[i] || 0), 0)
                  return (
                    <tr key={m}>
                      <td className="td-primary">{m}</td>
                      {STATES.map(s => (
                        <td key={s}>
                          <span style={{ fontWeight: 600, color: S_COLOR[s] }}>
                            {(trend.series[s]?.[i] || 0).toLocaleString()}
                          </span>
                        </td>
                      ))}
                      <td style={{ color: 'var(--text-2)' }}>{tot.toLocaleString()}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
