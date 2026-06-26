import { useState, useEffect } from 'react'
import { api } from '../api/client'
import { KPICard, Loading, Heatmap, InfoBox } from '../components/UI'

const STATES  = ['Low', 'Medium', 'High', 'Default']
const S_COLOR = { Low: '#05b96a', Medium: '#f5a623', High: '#e53e3e', Default: '#7c3aed' }

export default function Portfolio({ month, seg }) {
  const [matrix,     setMatrix]     = useState(null)
  const [forecast,   setForecast]   = useState(null)
  const [portfolio,  setPortfolio]  = useState(null)
  const [absorption, setAbsorption] = useState(null)
  const [horizon,    setHorizon]    = useState(6)
  const [loading,    setLoading]    = useState(true)

  useEffect(() => {
    Promise.all([
      api.matrix(),
      api.absorption().catch(() => null),
    ]).then(([m, a]) => { setMatrix(m); setAbsorption(a); setLoading(false) })
  }, [])

  useEffect(() => {
    api.forecast(horizon).then(setForecast)
    api.portfolioForecast(month, seg).then(setPortfolio)
  }, [horizon, month, seg])

  if (loading) return <Loading text="Loading analysis data..." />

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Portfolio Analysis</div>
        <div className="page-desc">Markov chain transition forecasts and long-run absorption analysis</div>
      </div>

      {/* key rates */}
      {matrix && (
        <div className="kpi-grid">
          {[
            { label: 'Low stability',    value: `${(matrix.key_rates.low_retention   * 100).toFixed(1)}%`, sub: 'Monthly self-retention',   color: S_COLOR.Low,    dot: true },
            { label: 'Medium → Low',     value: `${(matrix.key_rates.medium_recovery * 100).toFixed(1)}%`, sub: 'Monthly recovery rate',    color: S_COLOR.Medium, dot: true },
            { label: 'High → Default',   value: `${(matrix.key_rates.high_to_default * 100).toFixed(1)}%`, sub: '⚠ Monthly deterioration',  color: S_COLOR.High,   dot: true },
            { label: 'Default persists', value: `${(matrix.key_rates.default_persist * 100).toFixed(1)}%`, sub: 'Near-absorbing state',     color: S_COLOR.Default, dot: true },
          ].map(m => <KPICard key={m.label} {...m} />)}
        </div>
      )}

      <div className="g2">
        {/* matrix */}
        {matrix && (
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">1-Month Transition Matrix</div>
                <div className="card-sub">Empirical monthly transition probabilities</div>
              </div>
            </div>
            <Heatmap states={matrix.states} matrix={matrix.matrix} />
          </div>
        )}

        {/* forecast */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Multi-Step Forecast P<sup>n</sup></div>
              <div className="card-sub">Transition probabilities at selected horizon</div>
            </div>
          </div>
          <div style={{ marginBottom: 16 }}>
            <div className="field-label">Months ahead</div>
            <select className="select" value={horizon} onChange={e => setHorizon(+e.target.value)}>
              {[1, 2, 3, 6, 9, 12].map(h => (
                <option key={h} value={h}>{h} month{h > 1 ? 's' : ''}</option>
              ))}
            </select>
          </div>
          {forecast ? (
            <>
              <Heatmap states={forecast.states} matrix={forecast.matrix} />
              {forecast.high_to_default != null && (
                <InfoBox>
                  High-risk account today: <strong>{(forecast.high_to_default * 100).toFixed(1)}%</strong> probability of reaching Default within {horizon} months.
                </InfoBox>
              )}
            </>
          ) : <Loading text="Computing forecast..." />}
        </div>
      </div>

      {/* portfolio evolution */}
      {portfolio && (
        <div className="card">
          <div className="card-header">
            <div className="card-title">Portfolio Risk Distribution Forecast — Next 12 Months</div>
          </div>
          <div className="tbl-wrap">
            <table>
              <thead>
                <tr>
                  <th>Horizon</th>
                  {STATES.map(s => <th key={s} style={{ color: S_COLOR[s] }}>{s}</th>)}
                </tr>
              </thead>
              <tbody>
                {[0, 3, 6, 12].map(n => {
                  const row = portfolio[n]
                  return row ? (
                    <tr key={n}>
                      <td className="td-primary">{n === 0 ? `Now (${month})` : `In ${n} months`}</td>
                      {STATES.map(s => (
                        <td key={s}><strong style={{ color: S_COLOR[s] }}>{row[s]}%</strong></td>
                      ))}
                    </tr>
                  ) : null
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* absorption */}
      {absorption && (
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Layer 3 — Fundamental Matrix & Absorption Probabilities</div>
              <div className="card-sub">Kemeny & Snell (1960) absorbing Markov chain analysis</div>
            </div>
          </div>
          <div className="g3" style={{ marginBottom: 20 }}>
            {absorption.summary.map(r => (
              <KPICard key={r.state}
                label={`${r.state} → Default`}
                value={`${r.months} months`}
                sub={`${r.years} years · ${(r.probability * 100).toFixed(0)}% eventual probability`}
                color={S_COLOR[r.state]}
                dot
              />
            ))}
          </div>

          {absorption.fundamental_matrix?.values?.length > 0 && (
            <>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#0e5fbf', marginBottom: 6 }}>
                Fundamental Matrix N = (I − Q)⁻¹
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 12 }}>
                Expected months an account spends in each state before eventually reaching Default
              </div>
              <Heatmap states={absorption.fundamental_matrix.states} matrix={absorption.fundamental_matrix.values} />
            </>
          )}
        </div>
      )}
    </div>
  )
}
