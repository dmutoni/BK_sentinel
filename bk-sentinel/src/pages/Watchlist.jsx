import { useState, useEffect, useCallback } from 'react'
import { api } from '../api/client'
import { KPICard, Loading, Empty, Pagination } from '../components/UI'
import { Badge } from '../components/UI'

const STATES  = ['Low', 'Medium', 'High', 'Default']
const S_COLOR = { Low: '#05b96a', Medium: '#f5a623', High: '#e53e3e', Default: '#7c3aed' }

const FILTERS = [
  { id: 'about_to_default',   label: '🔴 About to Default — High with P(Default) > 25%' },
  { id: 'high_risk_any',      label: '🔴 All High-Risk Accounts' },
  { id: 'high_to_high',       label: '🔴 Stuck in High — elevated default risk' },
  { id: 'medium_to_high',     label: '🟠 Medium → High (fast deterioration)' },
  { id: 'medium_elevated',    label: '🟠 Medium at Risk — P(Default) > 10%' },
  { id: 'medium_to_default',  label: '🟠 Medium → Default (direct jump)' },
  { id: 'low_to_medium',      label: '🟡 Low → Medium (earliest warning)' },
  { id: 'low_to_high',        label: '🟡 Low → High or Default (large jump)' },
  { id: 'fast_deterioration', label: '⚠️ All Fast Deterioration' },
  { id: 'all_deteriorating',  label: '📉 All Deteriorating Accounts' },
  { id: 'all_performing',     label: '📋 Full Performing Portfolio' },
]

export default function Watchlist({ month, seg }) {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [filter,  setFilter]  = useState('all_deteriorating')
  const [horizon, setHorizon] = useState(1)
  const [page,    setPage]    = useState(1)

  const fetchWatchlist = useCallback((pg) => {
    if (!month) return
    setLoading(true)
    api.watchlist(month, seg, filter, horizon, pg)
      .then(d => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [month, seg, filter, horizon])

  // reload when month, seg, filter or horizon change
  useEffect(() => {
    if (!month) return
    setPage(1)
    fetchWatchlist(1)
  }, [month, seg, filter, horizon])

  // reload when page changes
  useEffect(() => {
    if (!month || page === 1) return
    fetchWatchlist(page)
  }, [page])

  const sum = data?.summary || {}

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Risk Watchlist</div>
        <div className="page-desc">
          Accounts predicted to deteriorate — already-defaulted accounts are excluded
        </div>
      </div>

      {/* KPIs */}
      <div className="kpi-grid">
        {[
          { l: 'Predicted Default', k: 'Default' },
          { l: 'Predicted High',    k: 'High' },
          { l: 'Predicted Medium',  k: 'Medium' },
          { l: 'Predicted Healthy', k: 'Low' },
        ].map(m => (
          <KPICard key={m.k}
            label={m.l}
            value={(sum[m.k] || 0).toLocaleString()}
            sub="next month"
            color={S_COLOR[m.k]}
            dot
          />
        ))}
      </div>

      <div className="card">
        {/* filter bar */}
        <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div style={{ flex: '2 1 240px' }}>
            <div className="field-label">Show accounts</div>
            <select className="select" value={filter}
              onChange={e => { setFilter(e.target.value); setPage(1) }}>
              {FILTERS.map(f => <option key={f.id} value={f.id}>{f.label}</option>)}
            </select>
          </div>
          <div style={{ flex: '1 1 160px' }}>
            <div className="field-label">Risk horizon</div>
            <select className="select" value={horizon}
              onChange={e => { setHorizon(+e.target.value); setPage(1) }}>
              {[1, 2, 3, 6, 12].map(h => (
                <option key={h} value={h}>{h} month{h > 1 ? 's' : ''} from now</option>
              ))}
            </select>
          </div>
        </div>

        {!month && (
          <Empty icon="📅" title="No month selected"
            desc="Select a month from the sidebar to load the watchlist." />
        )}

        {month && loading && <Loading text="Running AI predictions on all accounts..." />}

        {month && !loading && data?.total === 0 && (
          <Empty icon="✅" title="No accounts match this filter"
            desc={`No accounts matched for ${month}. Try "All Deteriorating Accounts" or change the month.`} />
        )}

        {month && !loading && data?.total > 0 && (
          <>
            <div style={{ fontSize: 13, color: 'var(--text-2)', marginBottom: 14 }}>
              <strong style={{ color: '#0e5fbf' }}>{data.total.toLocaleString()}</strong> accounts
              matched — ranked by combined risk score
            </div>

            <div className="tbl-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Loan ID</th>
                    <th>Customer ID</th>
                    <th>Segment</th>
                    <th>Transition</th>
                    <th>Risk in {horizon}m</th>
                    <th>Days in Arrears</th>
                    <th>Instalments Missed</th>
                    <th>P(Default)</th>
                  </tr>
                </thead>
                <tbody>
                  {data.records.map((r, i) => (
                    <tr key={i}>
                      <td className="td-primary">{r.loan_id}</td>
                      <td className="td-mono">{(r.customer_id || '').substring(0, 18)}</td>
                      <td>{r.segment}</td>
                      <td>
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                          <Badge state={r.risk_state} />
                          <span style={{ color: 'var(--text-3)', fontSize: 14 }}>→</span>
                          <Badge state={r.pred} />
                        </div>
                      </td>
                      <td>
                        <strong style={{ color: S_COLOR[r.pred] || 'var(--text-2)' }}>
                          {r.horizon_risk}
                        </strong>
                      </td>
                      <td style={{ fontWeight: 600, color: r.days_in_arrears > 90 ? S_COLOR.Default : r.days_in_arrears > 30 ? S_COLOR.High : r.days_in_arrears > 0 ? S_COLOR.Medium : 'var(--text-2)' }}>
                        {Math.round(r.days_in_arrears || 0)}
                      </td>
                      <td>{Math.round(r.instalments_in_arrears || 0)}</td>
                      <td>
                        <strong style={{ color: S_COLOR.Default }}>
                          {((r.p_default || 0) * 100).toFixed(1)}%
                        </strong>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <Pagination
              page={data.page}
              total={data.total}
              totalPages={data.total_pages}
              onPrev={() => setPage(p => p - 1)}
              onNext={() => setPage(p => p + 1)}
            />
          </>
        )}
      </div>
    </div>
  )
}
