import { useState } from 'react'
import { api } from '../api/client'
import { Badge, Loading, InfoBox, Empty } from '../components/UI'

const STATES  = ['Low', 'Medium', 'High', 'Default']
const S_COLOR = { Low: '#05b96a', Medium: '#f5a623', High: '#e53e3e', Default: '#7c3aed' }

function PaymentBehaviour({ acc }) {
  const metrics = [
    { label: 'Days in Arrears', value: Math.round(acc.days_in_arrears), sub: 'BNR rule: >90 days = Default', color: acc.days_in_arrears > 90 ? S_COLOR.Default : acc.days_in_arrears > 30 ? S_COLOR.High : acc.days_in_arrears > 0 ? S_COLOR.Medium : S_COLOR.Low },
    { label: 'Instalments in Arrears', value: Math.round(acc.instalments_in_arrears), sub: 'Consecutive missed payments', color: acc.instalments_in_arrears > 2 ? S_COLOR.High : acc.instalments_in_arrears > 0 ? S_COLOR.Medium : S_COLOR.Low },
    { label: 'Repayment Ratio', value: `${(acc.repayment_ratio * 100).toFixed(0)}%`, sub: 'Amount paid vs scheduled', color: acc.repayment_ratio > 0.8 ? S_COLOR.Low : acc.repayment_ratio > 0.5 ? S_COLOR.Medium : S_COLOR.High },
    { label: 'Penal Interest', value: `RWF ${Math.round(acc.penal_interest).toLocaleString()}`, sub: 'Accumulates on missed payments', color: acc.penal_interest > 0 ? S_COLOR.High : S_COLOR.Low },
    { label: 'Suspended Interest', value: `RWF ${Math.round(acc.suspended_interest).toLocaleString()}`, sub: 'Only triggered at DPD > 90', color: acc.suspended_interest > 0 ? S_COLOR.Default : S_COLOR.Low },
    { label: 'Principal Balance', value: `RWF ${Math.round(acc.principal_balance).toLocaleString()}`, sub: 'Remaining loan balance', color: '#0e5fbf' },
  ]
  return (
    <div className="g3">
      {metrics.map(m => (
        <div key={m.label} className="kpi" style={{ borderTopColor: m.color }}>
          <div className="kpi-label">{m.label}</div>
          <div className="kpi-value" style={{ color: m.color, fontSize: typeof m.value === 'string' && m.value.length > 8 ? '18px' : '26px' }}>{m.value}</div>
          <div className="kpi-sub">{m.sub}</div>
        </div>
      ))}
    </div>
  )
}

function AIPrediction({ pred }) {
  const risky = ['High', 'Default'].includes(pred.predicted_state)
  return (
    <div>
      <InfoBox>
        AI model predicts this account will be in <strong>{pred.predicted_state}</strong> state next month.
      </InfoBox>
      <div className="pred-grid">
        {STATES.map(s => {
          const prob = pred.probabilities[s] || 0
          const isPred = s === pred.predicted_state
          return (
            <div key={s} className={`pred-card${isPred ? ' is-pred' : ''}`} style={{ borderColor: isPred ? S_COLOR[s] : 'var(--border)' }}>
              {isPred && <div className="pred-tag" style={{ background: S_COLOR[s] }}>PREDICTED</div>}
              <div className="pred-card-label" style={{ color: S_COLOR[s] }}>{s}</div>
              <div className="pred-card-pct" style={{ color: S_COLOR[s] }}>{(prob * 100).toFixed(0)}%</div>
            </div>
          )
        })}
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Why {pred.predicted_state}?</div>
            <div className="card-sub">Top features driving this prediction — computed using SHAP values</div>
          </div>
        </div>
        {pred.shap_reasons.length > 0 ? (
          <div>
            <p style={{ fontSize: 12, color: 'var(--text-2)', marginBottom: 12 }}>
              Features in red push this account toward higher risk. Features in green are stabilising it.
            </p>
            {pred.shap_reasons.map((r, i) => {
              const pushes = r.shap > 0
              let cls, icon, msg
              if (risky && pushes)    { cls = 'pill-red';   icon = '↑'; msg = 'increasing risk' }
              else if (risky)         { cls = 'pill-green'; icon = '↓'; msg = 'reducing risk' }
              else if (!risky && pushes) { cls = 'pill-green'; icon = '↑'; msg = 'supports healthy state' }
              else                    { cls = 'pill-amber'; icon = '↓'; msg = 'minor factor' }
              const val = typeof r.value === 'number' ? r.value.toLocaleString() : r.value
              return (
                <span key={i} className={`pill ${cls}`}>
                  {icon} <strong>{r.feature}</strong> = {val} — {msg}
                </span>
              )
            })}
          </div>
        ) : (
          <InfoBox>No explanation available for this account.</InfoBox>
        )}
      </div>
    </div>
  )
}

function PaymentHistory({ history }) {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">Payment History — Last 16 Months</div>
      </div>
      <div className="tbl-wrap">
        <table>
          <thead>
            <tr>
              <th>Month</th><th>State</th><th>Days in Arrears</th>
              <th>Instalments Missed</th><th>Repayment Ratio</th><th>Penal Interest</th>
            </tr>
          </thead>
          <tbody>
            {[...history].reverse().map((h, i) => (
              <tr key={i}>
                <td className="td-primary">{h.observation_month}</td>
                <td><Badge state={h.risk_state} /></td>
                <td style={{ fontWeight: 600, color: h.days_in_arrears > 90 ? S_COLOR.Default : h.days_in_arrears > 30 ? S_COLOR.High : h.days_in_arrears > 0 ? S_COLOR.Medium : 'var(--text-2)' }}>
                  {Math.round(h.days_in_arrears || 0)}
                </td>
                <td>{Math.round(h.instalments_in_arrears || 0)}</td>
                <td>{((h.repayment_ratio || 0) * 100).toFixed(0)}%</td>
                <td>RWF {(h.penal_interest || 0).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function AccountLookup() {
  const [query,   setQuery]   = useState('')
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')
  const [tab,     setTab]     = useState('behaviour')

  async function handleSearch(e) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true); setError(''); setData(null)
    try {
      const d = await api.account(query.trim())
      setData(d); setTab('behaviour')
    } catch (err) {
      setError(err.message)
    }
    setLoading(false)
  }

  const acc  = data?.account
  const pred = data?.prediction

  return (
    <div>
      <div className="page-header">
        <div className="page-title">Account Lookup</div>
        <div className="page-desc">Search any loan account to see its risk profile and AI prediction</div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <form onSubmit={handleSearch} style={{ display: 'flex', gap: 12 }}>
          <div className="input-icon-wrap" style={{ flex: 1 }}>
            <span className="input-icon">🔍</span>
            <input
              className="input"
              style={{ paddingLeft: 40 }}
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Enter Loan ID or Customer ID — e.g. LN123456"
            />
          </div>
          <button className="btn btn-primary" type="submit" disabled={loading} style={{ padding: '10px 24px' }}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>

      {loading && <Loading text="Fetching account details..." />}
      {error   && <InfoBox>{error}</InfoBox>}

      {!data && !loading && !error && (
        <div className="card" style={{ padding: 0 }}>
          <Empty icon="🔍" title="Search for an account" desc="Enter a Loan ID or Customer ID above to see the full risk profile, AI prediction, and 16-month payment history." />
        </div>
      )}

      {data && acc && (
        <>
          {/* account header */}
          <div className="acct-hdr">
            <div className="acct-hdr-eyebrow">Loan Account</div>
            <div className="acct-hdr-id">{acc.loan_id}</div>
            <div className="acct-hdr-meta">
              {[
                { l: 'Customer ID',    v: acc.customer_id.substring(0, 28) },
                { l: 'Segment',        v: acc.segment },
                { l: 'Loan Type',      v: acc.loan_type },
                { l: 'Last Observed',  v: acc.last_month },
                { l: 'Current State',  v: <Badge state={acc.risk_state} /> },
              ].map(({ l, v }) => (
                <div key={l} className="acct-meta-item">
                  <div className="lbl">{l}</div>
                  <div className="val">{v}</div>
                </div>
              ))}
            </div>
          </div>

          {/* tabs */}
          <div className="tabs">
            {[
              { id: 'behaviour', label: '💳 Payment Behaviour' },
              { id: 'prediction', label: '🤖 AI Prediction' },
              { id: 'history', label: '📅 Payment History' },
            ].map(t => (
              <button key={t.id} className={`tab${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
                {t.label}
              </button>
            ))}
          </div>

          {tab === 'behaviour'  && <PaymentBehaviour acc={acc} />}
          {tab === 'prediction' && pred && <AIPrediction pred={pred} />}
          {tab === 'history'    && <PaymentHistory history={data.history} />}
        </>
      )}
    </div>
  )
}
