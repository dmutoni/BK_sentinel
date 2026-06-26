import { useState } from 'react'
import { api } from '../api/client'

export default function Login({ onLogin }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error,    setError]    = useState('')
  const [loading,  setLoading]  = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!username || !password) return
    setError(''); setLoading(true)
    try {
      const data = await api.login(username, password)
      localStorage.setItem('bk_token', data.token)
      onLogin(data)
    } catch (err) {
      setError(err.message)
    }
    setLoading(false)
  }

  return (
    <div className="login-root">
      {/* left panel */}
      <div className="login-left">
        <div className="login-brand">
          <div className="login-brand-icon">🏦</div>
          <div>
            <div className="login-brand-name">BK Sentinel</div>
            <div className="login-brand-tag">Credit Risk Monitoring</div>
          </div>
        </div>

        <div className="login-headline">
          Dynamic Credit<br />Risk Intelligence
        </div>

        <div className="login-body">
          A three-layer ML and Markov chain system that predicts loan defaults
          before they happen, enabling Bank of Kigali credit teams to intervene
          at the right moment.
        </div>

        <div className="login-stats">
          <div className="ls-item">
            <div className="ls-val">93.6%</div>
            <div className="ls-lbl">Model F1 Score</div>
          </div>
          <div className="ls-item">
            <div className="ls-val">16mo</div>
            <div className="ls-lbl">Panel Data</div>
          </div>
          <div className="ls-item">
            <div className="ls-val">3</div>
            <div className="ls-lbl">Analysis Layers</div>
          </div>
        </div>
      </div>

      {/* right panel */}
      <div className="login-right">
        <div className="login-form-box">
          <div className="lf-welcome">Welcome back</div>
          <div className="lf-sub">Sign in to BK Sentinel</div>

          {error && <div className="lf-error">⚠ {error}</div>}

          <form onSubmit={handleSubmit}>
            <div className="field">
              <label className="field-label">Username</label>
              <input
                className="input"
                value={username}
                onChange={e => setUsername(e.target.value)}
                placeholder="Enter your username"
                autoFocus
              />
            </div>
            <div className="field">
              <label className="field-label">Password</label>
              <input
                className="input"
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="Enter your password"
              />
            </div>
            <button
              className="btn btn-primary"
              style={{ width: '100%', justifyContent: 'center', padding: '12px', marginTop: 4 }}
              type="submit"
              disabled={loading}
            >
              {loading ? 'Signing in...' : 'Sign in →'}
            </button>
          </form>

          <div className="lf-hint">
            Demo accounts<br />
            <strong>analyst</strong> / bk2026 &nbsp;·&nbsp;
            <strong>manager</strong> / bk2026 &nbsp;·&nbsp;
            <strong>denyse</strong> / alu2026
          </div>
        </div>
      </div>
    </div>
  )
}
