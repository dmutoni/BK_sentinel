import { useState } from 'react'
import { api } from '../api/client'

export default function Login({ onLogin }) {
  const [mode,     setMode]     = useState('login') // 'login' | 'signup'
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [name,     setName]     = useState('')
  const [error,    setError]    = useState('')
  const [loading,  setLoading]  = useState(false)
  const [showPassword, setShowPassword] = useState(false)

  function toggleMode() {
    setMode(m => (m === 'login' ? 'signup' : 'login'))
    setError('')
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!username || !password || (mode === 'signup' && !name)) return
    setError(''); setLoading(true)
    try {
      const data = mode === 'login'
        ? await api.login(username, password)
        : await api.signup(username, password, name)
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
          <div className="lf-welcome">{mode === 'login' ? 'Welcome back' : 'Create an account'}</div>
          <div className="lf-sub">
            {mode === 'login' ? 'Sign in to BK Sentinel' : 'Sign up to start using BK Sentinel'}
          </div>

          {error && <div className="lf-error">⚠ {error}</div>}

          <form onSubmit={handleSubmit}>
            {mode === 'signup' && (
              <div className="field">
                <label className="field-label">Full name</label>
                <input
                  className="input"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  placeholder="Enter your full name"
                  autoFocus
                />
              </div>
            )}
            <div className="field">
              <label className="field-label">Username</label>
              <input
                className="input"
                value={username}
                onChange={e => setUsername(e.target.value)}
                placeholder="Enter your username"
                autoFocus={mode === 'login'}
              />
            </div>
            <div className="field">
              <label className="field-label">Password</label>
              <div style={{ position: 'relative' }}>
                <input
                  className="input"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder={mode === 'login' ? 'Enter your password' : 'At least 6 characters'}
                  style={{ paddingRight: 40 }}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(s => !s)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  title={showPassword ? 'Hide password' : 'Show password'}
                  style={{
                    position: 'absolute',
                    right: 10,
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    padding: 0,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    color: '#6b7280',
                  }}
                >
                  {showPassword ? (
                    /* eye-off icon */
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
                      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
                      <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
                      <line x1="1" y1="1" x2="23" y2="23" />
                    </svg>
                  ) : (
                    /* eye icon */
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                      <circle cx="12" cy="12" r="3" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
            <button
              className="btn btn-primary"
              style={{ width: '100%', justifyContent: 'center', padding: '12px', marginTop: 4 }}
              type="submit"
              disabled={loading}
            >
              {loading
                ? (mode === 'login' ? 'Signing in...' : 'Creating account...')
                : (mode === 'login' ? 'Sign in →' : 'Create account →')}
            </button>
          </form>

          <div className="lf-hint">
            {mode === 'login' ? "Don't have an account?" : 'Already have an account?'}{' '}
            <span
              onClick={toggleMode}
              style={{ color: '#0e5fbf', fontWeight: 600, cursor: 'pointer' }}
            >
              {mode === 'login' ? 'Sign up' : 'Sign in'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
