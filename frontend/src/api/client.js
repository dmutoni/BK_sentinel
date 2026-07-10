const BASE = '/api'

function token() {
  return localStorage.getItem('bk_token') || ''
}

async function request(path, options = {}) {
  const res = await fetch(BASE + path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token()}`,
      ...(options.headers || {}),
    },
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export const api = {
  // auth
  login:   (username, password) => request('/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) }),
  signup:  (username, password, name) => request('/auth/signup', { method: 'POST', body: JSON.stringify({ username, password, name }) }),
  logout:  ()                   => request('/auth/logout', { method: 'POST', body: '{}' }),
  me:      ()                   => request('/auth/me'),

  // meta
  months:   () => request('/overview/months'),
  segments: () => request('/overview/segments'),

  // overview
  snapshot: (month, segment) => request(`/overview/snapshot?month=${month}&segment=${encodeURIComponent(segment)}`),
  trend:    ()               => request('/overview/trend'),

  // transition
  matrix:             ()                           => request('/transition/matrix'),
  forecast:           (horizon)                    => request(`/transition/forecast?horizon=${horizon}`),
  portfolioForecast:  (month, segment, horizon=6) => request(`/transition/portfolio-forecast?month=${month}&segment=${encodeURIComponent(segment)}&horizon=${horizon}`),

  // absorption
  absorption: () => request('/absorption/summary'),

  // watchlist
  watchlist: (month, segment, filterType, horizon, page, pageSize = 25) =>
    request(`/watchlist?month=${month}&segment=${encodeURIComponent(segment)}&filter_type=${filterType}&horizon=${horizon}&page=${page}&page_size=${pageSize}`),

  // account
  account: (id) => request(`/account/${encodeURIComponent(id)}`),
}
