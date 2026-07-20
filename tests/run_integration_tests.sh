#!/bin/bash
# BK Sentinel — API Integration Tests (Table 10, section 4.3.5)
#
# 1. Start the backend first, in another terminal:
#      cd backend && uvicorn main:app --port 8000
# 2. Then run this script from the repo root:
#      bash tests/run_integration_tests.sh

B=http://localhost:8000/api
U='{"username":"test_analyst","password":"secret123"}'

echo "BK Sentinel — API Integration Tests"
echo "======================================================================"

# create the test user (ignored if it already exists), then log in
curl -s -X POST $B/auth/signup -H "Content-Type: application/json" \
  -d '{"username":"test_analyst","password":"secret123","name":"Test Analyst"}' > /dev/null
TOKEN=$(curl -s -X POST $B/auth/login -H "Content-Type: application/json" -d "$U" \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))")
[ -n "$TOKEN" ] && echo "[PASS] POST /auth/login      valid credentials -> bearer token" \
                || { echo "[FAIL] POST /auth/login      no token — is the backend running?"; exit 1; }
A="Authorization: Bearer $TOKEN"

C=$(curl -s -o /dev/null -w "%{http_code}" -X POST $B/auth/login \
  -H "Content-Type: application/json" -d '{"username":"test_analyst","password":"wrong"}')
[ "$C" = "401" ] && echo "[PASS] POST /auth/login      invalid credentials -> 401" \
                 || echo "[FAIL] invalid login -> $C"

curl -s "$B/overview/snapshot?month=2026-01" -H "$A" | python3 -c "
import sys, json; d = json.load(sys.stdin)
print(f'[PASS] GET /overview/snapshot Jan-2026: {d[\"total\"]:,} accounts, {d[\"counts\"]}')"

curl -s "$B/overview/trend" -H "$A" | python3 -c "
import sys, json; d = json.load(sys.stdin)
pts = d if isinstance(d, list) else d.get('trend') or d.get('months') or []
tag = 'PASS' if len(pts) == 16 else 'FAIL'
print(f'[{tag}] GET /overview/trend    {len(pts)} monthly points (expect 16)')"

curl -s "$B/watchlist?month=2026-01" -H "$A" | python3 -c "
import sys, json; d = json.load(sys.stdin)
print(f'[PASS] GET /watchlist         {d[\"total\"]:,} flagged, paginated')"

curl -s "$B/transition/matrix" -H "$A" | python3 -c "
import sys, json; d = json.load(sys.stdin)
m = d.get('matrix') or d
rows = m if isinstance(m, list) else list(m.values())
sums = [round(sum(r.values() if isinstance(r, dict) else r), 3) for r in rows]
tag = 'PASS' if all(abs(x-1.0) <= 0.001 for x in sums) else 'FAIL'
print(f'[{tag}] GET /transition/matrix row sums = {sums}')"

curl -s "$B/transition/forecast?months=3" -H "$A" | python3 -c "
import sys, json; d = json.load(sys.stdin)
print('[PASS] GET /transition/forecast P^n returned (n=3)' if not d.get('detail') else f'[FAIL] {d}')"

curl -s "$B/absorption/summary" -H "$A" | python3 -c "
import sys, json; d = json.load(sys.stdin)
print('[PASS] GET /absorption/summary N, B, months-to-default returned' if not d.get('detail') else f'[FAIL] {d}')"

# pick a real loan id from the verified panel, test valid + invalid lookup
LOAN=$(python3 -c "
import pandas as pd
df = pd.read_csv('model-training/bk_sentinel_verified.csv', usecols=['loan_id'], low_memory=False)
print(df['loan_id'].iloc[100])")
curl -s "$B/account/$LOAN" -H "$A" | python3 -c "
import sys, json; d = json.load(sys.stdin)
print(f'[PASS] GET /account/{{id}}     valid ID -> {len(d.get(\"history\", []))} history rows + prediction' if not d.get('detail') else f'[FAIL] {d}')"

C=$(curl -s -o /dev/null -w "%{http_code}" "$B/account/UNKNOWN_ID_999" -H "$A")
[ "$C" = "404" ] && echo "[PASS] GET /account/{id}     unknown ID -> 404" || echo "[FAIL] unknown id -> $C"

C=$(curl -s -o /dev/null -w "%{http_code}" "$B/overview/snapshot?month=2026-01")
[ "$C" = "401" ] && echo "[PASS] auth middleware       tokenless request -> 401 rejected" || echo "[FAIL] no-token -> $C"

echo "======================================================================"
