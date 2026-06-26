"""
BK Sentinel — Authentication Middleware
Simple token-based authentication for the API.
Tokens are stored in memory and issued on login.
"""

import secrets
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config import USERS

# ── token store ───────────────────────────────────────────────
# Maps token string → user dict
# In production this would be a Redis store or JWT
_token_store: dict[str, dict] = {}

security = HTTPBearer(auto_error=False)


def issue_token(username: str) -> str:
    """Generate a new token for the given username."""
    token = secrets.token_hex(32)
    _token_store[token] = {"username": username, **USERS[username]}
    return token


def revoke_token(username: str) -> None:
    """Revoke all tokens for the given username."""
    to_remove = [t for t, u in _token_store.items() if u["username"] == username]
    for t in to_remove:
        del _token_store[t]


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    FastAPI dependency — validates the bearer token.
    Raises 401 if the token is missing or invalid.
    Usage: user = Depends(get_current_user)
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")
    token = credentials.credentials
    if token not in _token_store:
        raise HTTPException(status_code=401, detail="Invalid or expired token. Please log in again.")
    return _token_store[token]


def validate_credentials(username: str, password: str) -> dict:
    """
    Validate username and password against the user store.
    Returns the user dict if valid, raises 401 otherwise.
    """
    user = USERS.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    return user
