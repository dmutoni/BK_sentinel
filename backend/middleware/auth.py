"""
BK Sentinel — Authentication Middleware
=======================================
Token-based auth backed by the users database (see database/db.py).

  • Passwords are stored HASHED (bcrypt) — never in plaintext.
  • Accounts live in the database, so signups persist across restarts
    and redeploys (when DATABASE_URL points at Postgres).
  • Tokens are kept in memory and re-issued on login; a restart simply
    means users log in again.
"""

import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from database.db import SessionLocal, User, hash_password, verify_password

# ── token store ───────────────────────────────────────────────
# Maps token string → public user dict {username, name, role}.
# In production this would be a Redis store or JWT.
_token_store: dict[str, dict] = {}

security = HTTPBearer(auto_error=False)


def issue_token(user: dict) -> str:
    """Generate a new token for an authenticated user."""
    token = secrets.token_hex(32)
    _token_store[token] = dict(user)
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
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required. Please log in.")
    token = credentials.credentials
    if token not in _token_store:
        raise HTTPException(status_code=401, detail="Invalid or expired token. Please log in again.")
    return _token_store[token]


def validate_credentials(username: str, password: str) -> dict:
    """
    Validate username and password against the database.
    Returns the public user dict if valid, raises 401 otherwise.
    """
    with SessionLocal() as db:
        user = db.get(User, (username or "").strip())
        if not user or not verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="Incorrect username or password.")
        return user.public()


def create_user(username: str, password: str, name: str, role: str = "Credit Analyst") -> dict:
    """
    Register a new account with a hashed password. Raises 400 if the
    username is taken or the input is invalid. Returns the public user dict.
    """
    username = (username or "").strip()
    password = (password or "").strip()
    name = (name or "").strip() or username

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    with SessionLocal() as db:
        if db.get(User, username) is not None:
            raise HTTPException(status_code=400, detail="That username is already taken.")
        user = User(
            username      = username,
            name          = name,
            role          = role or "Credit Analyst",
            password_hash = hash_password(password),
        )
        db.add(user)
        db.commit()
        return user.public()
