"""
BK Sentinel — Auth Database
============================
SQLAlchemy layer for user accounts.

Uses a single DATABASE_URL env var:
  • Local dev  → defaults to a SQLite file (no setup needed).
  • Production → set DATABASE_URL to the Render Postgres connection string.

The same code runs on both; only the connection string changes.
Loan/portfolio data is NOT stored here — it stays as CSV/pickle files
loaded by database/loader.py. This database holds user accounts only.
"""

import os

import bcrypt
from sqlalchemy import Column, DateTime, String, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

# ── connection ────────────────────────────────────────────────
# Render exposes the URL as `postgres://…` but SQLAlchemy 2.0 needs
# the `postgresql://` scheme, so normalise it.
_raw_url = os.environ.get("DATABASE_URL", "sqlite:///./bk_sentinel.db")
if _raw_url.startswith("postgres://"):
    _raw_url = _raw_url.replace("postgres://", "postgresql://", 1)
DATABASE_URL = _raw_url

_is_sqlite = DATABASE_URL.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# ── model ─────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    username      = Column(String, primary_key=True, index=True)
    name          = Column(String, nullable=False)
    role          = Column(String, nullable=False, default="Credit Analyst")
    password_hash = Column(String, nullable=False)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    def public(self) -> dict:
        """The user fields safe to expose (never the password hash)."""
        return {"username": self.username, "name": self.name, "role": self.role}


# ── password hashing ──────────────────────────────────────────
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


# ── init & seed ───────────────────────────────────────────────
def init_db() -> None:
    """
    Create the users table if missing, and seed the built-in accounts
    (analyst / manager / denyse) on first run. Safe to call on every startup.
    """
    from config import USERS  # seed definitions: {username: {password, name, role}}

    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        for username, info in USERS.items():
            if db.get(User, username) is None:
                db.add(User(
                    username      = username,
                    name          = info["name"],
                    role          = info.get("role", "Credit Analyst"),
                    password_hash = hash_password(info["password"]),
                ))
        db.commit()
