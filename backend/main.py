"""
BK Sentinel — FastAPI Backend
==============================
Three-layer dynamic credit risk monitoring system for Bank of Kigali.

Author:      Denyse Mutoni Uwingeneye
Institution: African Leadership University
Supervisor:  Emmanuel Adjei

Run with:
    cd backend
    uvicorn main:app --reload --port 8000

API docs available at:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import (
    auth_router,
    overview_router,
    transition_router,
    watchlist_router,
    account_router,
    absorption_router,
)

# ── app factory ───────────────────────────────────────────────
app = FastAPI(
    title       = "BK Sentinel API",
    description = (
        "Three-layer dynamic credit risk transition system. "
        "Layer 1: ML classification. "
        "Layer 2: Markov chain transition matrix. "
        "Layer 3: Absorbing state analysis."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS — allow the React frontend on port 3000 ──────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── register routers ──────────────────────────────────────────
app.include_router(auth_router)
app.include_router(overview_router)
app.include_router(transition_router)
app.include_router(watchlist_router)
app.include_router(account_router)
app.include_router(absorption_router)


# ── health check ──────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "system":  "BK Sentinel",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


# ── startup event ─────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Pre-load all data and models into cache on startup."""
    print("=" * 50)
    print("BK Sentinel Backend Starting Up")
    print("=" * 50)
    from database.loader import (
        get_portfolio_df,
        get_transition_matrix,
        get_model,
        get_absorption_data,
    )
    get_portfolio_df()
    get_transition_matrix()
    get_model()
    get_absorption_data()
    print("=" * 50)
    print("All data loaded. Server ready.")
    print("Docs: http://localhost:8000/docs")
    print("=" * 50)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
