"""
BK Sentinel — Auth Router
Handles login, logout, and current user endpoints.
"""

from fastapi import APIRouter, Depends
from schemas.models import LoginRequest, LoginResponse, SignupRequest, UserResponse
from middleware.auth import (
    create_user, get_current_user, issue_token, revoke_token, validate_credentials
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """
    Authenticate with username and password.
    Returns a bearer token to use in subsequent requests.
    """
    user  = validate_credentials(req.username, req.password)
    token = issue_token(user)
    return LoginResponse(token=token, name=user["name"], role=user["role"])


@router.post("/signup", response_model=LoginResponse)
def signup(req: SignupRequest):
    """
    Create a new account and log the user straight in.
    """
    user  = create_user(req.username, req.password, req.name)
    token = issue_token(user)
    return LoginResponse(token=token, name=user["name"], role=user["role"])


@router.post("/logout")
def logout(user: dict = Depends(get_current_user)):
    """Revoke the current user's token."""
    revoke_token(user["username"])
    return {"message": f"Goodbye, {user['name']}."}


@router.get("/me", response_model=UserResponse)
def me(user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return UserResponse(
        username=user["username"],
        name=user["name"],
        role=user["role"],
    )
