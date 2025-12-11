from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.orm import Session
from stateful_services.database import get_db
from services.auth_service import MicrosoftAuthService
from models.user import User
from datetime import datetime, timezone
from typing import Optional

router = APIRouter()
auth_service = MicrosoftAuthService()

@router.get("/login")
async def login():
    """Redirect user to Microsoft login page"""
    auth_url = auth_service.get_authorization_url()
    return RedirectResponse(url=auth_url)

@router.get("/callback")
async def auth_callback(code: str, db: Session = Depends(get_db)):
    """Handle OAuth callback from Microsoft"""
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not provided")
    
    # Exchange code for token
    token_result = await auth_service.get_token_from_code(code)
    if not token_result:
        raise HTTPException(status_code=401, detail="Failed to obtain access token")
    
    access_token = token_result["access_token"]
    
    # Get user information from Microsoft Graph
    user_info = await auth_service.get_user_info(access_token)
    if not user_info:
        raise HTTPException(status_code=401, detail="Failed to fetch user information")
    
    # Check if user exists in database
    microsoft_id = user_info["id"]
    email = user_info.get("mail") or user_info.get("userPrincipalName")
    name = user_info.get("displayName")
    
    user = db.query(User).filter(User.microsoft_id == microsoft_id).first()
    
    if not user:
        # Create new user
        user = User(
            email=email,
            name=name,
            microsoft_id=microsoft_id,
            is_active=True,
            created_at=datetime.now(timezone.utc),
            last_login=datetime.now(timezone.utc)
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # Update last login
        user.last_login = datetime.now(timezone.utc)
        db.commit()
    
    # Create JWT token
    jwt_token = auth_service.create_jwt_token(user_info)
    
    # Create user data for frontend
    import json
    import urllib.parse
    
    user_data = {
        "id": user.id,
        "email": user.email,
        "name": user.name
    }
    
    # Redirect to frontend callback page with token and user data
    frontend_callback_url = f"http://localhost:3000/auth/callback?token={jwt_token}&user={urllib.parse.quote(json.dumps(user_data))}"
    return RedirectResponse(url=frontend_callback_url)

@router.get("/logout")
async def logout():
    """Logout user"""
    return JSONResponse(content={"message": "Logged out successfully"})

@router.get("/me")
async def get_current_user(request: Request, db: Session = Depends(get_db)):
    """Get current authenticated user information"""
    # Extract token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = auth_header.replace("Bearer ", "")
    
    # Verify token
    payload = auth_service.verify_jwt_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    # Get user from database
    email = payload.get("email")
    user = db.query(User).filter(User.email == email).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "microsoft_id": user.microsoft_id,
        "is_active": user.is_active,
        "created_at": user.created_at,
        "last_login": user.last_login
    }
