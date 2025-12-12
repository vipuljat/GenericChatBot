from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.orm import Session
from stateful_services.database import get_db
from services.auth_service import MicrosoftAuthService
from stateful_services.db_schema import Employee  # Updated Employee model
from datetime import datetime, timezone
import uuid
import json
import urllib.parse

router = APIRouter()
auth_service = MicrosoftAuthService()

# Handle preflight OPTIONS requests for CORS
@router.options("/{any_path:path}")
async def preflight_handler(any_path: str):
    return JSONResponse(status_code=200, content={"message": "OK"})


@router.get("/login")
async def login():
    """Redirect employee to Microsoft login page"""
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

    # Get employee information from Microsoft Graph
    user_info = await auth_service.get_user_info(access_token)
    if not user_info:
        raise HTTPException(status_code=401, detail="Failed to fetch employee information")

    # Map Microsoft user to Employee
    microsoft_id = user_info["id"]
    employee_name = user_info.get("displayName")
    employee_email = user_info.get("mail") or user_info.get("userPrincipalName")
    employee_role = "employee"  # Customize as needed
    department = None  # Optional, can extract from Azure AD if needed

    # Check if employee exists
    employee = db.query(Employee).filter(Employee.employee_id == microsoft_id).first()

    if not employee:
        # Create new employee
        employee = Employee(
            employee_id=microsoft_id,
            employee_name=employee_name,
            employee_email=employee_email,
            employee_role=employee_role,
            department=department,
            meta_data={"email": employee_email},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        db.add(employee)
        db.commit()
        db.refresh(employee)
    else:
        # Update employee info
        employee.employee_name = employee_name
        employee.employee_email = employee_email
        employee.updated_at = datetime.now(timezone.utc)
        db.commit()

    # Create JWT token
    jwt_token = auth_service.create_jwt_token({"id": microsoft_id, "email": employee_email})

    # Prepare employee data for frontend
    employee_data = {
        "id": str(employee.id),
        "employee_id": employee.employee_id,
        "employee_name": employee.employee_name,
        "employee_email": employee.employee_email,
        "employee_role": employee.employee_role,
        "department": employee.department
    }

    # Redirect to frontend callback page with token and employee data
    frontend_callback_url = f"http://localhost:3000/auth/callback?token={jwt_token}&employee={urllib.parse.quote(json.dumps(employee_data))}"
    return RedirectResponse(url=frontend_callback_url)


@router.get("/logout")
async def logout():
    """Logout employee"""
    return JSONResponse(content={"message": "Logged out successfully"})


@router.get("/me")
async def get_current_employee(request: Request, db: Session = Depends(get_db)):
    """Get current authenticated employee information"""
    # Extract token from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = auth_header.replace("Bearer ", "")

    # Verify token
    payload = auth_service.verify_jwt_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Get employee from database
    microsoft_id = payload.get("id")  # JWT contains Microsoft ID
    employee = db.query(Employee).filter(Employee.employee_id == microsoft_id).first()

    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    return {
        "id": str(employee.id),
        "employee_id": employee.employee_id,
        "employee_name": employee.employee_name,
        "employee_email": employee.employee_email,
        "employee_role": employee.employee_role,
        "department": employee.department,
        "meta_data": employee.meta_data,
        "created_at": employee.created_at,
        "updated_at": employee.updated_at
    }
