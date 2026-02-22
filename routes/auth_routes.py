from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse
from sqlalchemy.orm import Session
import config
from stateful_services.database import get_db
from services.auth_service import MicrosoftAuthService
from stateful_services.db_schema import Employee
from datetime import datetime, timezone
import uuid
import json
import urllib.parse
from typing import Optional

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
    print(f"DEBUG: Redirecting to Microsoft login: {auth_url}")
    return RedirectResponse(url=auth_url)


@router.get("/callback")
async def auth_callback(
    code: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Handle OAuth callback from Microsoft"""
    
    # Handle OAuth errors
    if error:
        print(f"OAuth Error: {error}")
        print(f"Error Description: {error_description}")
        raise HTTPException(
            status_code=400, 
            detail=f"OAuth authentication failed: {error_description or error}"
        )
    
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not provided")
    
    # Check if code looks like a JWT (someone is misusing the endpoint)

    print(f"DEBUG: Received authorization code (length: {len(code)})")

    try:
        # Exchange code for token
        token_result = await auth_service.get_token_from_code(code)
        if not token_result:
            raise HTTPException(
                status_code=401, 
                detail="Failed to obtain access token. Please try logging in again."
            )

        access_token = token_result.get("access_token")
        if not access_token:
            print("ERROR: No access_token in token_result:", token_result)
            raise HTTPException(status_code=401, detail="No access token in response")

        print("DEBUG: Successfully obtained access token")

        # Get employee information from Microsoft Graph
        user_info = await auth_service.get_user_info(access_token)
        if not user_info:
            raise HTTPException(status_code=401, detail="Failed to fetch employee information")
        
        print("DEBUG USERINFO KEYS:", list(user_info.keys()))
        
        # Map Microsoft user to Employee - safely extract all fields
        microsoft_id = user_info.get("id")
        if not microsoft_id:
            raise HTTPException(status_code=400, detail="No user ID in Microsoft response")
        print(user_info)    
        employee_name = user_info.get("displayName") or "Unknown User"
        
        # Safely get email - try multiple fields
        employee_email = (
            user_info.get("mail") or 
            user_info.get("userPrincipalName") or 
            user_info.get("email") or
            user_info.get("preferredUsername")
        )
        
        if not employee_email:
            print("ERROR: No email found. Available fields:", list(user_info.keys()))
            raise HTTPException(
                status_code=400, 
                detail="Unable to retrieve email from Microsoft account. Please ensure your account has an email configured."
            )
        
        print(f"DEBUG: Extracted - ID: {microsoft_id}, Name: {employee_name}, Email: {employee_email}")
        
        department = user_info.get("department")

        # Check if employee exists
        employee = db.query(Employee).filter(Employee.employee_email == employee_email).first()

        if not employee:
            # First user ever becomes admin; all subsequent users are employees
            is_first_user = db.query(Employee.id).first() is None
            employee_role = "admin" if is_first_user else "employee"

            # Create new employee
            employee = Employee(
                employee_id=microsoft_id,
                employee_name=employee_name,
                employee_email=employee_email,
                employee_role=employee_role,
                department=department,
                meta_data={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            db.add(employee)
            db.commit()
            db.refresh(employee)
            print(f"DEBUG: Created new employee with ID: {employee.id}")
        else:
            # Update employee info
            employee.employee_id = microsoft_id
            employee.employee_name = employee_name
            employee.employee_email = employee_email    
            # employee.department = department
            employee.updated_at = datetime.now(timezone.utc)
            db.commit()
            print(f"DEBUG: Updated existing employee ID: {employee.id}")

        # Create JWT token with safe data
        jwt_payload = {
            "id": microsoft_id,
            "email": employee_email,
            "name": employee_name
        }
        jwt_token = auth_service.create_jwt_token(jwt_payload)

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
        frontend_callback_url = f"{config.FRONTEND_URL}/auth/callback?token={jwt_token}&employee={urllib.parse.quote(json.dumps(employee_data))}"
        return RedirectResponse(url=frontend_callback_url)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in auth_callback: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")


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
    microsoft_id = payload.get("sub")  # JWT 'sub' field contains Microsoft ID
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