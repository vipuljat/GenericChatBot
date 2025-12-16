"""
Employee Management Routes
Handles employee operations including role management
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Body
from sqlalchemy.orm import Session
from stateful_services.database import get_db
from stateful_services.db_schema import Employee
from typing import Optional, List
import uuid

router = APIRouter()


def get_current_employee_from_token(request: Request, db: Session) -> Employee:
    """
    Extract and verify current employee from auth token.
    Raises HTTPException if not authenticated.
    """
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        token = auth_header.replace("Bearer ", "")
        
        from auth.auth_service import MicrosoftAuthService
        auth_service = MicrosoftAuthService()
        payload = auth_service.verify_jwt_token(token)
        
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        email = payload.get("email")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        
        employee = db.query(Employee).filter(Employee.employee_email == email).first()
        
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        return employee
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")


def require_admin(request: Request, db: Session) -> Employee:
    """
    Verify that current user is an admin.
    Returns employee if admin, raises HTTPException otherwise.
    """
    employee = get_current_employee_from_token(request, db)
    
    if employee.employee_role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Access denied. Admin privileges required."
        )
    
    return employee


# ===================== ROUTES =====================

@router.get("/employees", tags=["Employees"])
def list_employees(
    request: Request,
    db: Session = Depends(get_db),
    role: Optional[str] = None
):
    """List all employees (admin only). Can filter by role."""
    # Require admin
    require_admin(request, db)
    
    query = db.query(Employee)
    if role:
        query = query.filter(Employee.employee_role == role)
    
    employees = query.all()
    
    return {
        "total": len(employees),
        "employees": [
            {
                "id": str(emp.id),
                "employee_id": emp.employee_id,
                "employee_name": emp.employee_name,
                "employee_email": emp.employee_email,
                "employee_role": emp.employee_role,
                "department": emp.department,
                "created_at": emp.created_at.isoformat() if hasattr(emp.created_at, 'isoformat') else None,
                "updated_at": emp.updated_at.isoformat() if hasattr(emp.updated_at, 'isoformat') else None
            }
            for emp in employees
        ]
    }


@router.get("/employees/{employee_id}", tags=["Employees"])
def get_employee(
    request: Request,
    employee_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get employee details by ID (admin only)."""
    # Require admin
    require_admin(request, db)
    
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    
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
        "created_at": employee.created_at.isoformat() if hasattr(employee.created_at, 'isoformat') else None,
        "updated_at": employee.updated_at.isoformat() if hasattr(employee.updated_at, 'isoformat') else None
    }


@router.put("/employees/{employee_id}/role", tags=["Employees"])
def update_employee_role(
    request: Request,
    employee_id: uuid.UUID,
    role: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """
    Update employee role (admin only).
    
    Allowed roles: employee, admin, manager, hr
    """
    # Require admin to perform this action
    admin = require_admin(request, db)
    
    # Validate role
    allowed_roles = ["employee", "admin", "manager", "hr"]
    if role not in allowed_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role. Allowed roles: {', '.join(allowed_roles)}"
        )
    
    # Get employee to update
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    # Prevent admin from demoting themselves
    if admin.id == employee.id and role != "admin":
        raise HTTPException(
            status_code=400,
            detail="You cannot change your own admin role"
        )
    
    # Store old role for logging
    old_role = employee.employee_role
    
    # Update role
    employee.employee_role = role
    db.commit()
    db.refresh(employee)
    
    return {
        "status": "success",
        "message": f"Employee role updated from '{old_role}' to '{role}'",
        "employee": {
            "id": str(employee.id),
            "employee_id": employee.employee_id,
            "employee_name": employee.employee_name,
            "employee_email": employee.employee_email,
            "employee_role": employee.employee_role,
            "department": employee.department,
            "updated_at": employee.updated_at.isoformat() if hasattr(employee.updated_at, 'isoformat') else None
        },
        "updated_by": {
            "id": str(admin.id),
            "name": admin.employee_name,
            "email": admin.employee_email
        }
    }


@router.put("/employees/{employee_id}", tags=["Employees"])
def update_employee(
    request: Request,
    employee_id: uuid.UUID,
    employee_name: Optional[str] = Body(None),
    employee_role: Optional[str] = Body(None),
    department: Optional[str] = Body(None),
    meta_data: Optional[dict] = Body(None),
    db: Session = Depends(get_db)
):
    """
    Update employee details (admin only).
    
    Can update: name, role, department, metadata
    """
    # Require admin
    admin = require_admin(request, db)
    
    employee = db.query(Employee).filter(Employee.id == employee_id).first()
    
    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    # Track changes
    changes = {}
    
    if employee_name is not None:
        employee.employee_name = employee_name
        changes['employee_name'] = employee_name
    
    if employee_role is not None:
        allowed_roles = ["employee", "admin", "manager", "hr"]
        if employee_role not in allowed_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role. Allowed roles: {', '.join(allowed_roles)}"
            )
        
        # Prevent self-demotion
        if admin.id == employee.id and employee_role != "admin":
            raise HTTPException(
                status_code=400,
                detail="You cannot change your own admin role"
            )
        
        employee.employee_role = employee_role
        changes['employee_role'] = employee_role
    
    if department is not None:
        employee.department = department
        changes['department'] = department
    
    if meta_data is not None:
        employee.meta_data = meta_data
        changes['meta_data'] = meta_data
    
    if not changes:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    db.commit()
    db.refresh(employee)
    
    return {
        "status": "success",
        "message": "Employee updated successfully",
        "employee": {
            "id": str(employee.id),
            "employee_id": employee.employee_id,
            "employee_name": employee.employee_name,
            "employee_email": employee.employee_email,
            "employee_role": employee.employee_role,
            "department": employee.department,
            "meta_data": employee.meta_data,
            "updated_at": employee.updated_at.isoformat() if hasattr(employee.updated_at, 'isoformat') else None
        },
        "changes_applied": changes,
        "updated_by": admin.employee_name
    }


@router.get("/me", tags=["Employees"])
def get_current_employee(
    request: Request,
    db: Session = Depends(get_db)
):
    """Get current logged-in employee's information."""
    employee = get_current_employee_from_token(request, db)
    
    return {
        "id": str(employee.id),
        "employee_id": employee.employee_id,
        "employee_name": employee.employee_name,
        "employee_email": employee.employee_email,
        "employee_role": employee.employee_role,
        "department": employee.department,
        "is_admin": employee.employee_role == "admin",
        "meta_data": employee.meta_data,
        "created_at": employee.created_at.isoformat() if hasattr(employee.created_at, 'isoformat') else None
    }
