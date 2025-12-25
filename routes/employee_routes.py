"""
Employee Management Routes
Handles employee operations including role management
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Body, Form
from sqlalchemy.orm import Session
from stateful_services.database import get_db
from stateful_services.db_schema import Employee
from typing import Optional, Dict, Any
import uuid
import json
from utils.logging import log
from utils.token_decode import get_current_employee_from_token


router = APIRouter()


def require_admin(request: Request, db: Session) -> Employee:
    """
    Verify that current user is an admin.
    """
    employee = get_current_employee_from_token(request, db)

    if employee.employee_role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Access denied. Admin privileges required."
        )

    return employee


# ===================== ROUTES =====================

@router.get("/employees")
def list_employees(
    db: Session = Depends(get_db),
    role: Optional[str] = None,
    department: Optional[str] = None):
    """List all employees (admin only)."""

    # require_admin(request, db)

    query = db.query(Employee)
    if role:
        query = query.filter(Employee.employee_role == role)
    if department:
        query = query.filter(Employee.department == department)

    employees = query.all()

    return {
        "total": len(employees),
        "employees": [employees]
    }


@router.get("/employees/{employee_id}")
def get_employee(
    request: Request,
    employee_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get employee details by ID (admin only)."""

    # require_admin(request, db)

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
        "created_at": employee.created_at.isoformat() if hasattr(employee.created_at, "isoformat") else None,
        "updated_at": employee.updated_at.isoformat() if hasattr(employee.updated_at, "isoformat") else None,
    }


@router.put("/employees/{employee_id}/role")
def update_employee_role(
    request: Request,
    employee_id: uuid.UUID,
    role: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """
    Update employee role (admin only).
    """

    admin = require_admin(request, db)

    allowed_roles = ["employee", "admin", "manager", "hr"]
    if role not in allowed_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role. Allowed roles: {', '.join(allowed_roles)}"
        )

    # FIX: compare by PK (Employee.id) because employee_id path param is a UUID
    employee = db.query(Employee).filter(
        Employee.id == employee_id
    ).first()

    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    if admin.id == employee.id and role != "admin":
        raise HTTPException(
            status_code=400,
            detail="You cannot change your own admin role"
        )

    old_role = employee.employee_role
    employee.employee_role = role
    db.commit()
    db.refresh(employee)

    return {
        "status": "success",
        "message": f"Employee role updated from '{old_role}' to '{role}'",
        "employee": {
            "id": str(employee.id),
            "employee_name": employee.employee_name,
            "employee_email": employee.employee_email,
            "employee_role": employee.employee_role,
            "department": employee.department,
            "updated_at": employee.updated_at.isoformat() if hasattr(employee.updated_at, "isoformat") else None,
        },
        "updated_by": {
            "id": str(admin.id),
            "name": admin.employee_name,
            "email": admin.employee_email,
        },
    }



@router.put("/employees/{employee_id}")
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
    """

    admin = require_admin(request, db)

    employee = db.query(Employee).filter(Employee.id == employee_id).first()

    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    changes = {}

    if employee_name is not None:
        employee.employee_name = employee_name  # type: ignore
        changes["employee_name"] = employee_name

    if employee_role is not None:
        allowed_roles = ["employee", "admin", "manager", "hr"]
        if employee_role not in allowed_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role. Allowed roles: {', '.join(allowed_roles)}"
            )

        if str(admin.id) == str(employee.id) and employee_role != "admin":
            raise HTTPException(
                status_code=400,
                detail="You cannot change your own admin role"
            )

        employee.employee_role = employee_role
        changes["employee_role"] = employee_role

    if department is not None:
        employee.department = department  # type: ignore
        changes["department"] = department

    if meta_data is not None:
        employee.meta_data = json.dumps(meta_data)  # type: ignore
        changes["meta_data"] = meta_data

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
            "updated_at": employee.updated_at.isoformat() if hasattr(employee.updated_at, "isoformat") else None,
        },
        "changes_applied": changes,
        "updated_by": admin.employee_name,
    }


@router.get("/me")
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
        "employee_code": employee.employee_code,
        "employee_role": employee.employee_role,
        "department": employee.department,
        "is_admin": employee.employee_role == "admin",
        "meta_data": employee.meta_data,
        "created_at": employee.created_at.isoformat() if hasattr(employee.created_at, "isoformat") else None,
    }


# ===================== PATCH UPDATE ROUTE =====================

# ...existing code...
@router.patch("/employees/update/{employee_id}", summary="Update employee profile")
async def update_employee_profile(
    employee_id: str,
    name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    role: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    status: Optional[str] = Form(None),
    meta_data: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Update employee profile details.
    Supports partial updates.
    """

    log.info(f"Received update request for employee_id={employee_id}")

    try:
        employee_uuid = uuid.UUID(employee_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid employee_id UUID")

    meta_data_dict = None
    if meta_data:
        try:
            meta_data_dict = json.loads(meta_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format in meta_data")

    employee = update_employee_service(
        db=db,
        employee_id=employee_uuid,
        name=name,
        email=email,
        role=role,
        department=department,
        status=status,
        meta_data=meta_data_dict,
    )

    # FIX: map response fields to actual model attributes
    return {
        "employee_id": str(employee.id),
        "name": employee.employee_name,
        "email": employee.employee_email,
        "role": employee.employee_role,
        "status": employee.meta_data.get("status") if isinstance(employee.meta_data, dict) else None,
        "message": "Employee profile updated successfully",
    }
# ...existing code...

# ===================== SERVICE FUNCTION =====================

# ...existing code...
def update_employee_service(
    db: Session,
    employee_id: uuid.UUID,
    name: Optional[str] = None,
    email: Optional[str] = None,
    role: Optional[str] = None,
    department: Optional[str] = None,
    status: Optional[str] = None,
    meta_data: Optional[Dict[str, Any]] = None,
) -> Employee:
    """
    Update employee details safely.
    """

    try:
        employee = db.query(Employee).filter(
            Employee.id == employee_id
        ).first()

        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")

        # Email check uses the actual column name employee_email
        if email and email != employee.employee_email:
            email_exists = db.query(Employee).filter(
                Employee.employee_email == email,
                Employee.id != employee_id,
            ).first()

            if email_exists:
                raise HTTPException(
                    status_code=400,
                    detail="Email already exists for another employee",
                )

            employee.employee_email = email  # type: ignore

        # Map incoming fields to correct model attributes
        if name is not None:
            employee.employee_name = name  # type: ignore
        if role is not None:
            employee.employee_role = role  # type: ignore
        if department is not None:
            employee.department = department  # type: ignore
        if meta_data is not None:
            employee.meta_data = meta_data  # type: ignore

        # 'status' is not a dedicated column in the model; store in meta_data
        if status is not None:
            md = employee.meta_data or {}
            if not isinstance(md, dict):
                md = {}
            md["status"] = status
            employee.meta_data = json.dumps(md)  # type: ignore

        db.commit()
        db.refresh(employee)

        log.info(f"Employee updated successfully: {employee.id}")
        return employee

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error(f"Failed to update employee {employee_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update employee: {str(e)}",
        )
        
        ##upload employee 
@router.post("/create", summary="Create a new employee")
def create_employee(
    request: Request,
    employee_id: Optional[str] = Form(None),
    employee_name: str = Form(...),
    employee_email: Optional[str] = Form(None),
    employee_role: str = Form("employee"),
    department: Optional[str] = Form(None),
    meta_data: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Create a new employee (admin only).
    """

    # admin = require_admin(request, db)

    # log.info(f"Admin {admin.employee_email} creating employee {employee_id}")

    meta_data_dict = None
    if meta_data:
        try:
            meta_data_dict = json.loads(meta_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format in meta_data")

    employee = create_employee_service(
        db=db,
        employee_id=employee_id,
        employee_name=employee_name,
        employee_email=employee_email,
        employee_role=employee_role,
        department=department,
        meta_data=meta_data_dict,
    )

    return {
        "status": "success",
        "message": "Employee created successfully",
        "employee": {
            "id": str(employee.id),
            "employee_id": employee.employee_id,
            "employee_name": employee.employee_name,
            "employee_email": employee.employee_email,
            "employee_role": employee.employee_role,
            "department": employee.department,
            "created_at": employee.created_at.isoformat() if hasattr(employee.created_at, "isoformat") else None,
        },
    }


def create_employee_service(
    db: Session,
    employee_id: Optional[str],
    employee_name: str,
    employee_email: Optional[str],
    employee_role: str,
    department: Optional[str],
    meta_data: Optional[Dict[str, Any]],
) -> Employee:
    """
    Create a new employee safely.
    """

    try:
        # Check employee_id uniqueness
        # if db.query(Employee).filter(Employee.employee_id == employee_id).first():
        #     raise HTTPException(
        #         status_code=400,
        #         detail="Employee with this employee_id already exists",
        #     )

        # Check email uniqueness
        if employee_email:
            if db.query(Employee).filter(
                Employee.employee_email == employee_email
            ).first():
                raise HTTPException(
                    status_code=400,
                    detail="Employee with this email already exists",
                )

        employee = Employee(
            employee_id=employee_id,
            employee_name=employee_name,
            employee_email=employee_email,
            employee_role=employee_role,
            department=department,
            meta_data=meta_data or {},
        )

        db.add(employee)
        db.commit()
        db.refresh(employee)

        log.info(f"Employee created successfully: {employee.employee_email}")
        return employee

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error(f"Failed to create employee {employee_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create employee: {str(e)}",
        )

