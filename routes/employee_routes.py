"""
Employee Management Routes
Handles employee operations including role management
"""

from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException, Request, Body, Form, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from stateful_services.database import get_db
from stateful_services.db_schema import Employee, ChatbotPermission, ChatbotAccess
from typing import Optional, Dict, Any
import uuid
import json
import openpyxl
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


@router.delete("/{employee_id}")
def delete_employee(
    request: Request,
    employee_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """
    Delete an employee (admin only).
    """

    admin = require_admin(request, db)

    employee = db.query(Employee).filter(Employee.id == employee_id).first()

    if not employee:
        raise HTTPException(status_code=404, detail="Employee not found")

    # Prevent admins from deleting themselves
    if admin.id == employee.id:
        raise HTTPException(
            status_code=400,
            detail="You cannot delete your own account"
        )

    employee_name = employee.employee_name
    employee_email = employee.employee_email
    employee_id_str = str(employee_id)

    # Clean up chatbot permissions where this employee is a reviewer
    # The FK has ondelete="SET NULL" so reviewer_id becomes NULL, we delete those orphaned records
    db.query(ChatbotPermission).filter(
        ChatbotPermission.reviewer_id == employee_id
    ).delete()
    
    # Clean up any references in can_review_users JSONB arrays
    all_permissions = db.query(ChatbotPermission).all()
    for perm in all_permissions:
        if perm.can_review_users and employee_id_str in perm.can_review_users:
            perm.can_review_users = [uid for uid in perm.can_review_users if uid != employee_id_str]
    
    # Clean up any references in ChatbotAccess allowed_users JSONB arrays
    all_access = db.query(ChatbotAccess).all()
    for access in all_access:
        if access.allowed_users and employee_id_str in access.allowed_users:
            access.allowed_users = [uid for uid in access.allowed_users if uid != employee_id_str]

    db.delete(employee)
    db.commit()

    return {
        "status": "success",
        "message": f"Employee '{employee_name}' deleted successfully",
        "deleted_employee": {
            "id": str(employee_id),
            "employee_name": employee_name,
            "employee_email": employee_email,
        },
        "deleted_by": {
            "id": str(admin.id),
            "name": admin.employee_name,
            "email": admin.employee_email,
        },
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
class UpdateEmployeeRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[str] = None
    department: Optional[str] = None
    status: Optional[str] = None
    meta_data: Optional[Dict[str, Any]] = None

# ...existing code...
@router.patch(
    "/update/{employee_id}",
    summary="Update employee profile",
)
async def update_employee_profile(
    employee_id: str,
    payload: UpdateEmployeeRequest = Body(...),
    db: Session = Depends(get_db),
):
    """
    Update employee profile details.
    Supports partial updates via JSON body.
    """

    log.info(f"Received update request for employee_id={employee_id}")

    try:
        employee_uuid = uuid.UUID(employee_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid employee_id UUID")

    employee = update_employee_service(
        db=db,
        employee_id=employee_uuid,
        name=payload.name,
        email=payload.email,
        role=payload.role,
        department=payload.department,
        status=payload.status,
        meta_data=payload.meta_data,
    )

    return {
        "employee_id": str(employee.id),
        "name": employee.employee_name,
        "email": employee.employee_email,
        "role": employee.employee_role,
        "status": (
            employee.meta_data.get("status")
            if isinstance(employee.meta_data, dict)
            else None
        ),
        "message": "Employee profile updated successfully",
    }


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


# ===================== EXCEL EXPORT =====================

_EXPORT_COLUMNS = ["employee_name", "employee_email", "employee_role", "department", "employee_code"]
_ALLOWED_ROLES  = ["employee", "admin", "manager", "hr"]


@router.get("/export", summary="Export all employees as Excel")
def export_employees(db: Session = Depends(get_db)):
    """Download an Excel file with all employees. Use it as an import template too."""
    employees = db.query(Employee).order_by(Employee.employee_name.asc()).all()

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Employees"
    ws.append(_EXPORT_COLUMNS)

    for emp in employees:
        ws.append([
            emp.employee_name,
            emp.employee_email or "",
            emp.employee_role,
            emp.department or "",
            emp.employee_code or "",
        ])

    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=employees.xlsx"},
    )


# ===================== EXCEL IMPORT =====================

@router.post("/import", summary="Bulk-import employees from Excel")
async def import_employees(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload an .xlsx file to bulk-create or update employees.

    Expected columns (row 1 = header):
      employee_name | employee_email | employee_role | department | employee_code

    Upsert logic (keyed on employee_email):
    - Email not in DB  → create new employee (role defaults to 'employee' if blank).
    - Email already in DB → update name, role, department, employee_code (email never changes).
    - Rows without employee_name are skipped.

    Response: { created, updated, skipped_rows }
    """
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only .xlsx / .xls files are accepted")

    file_bytes = await file.read()

    try:
        wb = openpyxl.load_workbook(filename=BytesIO(file_bytes), read_only=True, data_only=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse the uploaded file as Excel")

    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return {"created": 0, "updated": 0, "skipped_rows": []}

    # Normalise header
    header = [str(c).strip().lower() if c is not None else "" for c in rows[0]]
    required_cols = {"employee_name", "employee_email"}
    missing = required_cols - set(header)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    def col(name):
        return header.index(name)

    has_role  = "employee_role"  in header
    has_dept  = "department"     in header
    has_code  = "employee_code"  in header

    created_count = 0
    updated_count = 0
    skipped_rows  = []

    for row_idx, row in enumerate(rows[1:], start=2):
        name  = _cell(row, col("employee_name"))
        email = _cell(row, col("employee_email"))

        if not name:
            skipped_rows.append({"row": row_idx, "reason": "employee_name is empty"})
            continue

        role = _cell(row, col("employee_role")) if has_role else None
        if role and role not in _ALLOWED_ROLES:
            skipped_rows.append({"row": row_idx, "reason": f"invalid role '{role}'"})
            continue
        if not role:
            role = "employee"

        dept = _cell(row, col("department")) if has_dept else None
        code = _cell(row, col("employee_code")) if has_code else None

        if email:
            existing = db.query(Employee).filter(Employee.employee_email == email).first()
        else:
            existing = None

        if existing:
            existing.employee_name = name
            existing.employee_role = role
            if dept is not None:
                existing.department = dept
            if code is not None:
                existing.employee_code = code
            updated_count += 1
        else:
            new_emp = Employee(
                employee_id=str(uuid.uuid4()),  # placeholder until first SSO login
                employee_name=name,
                employee_email=email or None,
                employee_role=role,
                department=dept,
                employee_code=code,
                meta_data={},
            )
            db.add(new_emp)
            created_count += 1

    db.commit()
    return {"created": created_count, "updated": updated_count, "skipped_rows": skipped_rows}


def _cell(row: tuple, idx: int) -> Optional[str]:
    """Return a stripped string from a row cell, or None if empty."""
    val = row[idx] if idx < len(row) else None
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None

