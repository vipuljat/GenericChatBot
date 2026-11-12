"""
Shared Excel import/export for employees and their team/project assignments.

Used by both Manage Users (/employees/export, /employees/import) and
Team Config (/config/assignments/export, /config/assignments/import), so both
pages read and write the same file format with the same result.

Columns (row 1 = header, names are case-insensitive):
    employee_email | employee_name | employee_role | department | employee_code | teams | projects

Rules:
- employee_email is required. employee_name is required on Manage Users import.
- Manage Users (create_missing=True): existing employee (matched by email, case-insensitive)
  is updated, a soft-deleted one with that email is reactivated, and unknown emails create
  a new employee. Blank role / department / code keep the current value.
- Team Config (create_missing=False): only teams/projects of existing, non-deleted employees
  change. Unknown or deleted emails are skipped. No profile field is touched.
- teams / projects are comma-separated names.
    * Column present, cell blank  -> clear that dimension for the employee.
    * Column absent               -> assignments are left unchanged.
- Unknown team/project names are reported in unresolved_names and skipped.
"""

import uuid
from collections import defaultdict
from io import BytesIO

import openpyxl
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from stateful_services.db_schema import Employee, TeamProject, EmployeeTeamProjectMapping
from utils.logging import log


ALLOWED_ROLES = ["employee", "admin", "manager", "hr"]
HEADERS = [
    "employee_email",
    "employee_name",
    "employee_role",
    "department",
    "employee_code",
    "teams",
    "projects",
]
REQUIRED_HEADERS = ("employee_email", "employee_name")


# ===================== EXPORT =====================

def export_employees_workbook(db: Session) -> BytesIO:
    """Export all non-deleted employees with their team and project names."""
    employees = (
        db.query(Employee)
        .filter(Employee.status != "deleted")
        .order_by(Employee.employee_name.asc())
        .all()
    )

    teams_by_emp = defaultdict(list)
    projects_by_emp = defaultdict(list)
    mapping_rows = (
        db.query(EmployeeTeamProjectMapping.employee_id, TeamProject.type, TeamProject.name)
        .join(TeamProject, TeamProject.id == EmployeeTeamProjectMapping.team_project_id)
        .all()
    )
    for emp_id, tp_type, tp_name in mapping_rows:
        target = teams_by_emp if tp_type == "team" else projects_by_emp
        target[str(emp_id)].append(tp_name)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Employees"
    ws.append(HEADERS)
    for emp in employees:
        key = str(emp.id)
        ws.append([
            emp.employee_email or "",
            emp.employee_name,
            emp.employee_role or "",
            emp.department or "",
            emp.employee_code or "",
            ", ".join(sorted(teams_by_emp[key])),
            ", ".join(sorted(projects_by_emp[key])),
        ])

    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer


# ===================== IMPORT =====================

def import_employees_workbook(
    db: Session,
    file_bytes: bytes,
    create_missing: bool,
    deactivate_missing: bool = False,
    acting_employee_id=None,
) -> dict:
    """
    Update employees and apply team/project assignments from the shared format.

    create_missing=True  (Manage Users): unknown emails create a new employee;
                         soft-deleted employees with a matching email are reactivated.
    create_missing=False (Team Config):  unknown or soft-deleted emails are skipped;
                         no employee is ever created or reactivated.

    deactivate_missing=True (Manage Users only): active employees whose email is NOT in
                         the file are soft-deleted (status="deleted"). Employees without an
                         email are left alone, and the acting admin is never deleted.
    """
    try:
        wb = openpyxl.load_workbook(filename=BytesIO(file_bytes), read_only=True, data_only=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse the uploaded file as Excel")

    rows = list(wb.active.iter_rows(values_only=True))
    if not rows:
        return _result(0, 0, [], [])

    header = [str(c).strip().lower() if c is not None else "" for c in rows[0]]
    missing = [h for h in REQUIRED_HEADERS if h not in header]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")
    col = {name: header.index(name) for name in HEADERS if name in header}

    all_tps = db.query(TeamProject).all()
    team_lookup = {tp.name: str(tp.id) for tp in all_tps if tp.type == "team"}
    project_lookup = {tp.name: str(tp.id) for tp in all_tps if tp.type == "project"}
    team_type_ids = list(team_lookup.values())
    project_type_ids = list(project_lookup.values())

    created = updated = deactivated = 0
    skipped_rows = []
    unresolved = set()
    emails_in_file = set()

    try:
        for row_no, row in enumerate(rows[1:], start=2):
            email = _cell(row, col, "employee_email")
            name = _cell(row, col, "employee_name")
            if email:
                emails_in_file.add(email.lower())
            if not email:
                skipped_rows.append({"row": row_no, "reason": "employee_email is empty"})
                continue
            # Name and role are only needed when a profile is created or updated (Manage Users)
            if create_missing and not name:
                skipped_rows.append({"row": row_no, "reason": "employee_name is empty"})
                continue

            role = _cell(row, col, "employee_role")
            if create_missing and role and role not in ALLOWED_ROLES:
                skipped_rows.append({"row": row_no, "reason": f"invalid role '{role}'"})
                continue

            department = _cell(row, col, "department")
            code = _cell(row, col, "employee_code")

            lookup = db.query(Employee).filter(func.lower(Employee.employee_email) == email.lower())
            if not create_missing:
                lookup = lookup.filter(Employee.status != "deleted")
            emp = lookup.first()

            if emp is None and not create_missing:
                skipped_rows.append({"row": row_no, "reason": f"employee_email '{email}' not found"})
                continue

            if emp:
                if create_missing:
                    # Manage Users: the file is the source of truth for profile fields
                    emp.employee_name = name
                    if role:
                        emp.employee_role = role
                    if department:
                        emp.department = department
                    if code:
                        emp.employee_code = code
                    emp.status = "active"
                # Team Config: only teams/projects change (applied below)
                updated += 1
            else:
                emp = Employee(
                    employee_id=str(uuid.uuid4()),  # placeholder until first SSO login
                    employee_name=name,
                    employee_email=email,
                    employee_role=role or "employee",
                    department=department or None,
                    employee_code=code or None,
                    meta_data={},
                    status="active",
                )
                db.add(emp)
                db.flush()  # assigns emp.id for the mappings below
                created += 1

            if "teams" in col:
                ids = _resolve(_cell(row, col, "teams"), team_lookup, unresolved)
                _replace_mappings(db, emp.id, ids, team_type_ids)
            if "projects" in col:
                ids = _resolve(_cell(row, col, "projects"), project_lookup, unresolved)
                _replace_mappings(db, emp.id, ids, project_type_ids)

        if deactivate_missing:
            active = db.query(Employee).filter(Employee.status == "active").all()
            for emp in active:
                if emp.employee_email is None:
                    continue  # cannot be matched against the file; leave it alone
                if emp.employee_email.strip().lower() in emails_in_file:
                    continue
                if acting_employee_id is not None and emp.id == acting_employee_id:
                    continue  # never delete the admin running the import
                emp.status = "deleted"
                deactivated += 1

        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        log.error(f"Employee/assignment import failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Import failed")

    log.info(
        f"Employee import: created={created}, updated={updated}, "
        f"deactivated={deactivated}, skipped={len(skipped_rows)}"
    )
    result = _result(created, updated, skipped_rows, sorted(unresolved))
    result["deactivated"] = deactivated
    return result


# ===================== HELPERS =====================

def _result(created, updated, skipped_rows, unresolved_names) -> dict:
    return {
        "created": created,
        "updated": updated,
        "skipped_rows": skipped_rows,
        "unresolved_names": unresolved_names,
    }


def _cell(row: tuple, col: dict, name: str) -> str:
    """Stripped cell text, or "" when the column is absent or the cell is empty."""
    idx = col.get(name)
    if idx is None or idx >= len(row) or row[idx] is None:
        return ""
    text = str(row[idx])
    return " ".join(text.split()) if name == "employee_name" else text.strip()


def _resolve(raw: str, lookup: dict, unresolved: set) -> list:
    """Turn 'A, B' into [id_A, id_B]. Unknown names go into `unresolved`."""
    ids = []
    for name in (n.strip() for n in raw.split(",")):
        if not name:
            continue
        if name in lookup:
            ids.append(lookup[name])
        else:
            unresolved.add(name)
    return list(dict.fromkeys(ids))


def _replace_mappings(db: Session, employee_id, new_ids: list, type_ids: list) -> None:
    """Replace this employee's mappings of one type (team or project) with new_ids."""
    db.query(EmployeeTeamProjectMapping).filter(
        EmployeeTeamProjectMapping.employee_id == employee_id,
        EmployeeTeamProjectMapping.team_project_id.in_(type_ids),
    ).delete(synchronize_session=False)
    for tp_id in new_ids:
        db.add(EmployeeTeamProjectMapping(employee_id=employee_id, team_project_id=tp_id))
