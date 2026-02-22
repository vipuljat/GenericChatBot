"""
Team & Project Configuration Services
"""

from io import BytesIO
from typing import Optional, List

from sqlalchemy.orm import Session
from fastapi import HTTPException
import openpyxl

from stateful_services.db_schema import TeamProject, EmployeeTeamProjectMapping, Employee
from utils.logging import log


def list_team_projects(db: Session, type_filter: Optional[str] = None) -> list:
    query = db.query(TeamProject)
    if type_filter:
        query = query.filter(TeamProject.type == type_filter)
    items = query.order_by(TeamProject.created_at.asc()).all()
    return [
        {
            "id": str(tp.id),
            "name": tp.name,
            "type": tp.type,
            "created_at": tp.created_at,
            "updated_at": tp.updated_at,
        }
        for tp in items
    ]


def create_team_project(db: Session, name: str, type: str) -> dict:
    if type not in ("team", "project"):
        raise HTTPException(status_code=400, detail="type must be 'team' or 'project'")
    tp = TeamProject(name=name, type=type)
    db.add(tp)
    db.commit()
    db.refresh(tp)
    return {
        "id": str(tp.id),
        "name": tp.name,
        "type": tp.type,
        "created_at": tp.created_at,
        "updated_at": tp.updated_at,
    }


def update_team_project(db: Session, id: str, name: Optional[str], type: Optional[str]) -> dict:
    tp = db.query(TeamProject).filter(TeamProject.id == id).first()
    if not tp:
        raise HTTPException(status_code=404, detail="Team/Project not found")
    if type is not None and type not in ("team", "project"):
        raise HTTPException(status_code=400, detail="type must be 'team' or 'project'")
    if name is not None:
        tp.name = name
    if type is not None:
        tp.type = type
    db.commit()
    db.refresh(tp)
    return {
        "id": str(tp.id),
        "name": tp.name,
        "type": tp.type,
        "created_at": tp.created_at,
        "updated_at": tp.updated_at,
    }


def delete_team_project(db: Session, id: str) -> bool:
    tp = db.query(TeamProject).filter(TeamProject.id == id).first()
    if not tp:
        raise HTTPException(status_code=404, detail="Team/Project not found")
    db.delete(tp)
    db.commit()
    return True


def get_assignments(db: Session, employee_id: Optional[str] = None) -> list:
    query = (
        db.query(EmployeeTeamProjectMapping, TeamProject)
        .join(TeamProject, TeamProject.id == EmployeeTeamProjectMapping.team_project_id)
    )
    if employee_id:
        query = query.filter(EmployeeTeamProjectMapping.employee_id == employee_id)

    rows = query.all()
    return [
        {
            "mapping_id": str(mapping.id),
            "employee_id": str(mapping.employee_id),
            "team_project_id": str(tp.id),
            "team_project_name": tp.name,
            "team_project_type": tp.type,
            "created_at": mapping.created_at,
        }
        for mapping, tp in rows
    ]


def bulk_assign(db: Session, assignments: list) -> dict:
    """
    Replace assignments for each listed employee.
    assignments: list of {"employee_id": str, "team_project_ids": [str, ...]}
    Only employees in the payload are affected.
    """
    # Collect all referenced team_project_ids for validation
    all_tp_ids = set()
    for item in assignments:
        for tp_id in item.get("team_project_ids", []):
            all_tp_ids.add(tp_id)

    # Validate which IDs actually exist
    valid_tps = set()
    if all_tp_ids:
        existing = db.query(TeamProject.id).filter(
            TeamProject.id.in_(list(all_tp_ids))
        ).all()
        valid_tps = {str(row.id) for row in existing}

    skipped_ids = list(all_tp_ids - valid_tps)
    if skipped_ids:
        log.warning(f"bulk_assign: skipping unknown team_project_ids: {skipped_ids}")

    updated_count = 0
    for item in assignments:
        emp_id = item.get("employee_id")
        tp_ids = [tid for tid in item.get("team_project_ids", []) if tid in valid_tps]

        # Delete existing mappings for this employee
        db.query(EmployeeTeamProjectMapping).filter(
            EmployeeTeamProjectMapping.employee_id == emp_id
        ).delete(synchronize_session=False)

        # Insert new mappings
        for tp_id in tp_ids:
            mapping = EmployeeTeamProjectMapping(
                employee_id=emp_id,
                team_project_id=tp_id,
            )
            db.add(mapping)

        updated_count += 1

    db.commit()
    return {"updated": updated_count, "skipped_ids": skipped_ids}


def export_assignments(db: Session) -> BytesIO:
    """
    Build an Excel workbook with columns:
      employee_id | employee_name | teams | projects
    Each row is one employee; teams/projects are comma-separated names.
    Returns a BytesIO object ready for streaming.
    """
    # Load all employees
    employees = db.query(Employee).order_by(Employee.employee_name.asc()).all()

    # Load all mappings joined with team/project info in one query
    rows = (
        db.query(EmployeeTeamProjectMapping, TeamProject)
        .join(TeamProject, TeamProject.id == EmployeeTeamProjectMapping.team_project_id)
        .all()
    )

    # Build a lookup: employee UUID -> {teams: [...], projects: [...]}
    mapping_lookup: dict = {}
    for mapping, tp in rows:
        emp_key = str(mapping.employee_id)
        if emp_key not in mapping_lookup:
            mapping_lookup[emp_key] = {"teams": [], "projects": []}
        if tp.type == "team":
            mapping_lookup[emp_key]["teams"].append(tp.name)
        else:
            mapping_lookup[emp_key]["projects"].append(tp.name)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Assignments"
    ws.append(["employee_id", "employee_name", "teams", "projects"])

    for emp in employees:
        emp_uuid = str(emp.id)
        emp_data = mapping_lookup.get(emp_uuid, {"teams": [], "projects": []})
        ws.append([
            emp_uuid,
            emp.employee_name,
            ", ".join(emp_data["teams"]),
            ", ".join(emp_data["projects"]),
        ])

    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer


def import_assignments(db: Session, file_bytes: bytes) -> dict:
    """
    Parse an Excel file and replace assignments for each listed employee.
    Expected columns (row 1 = header): employee_id, employee_name, teams, projects

    - employee_id: UUID primary key of the employee (employees.id)
    - teams: comma-separated team names (empty string = clear all teams)
    - projects: comma-separated project names (empty string = clear all projects)
    - A missing/None cell means "no change" for that dimension.

    Returns: { updated, skipped_rows, unresolved_names }
    """
    wb = openpyxl.load_workbook(filename=BytesIO(file_bytes), read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return {"updated": 0, "skipped_rows": [], "unresolved_names": []}

    # Normalise header
    header = [str(c).strip().lower() if c is not None else "" for c in rows[0]]
    try:
        col_emp_id   = header.index("employee_id")
        col_teams    = header.index("teams")
        col_projects = header.index("projects")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Missing required column: {e}")

    # Pre-load name -> id maps for teams and projects
    all_tps = db.query(TeamProject).all()
    team_name_to_id    = {tp.name: str(tp.id) for tp in all_tps if tp.type == "team"}
    project_name_to_id = {tp.name: str(tp.id) for tp in all_tps if tp.type == "project"}

    # Pre-load valid employee UUIDs
    valid_employee_ids = {str(e.id) for e in db.query(Employee.id).all()}

    skipped_rows = []
    unresolved_names = set()
    assignments_to_save = []

    for row_idx, row in enumerate(rows[1:], start=2):
        raw_emp_id = row[col_emp_id]
        if raw_emp_id is None:
            skipped_rows.append({"row": row_idx, "reason": "employee_id is empty"})
            continue

        emp_id = str(raw_emp_id).strip()
        if emp_id not in valid_employee_ids:
            skipped_rows.append({"row": row_idx, "reason": f"employee_id '{emp_id}' not found"})
            continue

        # Resolve teams
        raw_teams = row[col_teams]
        team_ids = _resolve_names(raw_teams, team_name_to_id, unresolved_names)

        # Resolve projects
        raw_projects = row[col_projects]
        project_ids = _resolve_names(raw_projects, project_name_to_id, unresolved_names)

        assignments_to_save.append({
            "employee_id": emp_id,
            "team_project_ids": team_ids + project_ids,
        })

    if not assignments_to_save:
        return {"updated": 0, "skipped_rows": skipped_rows, "unresolved_names": list(unresolved_names)}

    result = bulk_assign(db, assignments_to_save)
    return {
        "updated": result["updated"],
        "skipped_rows": skipped_rows,
        "unresolved_names": list(unresolved_names),
    }


def _resolve_names(raw_cell, name_to_id: dict, unresolved_set: set) -> list:
    """
    Parse a comma-separated cell value into a list of IDs.
    None cell  → empty list (treat as "no entries provided, clear")
    Empty str  → empty list (explicit clear)
    Otherwise  → resolve each name; unknowns go into unresolved_set.
    """
    if raw_cell is None:
        return []
    names = [n.strip() for n in str(raw_cell).split(",") if n.strip()]
    ids = []
    for name in names:
        if name in name_to_id:
            ids.append(name_to_id[name])
        else:
            unresolved_set.add(name)
    return ids
