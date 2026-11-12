"""
Team & Project Configuration Services
"""

from typing import Optional, List

from sqlalchemy.orm import Session
from fastapi import HTTPException

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
