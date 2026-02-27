"""
Team & Project Configuration Routes
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from stateful_services.database import get_db
from services.team_project_service import (
    list_team_projects,
    create_team_project,
    update_team_project,
    delete_team_project,
    get_assignments,
    bulk_assign,
    export_assignments,
    import_assignments,
)

router = APIRouter(tags=["Team & Project Config"])


# ---- Request Bodies ----

class TeamProjectBody(BaseModel):
    name: str
    type: str  # 'team' | 'project'


class TeamProjectPatchBody(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None


class AssignmentItem(BaseModel):
    employee_id: str
    team_project_ids: List[str]


class BulkAssignBody(BaseModel):
    assignments: List[AssignmentItem]


# ---- Team/Project CRUD ----

@router.get("/team-projects")
def list_team_projects_route(
    type: Optional[str] = Query(None, description="Filter by 'team' or 'project'"),
    db: Session = Depends(get_db),
):
    return list_team_projects(db, type_filter=type)


@router.post("/team-projects", status_code=201)
def create_team_project_route(body: TeamProjectBody, db: Session = Depends(get_db)):
    return create_team_project(db, name=body.name, type=body.type)


@router.patch("/team-projects/{id}")
def update_team_project_route(id: str, body: TeamProjectPatchBody, db: Session = Depends(get_db)):
    return update_team_project(db, id=id, name=body.name, type=body.type)


@router.delete("/team-projects/{id}", status_code=204)
def delete_team_project_route(id: str, db: Session = Depends(get_db)):
    delete_team_project(db, id=id)


# ---- Assignments ----

@router.post("/assignments/bulk")
def bulk_assign_route(body: BulkAssignBody, db: Session = Depends(get_db)):
    assignments = [item.model_dump() for item in body.assignments]
    return bulk_assign(db, assignments=assignments)


@router.get("/assignments")
def get_assignments_route(
    employee_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    return get_assignments(db, employee_id=employee_id)


# ---- Excel Import / Export ----

@router.get("/assignments/export")
def export_assignments_route(db: Session = Depends(get_db)):
    buffer = export_assignments(db)
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=team_assignments.xlsx"},
    )


@router.post("/assignments/import")
async def import_assignments_route(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only .xlsx / .xls files are accepted")
    file_bytes = await file.read()
    return import_assignments(db, file_bytes)
