"""
seed_departments_as_teams.py
────────────────────────────
One-time (idempotent) script that:
  1. Reads every unique non-null department from the employees table.
  2. Creates a TeamProject row (type='team') for each department that
     doesn't already exist in the teams_projects table.
  3. Associates every employee to their department's team via
     EmployeeTeamProjectMapping — skipping rows that already exist.

Safe to re-run; it will not create duplicates.

Usage:
    python seed_departments_as_teams.py
"""

import sys

from stateful_services.database import db_manager
from stateful_services.db_schema import Employee, TeamProject, EmployeeTeamProjectMapping


def run():
    with db_manager.session() as db:
        # ── 1. Load all employees that have a department ──────────────────
        employees = (
            db.query(Employee)
            .filter(Employee.department.isnot(None), Employee.department != "")
            .all()
        )

        if not employees:
            print("No employees with a department found. Nothing to do.")
            return

        # ── 2. Build a set of unique department names ─────────────────────
        unique_departments = {emp.department.strip() for emp in employees if emp.department and emp.department.strip()}
        print(f"Found {len(unique_departments)} unique department(s): {sorted(unique_departments)}")

        # ── 3. Load or create a TeamProject row for each department ───────
        dept_to_team: dict = {}  # department name -> TeamProject.id (str)

        for dept in sorted(unique_departments):
            existing = (
                db.query(TeamProject)
                .filter(TeamProject.name == dept, TeamProject.type == "team")
                .first()
            )
            if existing:
                dept_to_team[dept] = str(existing.id)
                print(f"  [skip]   team already exists → '{dept}' (id={existing.id})")
            else:
                team = TeamProject(name=dept, type="team")
                db.add(team)
                db.flush()  # get the generated id before commit
                dept_to_team[dept] = str(team.id)
                print(f"  [create] team created        → '{dept}' (id={team.id})")

        # ── 4. Load existing mappings to avoid duplicates ─────────────────
        all_tp_ids = list(dept_to_team.values())
        existing_mappings: set = set()
        if all_tp_ids:
            rows = (
                db.query(
                    EmployeeTeamProjectMapping.employee_id,
                    EmployeeTeamProjectMapping.team_project_id,
                )
                .filter(EmployeeTeamProjectMapping.team_project_id.in_(all_tp_ids))
                .all()
            )
            existing_mappings = {(str(r.employee_id), str(r.team_project_id)) for r in rows}

        # ── 5. Create missing employee → team mappings ────────────────────
        created = 0
        skipped = 0

        for emp in employees:
            dept = emp.department.strip() if emp.department else None
            if not dept or dept not in dept_to_team:
                continue

            team_id = dept_to_team[dept]
            key = (str(emp.id), team_id)

            if key in existing_mappings:
                skipped += 1
                continue

            mapping = EmployeeTeamProjectMapping(
                employee_id=emp.id,
                team_project_id=team_id,
            )
            db.add(mapping)
            existing_mappings.add(key)  # prevent duplicates within this run
            created += 1

        print(f"\nMappings — created: {created}, already existed (skipped): {skipped}")
        print("Done. All changes committed.")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
