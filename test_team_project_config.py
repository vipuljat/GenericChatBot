"""
Manual HTTP test script for Team & Project Config endpoints.

Employees used:
  - Mansij Gupta  : id = 1f1dcab2-6a98-431a-a1a0-29e3ba40c1bf
  - COE_Chetna    : id = ff9bc0f5-4435-4de4-918e-6b44578cec63

Run with:
    python test_team_project_config.py
"""

import json
import requests

BASE = "http://localhost:8000"
CONFIG = f"{BASE}/config"
EMPLOYEES_URL = f"{BASE}/api/people-analyzer/employees"

EMP_MANSIJ = "1f1dcab2-6a98-431a-a1a0-29e3ba40c1bf"
EMP_CHETNA  = "ff9bc0f5-4435-4de4-918e-6b44578cec63"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def pretty(data):
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------
# 1. CREATE team and project
# ---------------------------------------------------------------
section("1. Create team & project")

r = requests.post(f"{CONFIG}/team-projects", json={"name": "Backend Team", "type": "team"})
check("POST /config/team-projects (team) → 201", r.status_code == 201, str(r.status_code))
team = r.json()
TEAM_ID = team.get("id")
print(f"     team id: {TEAM_ID}")

r = requests.post(f"{CONFIG}/team-projects", json={"name": "Alpha Project", "type": "project"})
check("POST /config/team-projects (project) → 201", r.status_code == 201, str(r.status_code))
project = r.json()
PROJECT_ID = project.get("id")
print(f"     project id: {PROJECT_ID}")

r = requests.post(f"{CONFIG}/team-projects", json={"name": "Bad Type", "type": "invalid"})
check("POST /config/team-projects (bad type) → 400", r.status_code == 400, str(r.status_code))


# ---------------------------------------------------------------
# 2. LIST team-projects
# ---------------------------------------------------------------
section("2. List team-projects")

r = requests.get(f"{CONFIG}/team-projects")
check("GET /config/team-projects → 200", r.status_code == 200, str(r.status_code))
all_items = r.json()
check("List contains created team", any(i["id"] == TEAM_ID for i in all_items))
check("List contains created project", any(i["id"] == PROJECT_ID for i in all_items))

r = requests.get(f"{CONFIG}/team-projects?type=team")
check("GET /config/team-projects?type=team → only teams", all(i["type"] == "team" for i in r.json()))

r = requests.get(f"{CONFIG}/team-projects?type=project")
check("GET /config/team-projects?type=project → only projects", all(i["type"] == "project" for i in r.json()))


# ---------------------------------------------------------------
# 3. UPDATE (PATCH) a team-project
# ---------------------------------------------------------------
section("3. Update team-project")

r = requests.patch(f"{CONFIG}/team-projects/{TEAM_ID}", json={"name": "Backend Team (Renamed)"})
check("PATCH /config/team-projects/{id} → 200", r.status_code == 200, str(r.status_code))
check("Name updated", r.json().get("name") == "Backend Team (Renamed)")

r = requests.patch(f"{CONFIG}/team-projects/00000000-0000-0000-0000-000000000000", json={"name": "X"})
check("PATCH non-existent id → 404", r.status_code == 404, str(r.status_code))


# ---------------------------------------------------------------
# 4. BULK ASSIGN employees
# ---------------------------------------------------------------
section("4. Bulk assign employees")

r = requests.post(f"{CONFIG}/assignments/bulk", json={
    "assignments": [
        {"employee_id": EMP_MANSIJ, "team_project_ids": [TEAM_ID, PROJECT_ID]},
        {"employee_id": EMP_CHETNA,  "team_project_ids": [TEAM_ID]},
    ]
})
check("POST /config/assignments/bulk → 200", r.status_code == 200, str(r.status_code))
result = r.json()
check("updated == 2", result.get("updated") == 2, str(result))
check("no skipped_ids", result.get("skipped_ids") == [], str(result.get("skipped_ids")))


# ---------------------------------------------------------------
# 5. GET assignments
# ---------------------------------------------------------------
section("5. Get assignments")

r = requests.get(f"{CONFIG}/assignments")
check("GET /config/assignments → 200", r.status_code == 200, str(r.status_code))
all_assignments = r.json()
mansij_rows = [a for a in all_assignments if a["employee_id"] == EMP_MANSIJ]
chetna_rows  = [a for a in all_assignments if a["employee_id"] == EMP_CHETNA]
check("Mansij has 2 assignments", len(mansij_rows) == 2, str(len(mansij_rows)))
check("Chetna has 1 assignment",  len(chetna_rows)  == 1, str(len(chetna_rows)))

r = requests.get(f"{CONFIG}/assignments?employee_id={EMP_MANSIJ}")
check("GET /config/assignments?employee_id= filter works", all(a["employee_id"] == EMP_MANSIJ for a in r.json()))


# ---------------------------------------------------------------
# 6. Bulk assign with an invalid team_project_id
# ---------------------------------------------------------------
section("6. Bulk assign with invalid ID (skipped)")

FAKE_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
r = requests.post(f"{CONFIG}/assignments/bulk", json={
    "assignments": [
        {"employee_id": EMP_CHETNA, "team_project_ids": [TEAM_ID, FAKE_ID]},
    ]
})
check("POST bulk with fake id → 200", r.status_code == 200, str(r.status_code))
result = r.json()
check("skipped_ids contains fake id", FAKE_ID in result.get("skipped_ids", []), str(result))

# Chetna should still have the valid team assignment
r = requests.get(f"{CONFIG}/assignments?employee_id={EMP_CHETNA}")
chetna_rows = r.json()
check("Chetna still has 1 valid assignment after partial bulk", len(chetna_rows) == 1, str(len(chetna_rows)))


# ---------------------------------------------------------------
# 7. Bulk assign with empty list clears employee mappings
# ---------------------------------------------------------------
section("7. Clear assignments for Chetna")

r = requests.post(f"{CONFIG}/assignments/bulk", json={
    "assignments": [
        {"employee_id": EMP_CHETNA, "team_project_ids": []},
    ]
})
check("POST bulk with [] → 200", r.status_code == 200, str(r.status_code))

r = requests.get(f"{CONFIG}/assignments?employee_id={EMP_CHETNA}")
check("Chetna now has 0 assignments", len(r.json()) == 0, str(len(r.json())))

# Mansij should be unaffected
r = requests.get(f"{CONFIG}/assignments?employee_id={EMP_MANSIJ}")
check("Mansij still has 2 assignments (unaffected)", len(r.json()) == 2, str(len(r.json())))


# ---------------------------------------------------------------
# 8. GET /api/people-analyzer/employees includes teams & projects
# ---------------------------------------------------------------
section("8. GET /api/people-analyzer/employees — teams & projects attached")

r = requests.get(EMPLOYEES_URL)
check("GET /api/people-analyzer/employees → 200", r.status_code == 200, str(r.status_code))

employees = r.json().get("employees", [])
mansij = next((e for e in employees if e["id"] == EMP_MANSIJ), None)
chetna  = next((e for e in employees if e["id"] == EMP_CHETNA), None)

if mansij:
    check("Mansij has 'teams' key",    "teams"    in mansij)
    check("Mansij has 'projects' key", "projects" in mansij)
    check("Mansij teams not empty",    len(mansij.get("teams", [])) > 0,    str(mansij.get("teams")))
    check("Mansij projects not empty", len(mansij.get("projects", [])) > 0, str(mansij.get("projects")))
else:
    print(f"  [{FAIL}] Mansij not found in response")

if chetna:
    check("Chetna has 'teams' key",    "teams"    in chetna)
    check("Chetna has 'projects' key", "projects" in chetna)
    check("Chetna teams == []",    chetna.get("teams")    == [], str(chetna.get("teams")))
    check("Chetna projects == []", chetna.get("projects") == [], str(chetna.get("projects")))
else:
    print(f"  [{FAIL}] Chetna not found in response")


# ---------------------------------------------------------------
# 9. DELETE a team-project (cascade removes its mappings)
# ---------------------------------------------------------------
section("9. DELETE team-project and verify cascade")

r = requests.delete(f"{CONFIG}/team-projects/{TEAM_ID}")
check("DELETE /config/team-projects/{id} → 204", r.status_code == 204, str(r.status_code))

r = requests.get(f"{CONFIG}/assignments?employee_id={EMP_MANSIJ}")
mansij_rows = r.json()
team_mapping_ids = [a["team_project_id"] for a in mansij_rows]
check("Deleted team no longer in Mansij assignments", TEAM_ID not in team_mapping_ids, str(team_mapping_ids))

r = requests.delete(f"{CONFIG}/team-projects/00000000-0000-0000-0000-000000000000")
check("DELETE non-existent id → 404", r.status_code == 404, str(r.status_code))


# ---------------------------------------------------------------
# Cleanup: delete remaining project
# ---------------------------------------------------------------
section("Cleanup")
r = requests.delete(f"{CONFIG}/team-projects/{PROJECT_ID}")
check(f"Cleanup: deleted project {PROJECT_ID}", r.status_code == 204, str(r.status_code))

print("\nDone.\n")
