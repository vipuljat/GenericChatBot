

from fastapi import HTTPException, Request
from requests import Session

from services.auth_service import MicrosoftAuthService
from stateful_services.db_schema import Employee


def get_current_employee_from_token(request: Request, db: Session) -> Employee:
    """
    Extract and verify current employee from auth token.
    TEMP: Auth can be disabled via DISABLE_EMPLOYEE_AUTH=true
    """
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Not authenticated")

        token = auth_header.replace("Bearer ", "")

        
        auth_service = MicrosoftAuthService()
        payload = auth_service.verify_jwt_token(token)

        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        email = payload.get("email")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        employee = db.query(Employee).filter(
            Employee.employee_email == email
        ).first()

        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")

        return employee

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Unauthorized: {str(e)}"
        )