from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from jose import jwt, JWTError
import config

# Paths that do NOT require authentication
PUBLIC_PATHS = {
    "/",
    "/health",
    "/auth/login",
    "/auth/callback",
    "/auth/logout",
}

# Path prefixes that do NOT require authentication.
# Using prefixes for docs so that Swagger sub-paths like
# /api/v1/docs/oauth2-redirect are also allowed through.
PUBLIC_PREFIXES = (
    "/api/v1/docs",
    "/api/v1/openapi.json",
    "/api/v1/redoc",
    "/auth/",
)


class JWTMiddleware(BaseHTTPMiddleware):
    """Global JWT authentication middleware.

    Every request to a non-public path must carry a valid
    'Authorization: Bearer <token>' header.  The decoded payload is
    attached to request.state.user so downstream handlers can access it
    without re-decoding.
    """

    async def dispatch(self, request: Request, call_next):
        # Allow OPTIONS (CORS pre-flight) through unconditionally
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path

        # Skip auth for public paths / prefixes
        if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
            return await call_next(request)

        # Extract token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"},
            )

        token = auth_header[len("Bearer "):]

        # Verify token
        try:
            payload = jwt.decode(
                token,
                config.JWT_SECRET_KEY,
                algorithms=[config.JWT_ALGORITHM],
            )
        except JWTError as exc:
            return JSONResponse(
                status_code=401,
                content={"detail": f"Invalid or expired token: {exc}"},
            )

        # Attach decoded payload to request state for downstream use
        request.state.user = payload

        return await call_next(request)
