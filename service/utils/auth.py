from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette import status

bearer_scheme = HTTPBearer(auto_error=False)


async def call_http_bearer(
    bearer_token: HTTPAuthorizationCredentials = Security(bearer_scheme)
) -> str:
    if not bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )
    return bearer_token.credentials


def check_token(expected: str, actual: str) -> None:
    if expected != actual:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
