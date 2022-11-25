from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from service.api.exceptions import UnauthorizedError

bearer_scheme = HTTPBearer(auto_error=False)


async def call_http_bearer(
    bearer_token: HTTPAuthorizationCredentials = Security(bearer_scheme)
) -> str:
    if not bearer_token:
        raise UnauthorizedError(
            error_message="Missing bearer token",
        )
    return bearer_token.credentials


def check_token(expected: str, actual: str) -> None:
    if expected != actual:
        raise UnauthorizedError(
            error_message="Invalid token",
        )
