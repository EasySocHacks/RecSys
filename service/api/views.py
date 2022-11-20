from typing import List

from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

import service.api.exceptions as exc
from service.log import app_logger
from service.models import Error
from service.rec_models import modelByName


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "https://www.youtube.com/watch?v=xm3YgoEiEDc"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses={
        404: {
            "model": Error,
            "description": "Model or User Not Found"
        },
    },
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if model_name not in modelByName:
        raise exc.ModelNotFoundError(
            error_message=f"Model {model_name} is unknown"
        )

    if user_id > 10**9:
        raise exc.UserNotFoundError(
            error_message=f"User {user_id} not found"
        )

    reco = modelByName[model_name].recommend(user_id, request.app.state.k_recs)
    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
