"""HTTP routes for service health and message handling."""

from fastapi import APIRouter

from app.core.assistant import handle_message
from app.core.models import HealthResponse, MessageRequest, MessageResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/messages/handle", response_model=MessageResponse)
async def handle_message_route(payload: MessageRequest) -> MessageResponse:
    return await handle_message(payload)
