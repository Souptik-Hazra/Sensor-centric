"""Telemetry and topology API routes."""

from fastapi import APIRouter, Query, Request

router = APIRouter(prefix="/api", tags=["Telemetry & State"])


@router.get(
    "/state",
    response_description="Complete sensor metadata array and directed spatial edge geometry",
)
def get_state(
    request: Request,
    city: str = Query(
        "la",
        pattern="^(la|sd|pems04|pems08|pems_bay|pems03|pems07)$",
        description="Target city/corridor",
    ),
):
    """Return the loaded sensor metadata and topology for a city."""
    return request.app.state.state_data.get(city.lower(), request.app.state.state_data["la"])
