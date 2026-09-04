"""LLM copilot API routes."""

from fastapi import APIRouter, Request

from ..schemas import LLMQueryRequest

router = APIRouter(prefix="/api/llm", tags=["AI Copilot"])


@router.post("/reasoning", response_description="Gemini LLM Causal Reroute Copilot Analysis")
def llm_reasoning(request: Request, query: LLMQueryRequest):
    """Generate natural-language traffic reasoning for a validated query."""
    return request.app.state.chatbot_service.respond(
        query,
        request.app.state.state_data,
        request.app.state.route_planner,
    )
