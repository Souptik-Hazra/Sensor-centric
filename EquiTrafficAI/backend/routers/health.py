"""Model health API routes."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/health", tags=["Neural Forecasting & GWNet"])


@router.get("/models")
def get_model_health(request: Request):
    """Report readiness of loaded forecasting checkpoints."""
    return {
        dataset: {
            "num_nodes": adapter.num_nodes,
            "checkpoint_loaded": getattr(adapter, "checkpoint_loaded", False),
            "error": getattr(adapter, "checkpoint_error", None),
            "device": str(adapter.device),
        }
        for dataset, adapter in request.app.state.gwnet_adapters.items()
    }
