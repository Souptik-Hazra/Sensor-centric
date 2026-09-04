"""Routing API routes."""

from fastapi import APIRouter, Request
import numpy as np

from ..services.routing_service import plan_smart_route
from ..schemas import RerouteRequest, RouteRequest

router = APIRouter(tags=["Routing & Navigation"])


@router.post(
    "/api/route/plan",
    response_description="Smart Origin-Destination Route & Alternate Paths",
)
def plan_route(request: Request, route_request: RouteRequest):
    """Compute a route using the loaded traffic graph."""
    return plan_smart_route(
        route_request,
        request.app.state.state_data,
        request.app.state.route_graph_data,
    )


@router.post("/reroute", response_description="Reroute Advisory Report")
def reroute(request: Request, reroute_request: RerouteRequest):
    """Generate a bottleneck rerouting advisory for a corridor sensor."""
    node_id = str(reroute_request.target_node_id)
    location_map = request.app.state.la_location_map
    corridor_name = location_map.get(
        node_id,
        {},
    ).get("location_label", f"Freeway Corridor Node #{node_id}")
    speeds = np.asarray(reroute_request.predicted_speeds)
    minimum_speed = float(np.min(speeds))
    average_speed = float(np.mean(speeds))
    return {
        "node_report": {
            "queried_sensor": node_id,
            "corridor": corridor_name,
            "min_predicted_speed_mph": round(minimum_speed, 2),
            "average_predicted_speed_mph": round(average_speed, 2),
            "severe_congestion_detected": minimum_speed < 25.0,
            "horizon_minutes": 15,
        },
        "smart_copilot_advisory": (
            f"[EquiTraffic-GPT Advisory] Severe bottleneck on {corridor_name} "
            f"({minimum_speed:.1f} mph). Rerouting recommended."
        ),
    }
