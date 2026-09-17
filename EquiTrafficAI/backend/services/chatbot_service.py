"""Chatbot orchestration for the traffic API.

This module keeps request parsing, sensor resolution, route-query handling, and
traffic context construction outside the FastAPI application module. The
provider-specific model remains behind ``TrafficLLMEngine``.
"""

import re
from typing import Any, Callable
from .speed_utils import standardized_speed_to_mph


class TrafficChatbotService:
    """Build validated traffic context and delegate language generation."""

    ROUTE_KEYWORDS=(
        "shortest path", "best route", "route between", "path between",
        "path from", "route from", "navigate from", "navigate to",
        "direction from", "direction to", "how to go from", "show route",
        "show path", "find route", "find path", "shortest route",
    )
    DIRECT_NODE_PATTERN=re.compile(
        r"(?:node|sensor|#)?\s*\b(\d+)\b\s*(?:to|and|-|->)\s*"
        r"(?:node|sensor|#)?\s*\b(\d+)\b",
        re.IGNORECASE,
    )

    def __init__(self, llm_engine):
        self.llm_engine=llm_engine

    @staticmethod
    def _resolve_sensor(reference: str, sensors: list[dict[str, Any]]):
        numeric=int(reference)
        return next(
            (sensor for sensor in sensors
             if sensor.get("id") == numeric
             or str(sensor.get("sensor_id")) == str(reference)),
            None,
        )

    def _route_request(self, prompt: str, city: str, sensors: list[dict[str, Any]],
                       route_planner: Callable):
        prompt_lower=prompt.lower()
        direct_match=self.DIRECT_NODE_PATTERN.search(prompt_lower)
        if not any(keyword in prompt_lower for keyword in self.ROUTE_KEYWORDS) and not direct_match:
            return None, None, None

        if direct_match:
            references=[direct_match.group(1), direct_match.group(2)]
        else:
            references=re.findall(r"(?:node|sensor|#)\s*(\d+)", prompt_lower)
            if len(references) < 2:
                references=re.findall(r"\b(\d+)\b", prompt)
        if len(references) < 2:
            return None, None, None

        origin=self._resolve_sensor(references[0], sensors)
        destination=self._resolve_sensor(references[1], sensors)
        if not origin or not destination:
            return None, None, None

        class RouteQuery:
            pass

        route_request=RouteQuery()
        route_request.origin_id=origin.get("id")
        route_request.destination_id=destination.get("id")
        route_request.target_time="08:00 AM"
        route_request.city=city

        try:
            return route_planner(route_request), origin, destination
        except Exception as error:
            print(f"[LLM Route] Failed to compute route: {error}")
            return None, origin, destination

    def respond(self, req, state_data: dict[str, Any], route_planner: Callable):
        city_key=req.city.lower()
        city_data=state_data.get(city_key, state_data.get("la", {}))
        sensors=city_data.get("sensors", [])
        input_id=req.sensor_id

        route_result, route_origin, route_destination=self._route_request(
            req.prompt, city_key, sensors, route_planner
        )
        if route_origin and route_destination:
            input_id=route_origin.get("id")

        selected=next(
            (sensor for sensor in sensors if str(sensor.get("sensor_id")) == str(input_id)),
            None,
        ) or next(
            (sensor for sensor in sensors if sensor.get("id") == input_id),
            sensors[0] if sensors else {},
        )
        node_id=selected.get("id", input_id)
        real_sensor_id=selected.get("sensor_id", input_id)
        neighbors=state_data.get("graph_neighbors", {})
        downstream_ids=[]
        for neighbor_id in neighbors.get(node_id, [node_id + 1, node_id + 2])[:3]:
            neighbor=next((sensor for sensor in sensors if sensor.get("id") == neighbor_id), None)
            downstream_ids.append(neighbor.get("sensor_id", neighbor_id) if neighbor else neighbor_id)

        speed=float(selected.get("speed", 55.0))
        predicted_speed=max(10.0, speed - 5.0)
        status=selected.get("status", "HEALTHY")

        history=state_data.get("his_npz_sd" if city_key == "sd" else "his_npz_la")
        if history is not None and len(history) and 0 <= node_id < history.shape[1]:
            current_index=max(0, min(history.shape[0] - 1, int(req.step)))
            future_index=min(history.shape[0] - 1, current_index + 3)
            speed=standardized_speed_to_mph(history[current_index, node_id, 0], speed)
            predicted_speed=standardized_speed_to_mph(history[future_index, node_id, 0], speed)
            if speed < 30.0 or predicted_speed < 30.0:
                status="CONGESTED"

        if route_result and route_result.get("recommended_path_coords"):
            primary=route_result.get("primary_route", {})
            origin_label=route_result.get("origin", {}).get("label", f"Node #{route_origin.get('id')}")
            destination_label=route_result.get("destination", {}).get("label", f"Node #{route_destination.get('id')}")
            bottleneck=primary.get("bottleneck_detected", False)
            alternate=route_result.get("recommended_alternate_route", {})
            primary_time=float(primary.get("travel_time_minutes", 14))
            alternate_time=float(alternate.get("travel_time_minutes", primary_time))
            time_delta=round(primary_time - alternate_time, 1)
            time_delta_label=(
                f"Time Saved: {time_delta:.1f} mins"
                if time_delta > 0
                else f"Time Lost: {abs(time_delta):.1f} mins"
                if time_delta < 0
                else "Time Difference: 0.0 mins"
            )
            response_text=(
                "EquiTraffic-GPT (GWNet Shortest Path Planner)\n\n"
                f"Route: {origin_label} -> {destination_label}\n\n"
                f"Distance: {primary.get('distance_miles', 4.2):.1f} miles\n"
                f"Estimated Travel Time: {primary_time:.1f} mins\n"
                f"Alternate Distance: {float(alternate.get('distance_miles', primary.get('distance_miles', 4.2))):.1f} miles\n"
                f"Alternate Travel Time: {alternate_time:.1f} mins\n"
                f"{time_delta_label}\n"
                f"Average Speed: {primary.get('average_speed_mph', 45.0):.1f} mph\n"
                f"Highway Waypoints: {primary.get('path_sensor_count', 0)} sensors\n"
                f"Bottleneck Detected: {'Yes - reroute recommended' if bottleneck else 'No - clear path'}\n\n"
                "The route is highlighted on the GIS map."
            )
            return {
                "sensor_id": real_sensor_id,
                "user_prompt": req.prompt,
                "llm_response": response_text,
                "downstream_neighbors": downstream_ids,
                "gwnet_forecast_horizon": "15-min",
                "location": route_result.get("origin", {}).get("label", selected.get("location_label", "")),
                "recommended_path_coords": route_result["recommended_path_coords"],
                "route_result": route_result,
            }

        analysis=self.llm_engine.generate_causal_reasoning(
            prompt=req.prompt,
            sensor_id=real_sensor_id,
            speed=speed,
            predicted_speed=predicted_speed,
            rel=float(selected.get("reliability", 0.92)) * 100.0,
            status=status,
            downstream_nodes=downstream_ids,
            city=city_key,
            time_label=req.time_label,
            date_label=req.date_label,
            origin_id=route_origin.get("id", req.origin_id) if route_origin else req.origin_id,
            destination_id=route_destination.get("id", req.destination_id) if route_destination else req.destination_id,
        )
        return {
            "sensor_id": real_sensor_id,
            "user_prompt": req.prompt,
            "llm_response": analysis,
            "downstream_neighbors": downstream_ids,
            "gwnet_forecast_horizon": "15-min",
            "location": selected.get("location_label", ""),
        }
