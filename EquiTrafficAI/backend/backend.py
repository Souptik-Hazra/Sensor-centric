"""FastAPI application composition for EquiTraffic-GPT."""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import app_state
from .lifecycle import load_all_data
from .routers.forecast import direct_router, router as forecast_router
from .routers.health import router as health_router
from .routers.llm import router as llm_router
from .routers.routing import router as routing_router
from .routers.state import router as state_router


async def lifespan(app: FastAPI):
    load_all_data()
    app.state.state_data=app_state.state_data
    app.state.gwnet_adapters=app_state.gwnet_adapters
    app.state.route_graph_data=app_state.route_graph_data
    app.state.la_location_map=app_state.la_location_map
    app.state.chatbot_service=app_state.chatbot_service
    app.state.route_planner=app_state.route_planner
    yield


app=FastAPI(
    title="EquiTraffic-GPT Master API",
    description="SOTA Traffic LLM Copilot & Graph WaveNet (GWNet) Neural Forecasting API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast_router)
app.include_router(direct_router)
app.include_router(health_router)
app.include_router(llm_router)
app.include_router(routing_router)
app.include_router(state_router)


dist_candidates=[
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "frontend", "dist")),
]
for candidate in dist_candidates:
    if os.path.exists(candidate):
        @app.get("/map", include_in_schema=False)
        def serve_map_app():
            return FileResponse(os.path.join(candidate, "index.html"))

        app.mount("/", StaticFiles(directory=candidate, html=True), name="static")
        print(f"[OK] Single-Server Mode Active: Serving React Web GIS from {candidate}")
        break


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
