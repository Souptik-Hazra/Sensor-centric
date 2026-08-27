# ⚙️ EquiTraffic-GPT Backend Engine

FastAPI MLOps serving engine connecting PyTorch spatiotemporal GraphWaveNet predictions, OpenStreetMap OSRM highway route planning, and Google Gemini 2.5 Flash Lite LLM Copilot advisories.

## ⚙️ Core API Endpoints

1. **`GET /api/state?city=la`**: Returns 923 geocoded loop detector locations and spatial adjacency graph edges.
2. **`GET /api/predict/congestion_15min`**: Evaluates 15-minute proactive speed drops (< 25 mph) across all nodes.
3. **`POST /api/route/plan`**: Computes A* shortest path with OSRM highway curve polylines and recommended departure times.
4. **`POST /api/llm/reasoning`**: Asynchronously calls Gemini 2.5 LLM to generate plain-English bottleneck advisories.
5. **`POST /predict` & `POST /reroute`**: Batch tensor GNN forecasting and Caltrans real-ID index resolution endpoints.

## 📁 Module Overview

* **`backend.py`**: Master FastAPI application router, OSRM router, and static frontend dist file server.
* **`llm_engine.py`**: Google Gemini 2.5 Flash Lite Generative AI Copilot integration.
* **`sensor_location_mapper.py`**: Geographic coordinate mapper & California freeway corridor registry.

## 🚀 Execution

From the project root directory:
```bash
python backend.py
```
Server runs on **http://127.0.0.1:8000**
