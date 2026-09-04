# ⚙️ EquiTraffic-GPT Backend Serving Engine

FastAPI serving engine connecting one-time Colab-trained PyTorch spatiotemporal GraphWaveNet checkpoints, OpenStreetMap OSRM highway route planning, YAML-driven configurations, and Google Gemini 2.5 Flash Lite LLM Copilot advisories.

---

## ⚙️ Core API Endpoints

1. **`GET /api/state?city=la`**: Returns loaded sensor metadata and topology edges.
2. **`GET /api/health/models`**: Reports checkpoint readiness for each loaded model.
3. **`GET /api/predict/congestion_15min`**: Evaluates 15-minute proactive speed drops (< 25 mph) across all nodes.
4. **`POST /api/route/plan`**: Computes the LA A* shortest path with OSRM highway curve polylines.
5. **`POST /api/llm/reasoning`**: Calls the configured LLM provider or offline advisory engine.
6. **`POST /predict` & `POST /reroute`**: Direct tensor forecasting and corridor rerouting advisory endpoints.

---

## 📄 YAML Configuration Schemas

* **[`backend_config.yaml`](file:///c:/Users/User/Downloads/metr-la-dissertation-complete/EquiTrafficAI/backend/backend_config.yaml)**: Host/port server parameters, dataset directory candidates, CORS allowed origins, anomaly detection CUSUM/EWMA thresholds, and TTL caching.
* **[`model_config.yaml`](file:///c:/Users/User/Downloads/metr-la-dissertation-complete/EquiTrafficAI/backend/model_config.yaml)**: Graph WaveNet GNN hyper-parameters (adaptive matrix embeddings, spatial/temporal kernel sizes), Causal SCM decomposition ratios ($C_{tf\_DE} = 21.4\%$, $C_{tf\_IE\_R} = 61.3\%$), Pareto frontier optimization points, and Gemini 2.5 LLM prompt templates.

---

## 📁 Module Overview

* **`backend.py`**: FastAPI application composition, middleware, router registration, and static frontend dist file server.
* **`app_state.py`**: Shared model, location, and mutable runtime state initialization.
* **`lifecycle.py`**: Startup configuration, tensor, topology, and route-graph loading.
* **`routers/`**: FastAPI endpoint modules for state, health, forecast, routing, and LLM operations.
* **`services/`**: Forecast, routing, chatbot, speed conversion, and synthetic topology logic.
* **`llm_engine.py`**: Google Gemini 2.5 Flash Lite Generative AI Copilot integration.
* **`sensor_location_mapper.py`**: Geographic coordinate mapper & California freeway corridor registry.
* **`backend_config.yaml`**: Server & dataset configuration schema.
* **`model_config.yaml`**: GNN, Causal SCM, & LLM hyper-parameter schema.

---

## 🚀 Execution

From the project root directory:
```bash
python backend.py
```
Master server runs on **http://127.0.0.1:8000** (serves both API endpoints and React production web interface).
