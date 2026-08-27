# ⚙️ EquiTraffic-GPT Backend MLOps Engine

FastAPI MLOps serving engine connecting PyTorch spatiotemporal GraphWaveNet predictions, Judea Pearl's Level-3 Causal Mediation Analysis, OpenStreetMap OSRM highway route planning, YAML-driven configurations, and Google Gemini 2.5 Flash Lite LLM Copilot advisories.

---

## ⚙️ Core API Endpoints

1. **`GET /api/state?city=la`**: Returns 923 geocoded loop detector locations and spatial adjacency graph edges for METR-LA, SD400, and PeMS corridors.
2. **`GET /api/predict/congestion_15min`**: Evaluates 15-minute proactive speed drops (< 25 mph) across all nodes.
3. **`POST /api/route/plan`**: Computes A* shortest path with OSRM highway curve polylines and recommended departure times.
4. **`POST /api/llm/reasoning`**: Asynchronously calls Gemini 2.5 LLM to generate plain-English bottleneck advisories.
5. **`POST /api/policy/pareto`**: Evaluates suburban equity vs maximum throughput Pareto trade-off policies.
6. **`POST /predict` & `POST /reroute`**: Batch tensor GNN forecasting and Caltrans real-ID index resolution endpoints.

---

## 📄 YAML Configuration Schemas

* **[`backend_config.yaml`](file:///c:/Users/User/Downloads/metr-la-dissertation-complete/EquiTrafficAI/backend/backend_config.yaml)**: Host/port server parameters, dataset directory candidates, CORS allowed origins, anomaly detection CUSUM/EWMA thresholds, and TTL caching.
* **[`model_config.yaml`](file:///c:/Users/User/Downloads/metr-la-dissertation-complete/EquiTrafficAI/backend/model_config.yaml)**: Graph WaveNet GNN hyper-parameters (adaptive matrix embeddings, spatial/temporal kernel sizes), Causal SCM decomposition ratios ($C_{tf\_DE} = 21.4\%$, $C_{tf\_IE\_R} = 61.3\%$), Pareto frontier optimization points, and Gemini 2.5 LLM prompt templates.

---

## 📁 Module Overview

* **`backend.py`**: Master FastAPI application router, OSRM highway path router, CUSUM anomaly detector, and static frontend dist file server.
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
