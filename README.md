# 🚦 EquiTraffic-GPT: Sensor-Centric Spatial-Temporal Traffic Forecasting & Generative Rerouting Twin

> **EquiTraffic-GPT (also referred to as CausalTwin-GPT)** is an enterprise-grade MLOps platform combining **Graph WaveNet (PyTorch 2.x)**, **Judea Pearl's Level-3 Causal Mediation**, **OpenStreetMap OSRM Map-Matching**, **YAML-driven Microservices**, and **Google Gemini 2.5 Flash Lite Generative Copilot** to deliver proactive 15-minute travel advisories, A* shortest-path highway routing, and suburban regional equity analysis across 923 California sensors (**METR-LA** & **San Diego SD400**).

---

## 🌟 Key System Capabilities

1. **🔮 Authentic 15-Minute Neural Forecasts**:
   - Computes GraphWaveNet forward predictions ($\hat{y}_{t+3}$) on real 3D speed sequence tensors ($23,974 \times N \times 3$) from `metr_la_his.npz` and `sd400_his.npz`.
   - Automatically detects speed drops below $25\text{ mph}$ 15 minutes before physical shockwaves form.

2. **🎨 Production Master Design System & Zero Dead CSS**:
   - Consolidated 100% of frontend styles into a unified, DRY design system ([`src/index.css`](file:///c:/Users/User/Downloads/metr-la-dissertation-complete/EquiTrafficAI/frontend/src/index.css)) with zero static inline styles (`style={{ ... }}`) and over 606 lines of redundant CSS purged via Python AST automated audits.

3. **🧱 Modular De-bloated React 18 Architecture**:
   - Monolithic components refactored into pure, single-responsibility sub-components (`MapPlaybackCard`, `MapMarkerLayer`, `ChatHeaderBar`, `ChatInputFooter`, `RouteControlPanel`, `ExecutiveMetricsGrid`, `ParetoFrontierMatrix`, `SpeedTrendSvgChart`).
   - Wrapped in `React.memo` and atomic Zustand selectors for 60 FPS timeline scrubbing.

4. **⚙️ YAML-Driven Enterprise Configurations**:
   - Modularized settings into 3 dedicated YAML configuration schemas:
     - [`frontend_config.yaml`](file:///c:/Users/User/Downloads/metr-la-dissertation-complete/EquiTrafficAI/frontend/frontend_config.yaml): Web GIS engine, GPU Canvas rendering, Vite asset chunking, and color tokens.
     - [`backend_config.yaml`](file:///c:/Users/User/Downloads/metr-la-dissertation-complete/EquiTrafficAI/backend/backend_config.yaml): FastAPI worker settings, dataset candidate paths, CUSUM/EWMA anomaly thresholds.
     - [`model_config.yaml`](file:///c:/Users/User/Downloads/metr-la-dissertation-complete/EquiTrafficAI/backend/model_config.yaml): Graph WaveNet GNN hyper-parameters, Causal SCM ratios ($C_{tf\_DE} = 21.4\%$, $C_{tf\_IE\_R} = 61.3\%$), Pareto equity targets, and Gemini Flash 2.5 prompt templates.

5. **♿ Accessibility (a11y) & GPU Canvas Hardware Acceleration**:
   - Enabled `preferCanvas={true}` for 716 San Diego sensors GPU rendering alongside high-contrast `:focus-visible` keyboard focus rings and ARIA screen reader landmark roles.

---

## 🏗️ Architecture Overview

```text
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   Web GIS Frontend (React 18 + Leaflet)                  │
 │      MapView.jsx | AnalyticsView.jsx | LlmChatbot.jsx | index.css        │
 └─────────────────────────────────────────────────────────────────────────┘
                                     │ (HTTP REST API)
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                       FastAPI MLOps Backend (backend.py)                │
 │       GET /api/state | GET /api/predict/congestion_15min | /api/route/plan │
 └─────────────────────────────────────────────────────────────────────────┘
          │                                  │                        │
          ▼                                  ▼                        ▼
 ┌───────────────────┐             ┌───────────────────┐    ┌───────────────────┐
 │ PyTorch GWNet     │             │ A* Search Router  │    │ Gemini 2.5 LLM    │
 │ (gwnet_adapter.py)│             │ (OSRM Geometry)   │    │ Copilot Engine    │
 └───────────────────┘             └───────────────────┘    └───────────────────┘
```

---

## 🚀 Quickstart Guide

### Single-Server Mode (Recommended)
Run the master single-command runner from the root directory:
```bash
# Clone the repository
git clone https://github.com/Souptik-Hazra/Sensor-centric.git
cd Sensor-centric

# Launch Unified Single-Server (FastAPI Backend + React Web GIS UI)
python backend.py
```

Open **`http://127.0.0.1:8000/`** in your web browser.

### Separate Development Mode (Vite HMR)
```bash
# Navigate to web frontend directory
cd EquiTrafficAI/frontend

# Install Node dependencies & launch Vite Dev Server
npm install
npm run dev
```

Open **`http://localhost:5173/`** in your web browser.

---

## 📊 Benchmark Performance Results

| Model / Paradigm | MAE (mph) | RSF Equity Disparity | Pareto Status |
|---|---|---|---|
| DCRNN Baseline | 2.77 | 0.380 | Dominated |
| FairSTG Baseline | 2.45 | 0.280 | Sub-optimal |
| **GWNet (Suburban Equity)** | **2.15** | **0.140** | **Pareto Optimal** |
| **GWNet (Max Throughput)** | **1.82** | **0.220** | **Pareto Optimal** |

---

## 📜 License
Developed as part of academic research on Spatial-Temporal Traffic Neural Networks, Causal Machine Learning, and Web GIS Digital Twins. Distributed under the MIT License.
