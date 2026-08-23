# 🚦 EquiTraffic-GPT: Sensor-Centric Spatial-Temporal Traffic Forecasting & Generative Rerouting Twin

> **EquiTraffic-GPT (also referred to as CausalTwin-GPT)** is an enterprise-grade MLOps platform combining **Graph WaveNet (PyTorch 2.x)**, **Judea Pearl's Level-3 Causal Mediation**, **OpenStreetMap OSRM Map-Matching**, and **Google Gemini 2.5 Flash Lite Generative Copilot** to deliver proactive 15-minute travel advisories, A* shortest-path highway routing, and suburban regional equity analysis across 923 California sensors (**METR-LA** & **San Diego SD400**).

---

## 🌟 Key System Capabilities

1. **🔮 Authentic 15-Minute Neural Forecasts**:
   - Computes GraphWaveNet forward predictions ($\hat{y}_{t+3}$) on real 3D speed sequence tensors ($23,974 \times N \times 3$) from `metr_la_his.npz` and `sd400_his.npz`.
   - Automatically detects speed drops below $25\text{ mph}$ 15 minutes before physical shockwaves form.

2. **🛣️ Real-World OSRM Highway Map-Matching**:
   - Integrates OpenStreetMap OSRM API for per-segment curved highway polyline map-matching, replacing 2-point straight diagonal slants with granular physical street curves.

3. **🤖 Google Gemini 2.5 Flash Lite Copilot**:
   - Asynchronously streams natural language travel advisories, translating complex neural speed matrices into plain-English bottleneck warnings and estimated time savings (e.g. *"Saves 14–18 minutes"*).

4. **🧩 Level-3 Causal Mediation Analysis**:
   - Quantifies structural causal effects across physical loop detectors: **21.4% Direct Effect** ($C_{tf\_DE}$) from local merging vs. **61.3% Indirect Relay Effect** ($C_{tf\_IE\_R}$) from downstream bottlenecks.

5. **⚖️ Pareto Spatial Equity (RSF Metric)**:
   - Evaluates Region-based Static Fairness (RSF) profiles to ensure suburban outer-district commuters are not subjected to unequal congestion delays.

6. **🛠️ Telemetry Fault Diagnostics & Virtual Repair**:
   - Runs counterfactual $do(R_i)$ spatial neighborhood checks to distinguish sensor zero-reading hardware failures ($8.45\%$ METR-LA zero-dropout rate) from true traffic jams.

---

## 🏗️ Architecture Overview

```text
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   Web GIS Frontend (React 18 + Leaflet)                  │
 │           MapView.jsx | AnalyticsView.jsx | LlmChatbot.jsx              │
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

### 1. Backend Server Setup (FastAPI)
```bash
# Clone the repository
git clone https://github.com/Souptik-Hazra/Sensor-centric.git
cd Sensor-centric

# Install Python dependencies
pip install -r requirements.txt

# Launch FastAPI backend server (Port 8000)
python EquiTrafficAI/backend/backend.py
```

### 2. Frontend Setup (React + Vite)
```bash
# Navigate to web frontend directory
cd traffic-system-web

# Install Node dependencies
npm install

# Start Vite React Dev Server
npm run dev
```

Open **`http://localhost:5174/map`** in your web browser.

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
Developed as part of academic research on Spatial-Temporal Traffic Neural Networks and Causal Machine Learning. Distributed under the MIT License.
