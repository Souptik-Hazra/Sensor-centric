# 🌐 EquiTraffic-GPT Frontend

Interactive React 18 + Leaflet Web GIS application for real-time traffic monitoring, neural forecasting, OSRM route planning, and Gemini LLM Copilot advisories.

## ⚙️ Functional Requirements (FR)

1. **FR-1: Real-time GIS Visualization**: Render 923 sensor markers with dynamic speed colors (🟢 &ge;50 mph, 🟡 25–49 mph, 🔴 &lt;25 mph).
2. **FR-2: 15-Min Neural Forecast Alerts**: Automatically highlight predicted bottleneck speed drops 15 minutes in advance.
3. **FR-3: OSRM Highway Route Planning**: Display cyan optimal paths (`#06b6d4`) and dashed red congested links (`#f43f5e`) on real road curves.
4. **FR-4: Gemini LLM Smart Copilot**: Provide conversational reroute advice and active bottleneck speed reports (&lt;25 mph).
5. **FR-5: Pareto Equity Analytics**: Plot network speed curves across 288 timesteps and evaluate suburban fairness disparity.

## 🚀 How to Run

### Single-Server Mode (Port 8000)
```bash
python backend.py
```
Open **http://127.0.0.1:8000** in your browser.

### Vite Dev Server Mode (Port 5173)
```bash
npm run dev
```
Open **http://localhost:5173** in your browser.

## 📁 Folder Overview
* `src/modules/gis/MapView.jsx`: GIS map, sensor markers & OSRM polylines
* `src/modules/analytics/AnalyticsView.jsx`: Speed curves & Pareto fairness metrics
* `src/modules/monitoring/MonitoringView.jsx`: Live telemetry data table
* `src/core/LlmChatbot.jsx`: Gemini 2.5 LLM Smart Reroute Copilot
* `src/services/apiService.js`: FastAPI HTTP client
