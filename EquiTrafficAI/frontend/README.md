# 🌐 EquiTraffic-GPT Frontend (React 18 + Web GIS Engine)

High-performance, accessible React 18 + Leaflet Web GIS application for real-time traffic monitoring, neural GNN forecasting, OSRM route planning, and Gemini 2.5 LLM Copilot advisories.

---

## 🎨 Architectural Highlights

1. **Dry Master Design System**:
   - Centralized styling in `src/index.css` with 0 inline styles (`style={{ ... }}`) across all JSX files. Over 606 lines of dead CSS purged via Python AST script.
2. **De-bloated Modular Architecture**:
   - All page views decomposed into pure, single-responsibility sub-components (`MapPlaybackCard`, `MapMarkerLayer`, `ChatHeaderBar`, `ChatInputFooter`, `RouteControlPanel`, `ExecutiveMetricsGrid`, `ParetoFrontierMatrix`, `SpeedTrendSvgChart`).
3. **60 FPS React Performance**:
   - Wrapped in `React.memo`, $O(1)$ Hash Map speed lookups, `useMemo` animation frame filters, and atomic Zustand slice selectors (`useTrafficStore`).
4. **GPU Hardware Acceleration**:
   - Uses Leaflet's `preferCanvas={true}` GPU context for 716 San Diego SD400 loop detector markers and polylines.
5. **Accessibility (a11y)**:
   - Built with high-contrast `:focus-visible` cyan focus rings, ARIA landmark roles, and screen-reader label associations.
6. **YAML Configuration Driven**:
   - Configured via `frontend_config.yaml` for dynamic environment targets, asset chunking, and theme tokens.

---

## ⚙️ Functional Requirements (FR)

1. **FR-1: Real-time GIS Visualization**: Render 923 sensor markers with dynamic speed colors (🟢 &ge;50 mph, 🟡 25–49 mph, 🔴 &lt;25 mph).
2. **FR-2: 15-Min Neural Forecast Alerts**: Automatically highlight predicted bottleneck speed drops 15 minutes in advance.
3. **FR-3: OSRM Highway Route Planning**: Display cyan optimal paths (`#00e5ff`) and dashed red congested links (`#ff0055`) on real road curves.
4. **FR-4: Gemini LLM Smart Copilot**: Provide conversational reroute advice and active bottleneck speed reports (&lt;25 mph).
5. **FR-5: Pareto Equity Analytics**: Plot network speed curves across 288 timesteps and evaluate suburban fairness disparity.

---

## 🚀 How to Run & Build

### Development Mode (Vite HMR - Port 5173)
```bash
npm install
npm run dev
```

### Production Build Verification
```bash
npm run build
```
Outputs minified assets to `dist/` in ~1.05s with zero compilation errors.

---

## 📁 Modular Directory Structure

```text
src/
├── core/
│   ├── components/
│   │   ├── ChatHeaderBar.jsx        # Copilot header bar & reset controls
│   │   ├── ChatInputFooter.jsx      # Copilot message input & send button
│   │   ├── ChatMessageList.jsx      # Scrollable chat message stream
│   │   └── QuickPromptChips.jsx     # Quick action prompt pills
│   ├── Layout.jsx                   # Master app shell & topbar
│   ├── Sidebar.jsx                  # Collapsible navigation drawer
│   └── LlmChatbot.jsx               # Gemini 2.5 LLM Copilot drawer container
├── modules/
│   ├── analytics/
│   │   ├── components/
│   │   │   ├── ExecutiveMetricsGrid.jsx # Metric KPI cards
│   │   │   ├── ParetoFrontierMatrix.jsx # Multi-objective policy matrix
│   │   │   └── SpeedTrendSvgChart.jsx   # 24-hour SVG playback chart
│   │   └── AnalyticsView.jsx        # Data analytics view container
│   ├── gis/
│   │   ├── components/
│   │   │   ├── CongestionWarningsCard.jsx # 15-min warning alerts
│   │   │   ├── MapLegend.jsx             # Map status dot legend
│   │   │   ├── MapMarkerLayer.jsx        # Leaflet markers & popups
│   │   │   ├── MapPlaybackCard.jsx       # 24-hour scrubber engine
│   │   │   ├── MapViewHeader.jsx         # Top overlay controls
│   │   │   └── RouteControlPanel.jsx     # Origin/Dest route planner
│   │   └── MapView.jsx              # Main Web GIS page container
│   ├── monitoring/
│   │   └── MonitoringView.jsx       # Telemetry data table
│   └── settings/
│       └── SettingsView.jsx         # System configuration panel
├── services/
│   └── apiService.js                # Axios/Fetch API client
├── store/
│   └── useTrafficStore.js           # Atomic Zustand global store
└── index.css                        # Master Design System (0 Dead CSS)
```
