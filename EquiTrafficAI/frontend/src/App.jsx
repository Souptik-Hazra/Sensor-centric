import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './core/Layout';
import MonitoringView from './modules/monitoring/MonitoringView';
import MapView from './modules/gis/MapView';
import AnalyticsView from './modules/analytics/AnalyticsView';
import SettingsView from './modules/settings/SettingsView';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<MonitoringView />} />
          <Route path="map" element={<MapView />} />
          <Route path="analytics" element={<AnalyticsView />} />
          <Route path="settings" element={<SettingsView />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
