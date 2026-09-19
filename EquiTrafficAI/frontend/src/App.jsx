import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './core/Layout';
import MonitoringView from './modules/monitoring/MonitoringView';
import MapView from './modules/gis/MapView';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<MonitoringView />} />
          <Route path="map" element={<MapView />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;