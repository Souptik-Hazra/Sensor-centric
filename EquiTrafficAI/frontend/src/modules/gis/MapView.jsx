import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { MapContainer, TileLayer, Polyline, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import styles from './MapView.module.css';
import MapLegend from './components/MapLegend';
import RouteControlPanel from './components/RouteControlPanel';
import MapViewHeader from './components/MapViewHeader';
import CongestionWarningsCard from './components/CongestionWarningsCard';
import MapPlaybackCard from './components/MapPlaybackCard';
import MapMarkerLayer from './components/MapMarkerLayer';

import simulationData from '../../core/simulationData.json';
const { empiricalProfiles } = simulationData;

const CARTO_URL = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

function MapResizeHandler({ isRightSidebarOpen }) {
  const map = useMap();
  useEffect(() => {
    const timer = setTimeout(() => {
      map.invalidateSize();
    }, 250);
    return () => clearTimeout(timer);
  }, [isRightSidebarOpen, map]);
  return null;
}

export default function MapView() {
  const [selectedCity, setSelectedCity] = useState('la');
  const [baseNodes, setBaseNodes] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [step, setStep] = useState(96); // 08:00 AM
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState(0);
  const [date, setDate] = useState('2012-03-15');
  const [speedMultiplier, setSpeedMultiplier] = useState(10);
  
  const [showEdges, setShowEdges] = useState(true);
  const [isRightSidebarOpen, setIsRightSidebarOpen] = useState(true);
  
  // Feature 1: Mapped Sensor Origin / Destination Route Planner State
  const [originNodeId, setOriginNodeId] = useState(0);
  const [destinationNodeId, setDestinationNodeId] = useState(15);
  const [targetArrivalTime, setTargetArrivalTime] = useState('08:45 AM');
  const [routeResult, setRouteResult] = useState(null);
  const [isRouting, setIsRouting] = useState(false);

  // Feature 2: 15-Minute Congestion Warning Detector State
  const [upcoming15MinWarnings, setUpcoming15MinWarnings] = useState([]);

  // Feature 3: "Something Interesting" — 🔮 15-Min Future Vision Mode State
  const [isFutureVisionActive, setIsFutureVisionActive] = useState(false);

  // Pareto Policy State
  const [policyGoal, setPolicyGoal] = useState('equity');
  const [activePolicy, setActivePolicy] = useState(null);

  // LLM State
  const [llmPrompt, setLlmPrompt] = useState('');
  const [llmResponse, setLlmResponse] = useState(null);
  const [isLlmLoading, setIsLlmLoading] = useState(false);

  // Fetch City Datasets
  useEffect(() => {
    const fetchCityState = async () => {
      try {
        const response = await fetch(`/api/state?city=${selectedCity}`);
        if (response.ok) {
          const data = await response.json();
          setBaseNodes(data.sensors || []);
          setNodes(data.sensors || []);
          setEdges(data.edges || []);
          setMetrics({
            baseline_rsf: data.baseline_rsf || 0.0920,
            count: data.count || (data.sensors || []).length
          });
          if ((data.sensors || []).length > 15) {
            setOriginNodeId(data.sensors[0].id);
            setDestinationNodeId(data.sensors[15].id);
          }
        }
      } catch (err) {
        console.error('Failed to fetch city state:', err);
      }
    };
    fetchCityState();
  }, [selectedCity]);

  // Fetch 15-minute Congestion Warnings
  useEffect(() => {
    const fetch15MinWarnings = async () => {
      try {
        const response = await fetch(`/api/predict/congestion_15min?city=${selectedCity}&timestamp_index=${step}`);
        if (response.ok) {
          const data = await response.json();
          setUpcoming15MinWarnings(data.congested_nodes || []);
        }
      } catch (err) {
        console.error('Failed to fetch 15-min warnings:', err);
      }
    };
    fetch15MinWarnings();
  }, [selectedCity, step]);

  // 24-Hour Playback Loop Engine
  useEffect(() => {
    let timer;
    if (isPlaying) {
      const intervalMs = Math.max(50, Math.floor(1000 / speedMultiplier));
      timer = setInterval(() => {
        setStep((prev) => (prev + 1) % 288);
      }, intervalMs);
    }
    return () => clearInterval(timer);
  }, [isPlaying, speedMultiplier]);

  // Dynamic Speed Profile Calculation
  useEffect(() => {
    if (baseNodes.length === 0) return;
    const updated = baseNodes.map((n) => {
      let speed;
      const empirical = empiricalProfiles ? empiricalProfiles[String(n.sensor_id || n.id)] : null;
      if (empirical && empirical.length > step) {
        speed = empirical[step];
      } else {
        const t = step * 5 / 60.0;
        const morningPeak = Math.exp(-Math.pow(t - 8.0, 2) / 4.0);
        const eveningPeak = Math.exp(-Math.pow(t - 17.5, 2) / 4.0);
        const baseSpeed = n.speed || 58.5;
        const dip = (morningPeak + eveningPeak) * 30.0;
        const noise = (Math.sin(n.id * 1.5 + step * 0.2) * 2.5);
        speed = Math.max(10.0, Math.min(75.0, baseSpeed - dip + noise));
      }

      let color = '#2ecc71';
      let status = 'Clear';
      if (speed < 25.0) { color = '#e74c3c'; status = 'Congested'; }
      else if (speed < 45.0) { color = '#f1c40f'; status = 'Moderate'; }

      if (n.zero_rate && n.zero_rate > 0.4) {
        color = '#38bdf8';
        status = 'Zero-Flow Anomaly';
      }

      return { ...n, speed: Math.round(speed * 10) / 10, color, status };
    });
    setNodes(updated);
  }, [baseNodes, step]);

  const getDisplayTime = useCallback((stepIdx) => {
    const totalMinutes = stepIdx * 5;
    const hours = Math.floor(totalMinutes / 60);
    const mins = totalMinutes % 60;
    const period = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 === 0 ? 12 : hours % 12;
    const padMins = mins < 10 ? `0${mins}` : mins;
    return `${displayHours}:${padMins} ${period}`;
  }, []);

  const calculateSmartRoute = useCallback(async (oId, dId) => {
    setIsRouting(true);
    try {
      const response = await fetch('/api/route/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          origin_id: oId,
          destination_id: dId,
          target_time: targetArrivalTime,
          city: selectedCity
        })
      });
      if (response.ok) {
        const data = await response.json();
        setRouteResult(data);
      }
    } catch (err) {
      console.error('Failed to calculate route:', err);
    } finally {
      setIsRouting(false);
    }
  }, [targetArrivalTime, selectedCity]);

  const runLlmQuery = useCallback(async (customPrompt = '') => {
    setIsLlmLoading(true);
    const targetSensor = selectedNodeId !== null ? selectedNodeId : 0;
    const promptText = customPrompt || llmPrompt || `Which way to avoid and use if starting now for Sensor #${targetSensor}?`;
    
    try {
      const response = await fetch('/api/llm/reasoning', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sensor_id: targetSensor,
          prompt: promptText,
          city: selectedCity
        })
      });
      if (response.ok) {
        const data = await response.json();
        setLlmResponse(data.llm_response);
      }
    } catch (err) {
      console.error('Failed to run LLM query:', err);
    } finally {
      setIsLlmLoading(false);
    }
  }, [selectedNodeId, llmPrompt, selectedCity]);

  const mapCenter = useMemo(() => {
    switch (selectedCity) {
      case 'sd': return [32.7157, -117.1611];
      case 'pems04': return [37.7749, -122.4194];
      case 'pems08': return [34.1083, -117.2898];
      case 'pems_bay': return [37.3382, -121.8863];
      case 'pems03': return [38.5816, -121.4944];
      default: return [34.0522, -118.2437];
    }
  }, [selectedCity]);

  return (
    <div className={styles.mapContainer}>
      
      {/* Header Overlay Bar Sub-Component */}
      <MapViewHeader 
        isRightSidebarOpen={isRightSidebarOpen}
        selectedCity={selectedCity}
        setSelectedCity={setSelectedCity}
        isFutureVisionActive={isFutureVisionActive}
        setIsFutureVisionActive={setIsFutureVisionActive}
        setIsRightSidebarOpen={setIsRightSidebarOpen}
      />

      <div className={styles.mapWrapper}>
        <MapContainer key={selectedCity} center={mapCenter} zoom={selectedCity === 'sd' ? 10 : 11} preferCanvas={true} style={{ height: "100%", width: "100%" }} zoomControl={false}>
          <MapResizeHandler isRightSidebarOpen={isRightSidebarOpen} />
          <TileLayer
            attribution='&copy; <a href="https://carto.com/">CARTO</a>'
            url={CARTO_URL}
          />
          {showEdges && edges.map((edge, idx) => (
            <Polyline
              key={idx}
              positions={edge}
              pathOptions={{ color: '#38bdf8', weight: 1, opacity: 0.35 }}
            />
          ))}

          {/* HIGHLIGHTED RECOMMENDED ROUTE EDGES (Clean Neon Cyan Path) */}
          {routeResult && routeResult.recommended_path_coords && routeResult.recommended_path_coords.map((pair, idx) => (
            <Polyline
              key={`rec-${idx}`}
              positions={pair}
              pathOptions={{ color: '#00e5ff', weight: 4, opacity: 0.9 }}
            />
          ))}

          {/* HIGHLIGHTED CONGESTED BOTTLENECK EDGES TO AVOID (Pulsing Crimson Red) */}
          {routeResult && routeResult.congested_avoid_coords && routeResult.congested_avoid_coords.map((pair, idx) => (
            <Polyline
              key={`avoid-${idx}`}
              positions={pair}
              pathOptions={{ color: '#ff0055', weight: 4, dashArray: '6, 6', opacity: 0.95 }}
            />
          ))}

          {/* Leaflet CircleMarker & Popup Layer Sub-Component */}
          <MapMarkerLayer 
            nodes={nodes}
            isFutureVisionActive={isFutureVisionActive}
            upcoming15MinWarnings={upcoming15MinWarnings}
            originNodeId={originNodeId}
            destinationNodeId={destinationNodeId}
            selectedNodeId={selectedNodeId}
            setSelectedNodeId={setSelectedNodeId}
            runLlmQuery={runLlmQuery}
            setOriginNodeId={setOriginNodeId}
            setDestinationNodeId={setDestinationNodeId}
            calculateSmartRoute={calculateSmartRoute}
          />
        </MapContainer>
        
        {/* Map Legend Sub-Component */}
        <MapLegend />
      </div>

      {/* Right Control Drawer */}
      <div className={`${styles.sidebar} ${!isRightSidebarOpen ? styles.sidebarClosed : ''}`}>
        
        {/* FEATURE 1: Smart Route Control Panel Sub-Component */}
        <RouteControlPanel 
          nodes={nodes}
          originNodeId={originNodeId}
          setOriginNodeId={setOriginNodeId}
          destinationNodeId={destinationNodeId}
          setDestinationNodeId={setDestinationNodeId}
          targetArrivalTime={targetArrivalTime}
          setTargetArrivalTime={setTargetArrivalTime}
          calculateSmartRoute={calculateSmartRoute}
          isRouting={isRouting}
          routeResult={routeResult}
        />

        {/* FEATURE 2: 15-Minute Congestion Warning Alerts Sub-Component */}
        <CongestionWarningsCard 
          upcoming15MinWarnings={upcoming15MinWarnings}
          getDisplayTime={getDisplayTime}
          step={step}
        />

        {/* FEATURE 3: Replay & Calendar Engine Card Sub-Component */}
        <MapPlaybackCard 
          date={date}
          setDate={setDate}
          step={step}
          setStep={setStep}
          isPlaying={isPlaying}
          setIsPlaying={setIsPlaying}
          getDisplayTime={getDisplayTime}
          speedMultiplier={speedMultiplier}
          setSpeedMultiplier={setSpeedMultiplier}
        />
      </div>
    </div>
  );
}
