import React, { useEffect, useState, useMemo, useCallback, useRef } from 'react';
import { MapContainer, Pane, TileLayer, Polyline, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import styles from './MapView.module.css';
import MapLegend from './components/MapLegend';
import RouteControlPanel from './components/RouteControlPanel';
import MapViewHeader from './components/MapViewHeader';
import CongestionWarningsCard from './components/CongestionWarningsCard';
import MapPlaybackCard from './components/MapPlaybackCard';
import MapMarkerLayer from './components/MapMarkerLayer';
import { fetchCityState, fetchTrafficState, planSmartRoute, requestLlmReasoning } from '../../services/apiService';

import simulationData from '../../core/simulationData.json';
const { empiricalProfiles }=simulationData;

const CARTO_URL='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const STEPS_PER_DAY=288;
const HISTORICAL_START_DATE='2012-03-01';

function MapResizeHandler({ isRightSidebarOpen }) {
  const map=useMap();
  useEffect(()=>{
    const timer=setTimeout(()=>{
      map.invalidateSize();
    }, 250);
    return () => clearTimeout(timer);
  }, [isRightSidebarOpen, map]);
  return null;
}

export default function MapView() {
  const [selectedCity, setSelectedCity]=useState('la');
  const [baseNodes, setBaseNodes]=useState([]);
  const [nodes, setNodes]=useState([]);
  const [edges, setEdges]=useState([]);
  const [step, setStep]=useState(96); // 08:00 AM
  const [isPlaying, setIsPlaying]=useState(false);
  const [selectedNodeId, setSelectedNodeId]=useState(0);
  const [date, setDate]=useState('2012-03-15');
  const [speedMultiplier,setSpeedMultiplier]=useState(10);
  
  const [isRightSidebarOpen,setIsRightSidebarOpen]=useState(true);
  
  // Feature 1: Mapped Sensor Origin / Destination Route Planner State
  const [originNodeId,setOriginNodeId]=useState(0);
  const [destinationNodeId,setDestinationNodeId]=useState(15);
  const [targetArrivalTime,setTargetArrivalTime]=useState('08:45 AM');
  const [routeResult,setRouteResult]=useState(null);
  const [isRouting,setIsRouting]=useState(false);

  // Feature 2: 15-Minute Congestion Warning Detector State
  const [upcoming15MinWarnings,setUpcoming15MinWarnings]=useState([]);
  const [futurePredictedSpeeds,setFuturePredictedSpeeds]=useState({});

  // Feature 3: "Something Interesting" — 🔮 15-Min Future Vision Mode State
  const [isFutureVisionActive,setIsFutureVisionActive]=useState(false);

  const cityRequestIdRef=useRef(0);
  const trafficRequestIdRef=useRef(0);
  const routeRequestIdRef=useRef(0);
  const llmRequestIdRef=useRef(0);

  // Sync state to LlmChatbot via event
  useEffect(()=>{
    window.dispatchEvent(new CustomEvent('app-state-sync', {
      detail: { 
        step: step, 
        city: selectedCity, 
        date: date,
        origin_id: originNodeId,
        destination_id: destinationNodeId
      }
    }));
  }, [step, selectedCity, date, originNodeId, destinationNodeId]);

  const getHistoricalTimestampIndex=useCallback((dateValue, stepValue)=>{
    const [year, month, day]=String(dateValue || HISTORICAL_START_DATE).split('-').map(Number);
    const [startYear, startMonth, startDay]=HISTORICAL_START_DATE.split('-').map(Number);
    const selectedUtc=Date.UTC(year, month - 1, day);
    const startUtc=Date.UTC(startYear, startMonth - 1, startDay);
    if(!Number.isFinite(selectedUtc) || !Number.isFinite(startUtc)) return stepValue;
    const dayOffset=Math.floor((selectedUtc - startUtc) / 86400000);
    return Math.max(0, dayOffset * STEPS_PER_DAY + stepValue);
  }, []);

  // Listen for LLM chatbot route results (cross-component event)
  useEffect(()=>{
    const handleLlmRoute=(e)=>{
      if(e.detail) {
        setRouteResult(e.detail);
      }
    };
    window.addEventListener('llm-route-result', handleLlmRoute);
    return () => window.removeEventListener('llm-route-result', handleLlmRoute);
  }, []);

  // Fetch City Datasets
  useEffect(()=>{
    const requestId=++cityRequestIdRef.current;
    const loadCityState=async ()=>{
      try {
        const data=await fetchCityState(selectedCity);
        if(requestId!==cityRequestIdRef.current) return;
        if(data) {
          const sensors=data.sensors || [];
          setBaseNodes(sensors);
          setNodes(sensors);
          setEdges(data.edges || []);
          setRouteResult(null);
          setSelectedNodeId(sensors.length > 0 ? sensors[0].id : null);
          if(sensors.length > 15) {
            setOriginNodeId(sensors[0].id);
            setDestinationNodeId(sensors[15].id);
          } else if(sensors.length > 1) {
            setOriginNodeId(sensors[0].id);
            setDestinationNodeId(sensors[sensors.length - 1].id);
          } else {
            setOriginNodeId(null);
            setDestinationNodeId(null);
          }
        }
      } catch (err) {
        if(requestId===cityRequestIdRef.current) {
          console.error('Failed to fetch city state:', err);
        }
      }
    };
    loadCityState();
  }, [selectedCity]);

  // Fetch 15-minute Congestion Warnings
  useEffect(()=>{
    if(isPlaying && step % 3 !== 0) return;
    const requestId=++trafficRequestIdRef.current;
    const fetch15MinWarnings=async ()=>{
      try {
        const timestampIndex=getHistoricalTimestampIndex(date, step);
        const data=await fetchTrafficState(timestampIndex, selectedCity);
        if(requestId!==trafficRequestIdRef.current) return;
        if(data) {
          setUpcoming15MinWarnings(data.readings || data.congested_nodes || []);
          setFuturePredictedSpeeds(data.predictedSpeeds || {});
        }
      } catch (err) {
        if(requestId===trafficRequestIdRef.current) {
          console.error('Failed to fetch 15-min warnings:', err);
        }
      }
    };
    fetch15MinWarnings();
  }, [selectedCity, step, date, isPlaying, getHistoricalTimestampIndex]);

  // 24-Hour Playback Loop Engine
  useEffect(()=>{
    let timer;
    if(isPlaying) {
      const intervalMs=Math.max(50, Math.floor(1000 / speedMultiplier));
      timer=setInterval(()=>{
        setStep((prev) => (prev + 1) % STEPS_PER_DAY);
      }, intervalMs);
    }
    return () => clearInterval(timer);
  }, [isPlaying, speedMultiplier]);

  // Dynamic Speed Profile Calculation
  useEffect(()=>{
    if(baseNodes.length === 0) return;
    const dateObj=new Date(`${date}T00:00:00`);
    const dayOfWeek=dateObj.getDay();
    const dayOfMonth=dateObj.getDate();
    const isWeekend=(dayOfWeek === 0 || dayOfWeek === 6);
    const strKey=String(dayOfWeek % (Object.keys(empiricalProfiles || {}).length || 7));
    const empDayData=selectedCity === 'la' && (empiricalProfiles[strKey] || empiricalProfiles[dayOfWeek]) ? (empiricalProfiles[strKey] || empiricalProfiles[dayOfWeek]) : null;
    const targetStep=isFutureVisionActive ? (step + 3) % STEPS_PER_DAY : step;
    const empStepSpeeds=(empDayData && empDayData[targetStep]) ? empDayData[targetStep] : null;
    const dateMult=isWeekend ? 0.45 : 1.0;
    const dateOffset=(dayOfMonth % 5 - 2) * 0.8;
    const totalMins=targetStep * 5;
    const hour=totalMins / 60.0;
    const rushFactor=(Math.exp(-Math.pow(hour - 8.0, 2) / 4.0) + Math.exp(-Math.pow(hour - 17.5, 2) / 4.0)) * dateMult;

    const updated=baseNodes.map((n, idx)=>{
      let speed=n.speed ?? 55.0;
      if(empStepSpeeds && idx < empStepSpeeds.length) {
        speed=empStepSpeeds[idx];
      } else {
        const noise=(Math.sin(idx * 1.5 + targetStep * 0.2) * 4.0 + Math.cos(idx * 0.8 - targetStep * 0.1) * 3.0);
        const drop=(idx % 7 === 0) ? (rushFactor * 26.0) : (idx % 3 === 0) ? (rushFactor * 16.0) : (rushFactor * 8.0);
        speed=Math.max(10.0, Math.min(75.0, speed - drop + noise + dateOffset));
      }

      let color='#2ecc71';
      let status='Clear';
      if(speed < 25.0) { color='#e74c3c'; status='Congested'; }
      else if(speed < 45.0) { color='#f1c40f'; status='Moderate'; }

      if(n.zero_rate && n.zero_rate > 0.4) {
        color='#38bdf8';
        status='Zero-Flow Anomaly';
      }

      return { ...n, speed: Math.round(speed * 10) / 10, color, status };
    });
    setNodes(updated);
  }, [baseNodes, step, date, selectedCity, isFutureVisionActive]);

  const getDisplayTime=useCallback((stepIdx)=>{
    const totalMinutes=stepIdx * 5;
    const hours=Math.floor(totalMinutes / 60);
    const mins=totalMinutes % 60;
    const period=hours >= 12 ? 'PM' : 'AM';
    const displayHours=hours % 12 === 0 ? 12 : hours % 12;
    const padMins=mins < 10 ? `0${mins}` : mins;
    return `${displayHours}:${padMins} ${period}`;
  }, []);

  const calculateSmartRoute=useCallback(async (oId, dId)=>{
    const actualOrigin=oId !== undefined ? oId : originNodeId;
    const actualDest=dId !== undefined ? dId : destinationNodeId;

    if(selectedCity !== 'la' || actualOrigin === null || actualOrigin === undefined || actualDest === null || actualDest === undefined) {
      setRouteResult(null);
      return;
    }

    const requestId=++routeRequestIdRef.current;
    setIsRouting(true);

    try {
      const data=await planSmartRoute(actualOrigin, actualDest, selectedCity, targetArrivalTime);
      if(requestId===routeRequestIdRef.current && data) setRouteResult(data);
    } catch (err) {
      if(requestId===routeRequestIdRef.current) {
        console.error('Failed to calculate route:', err);
      }
    } finally {
      if(requestId===routeRequestIdRef.current) setIsRouting(false);
    }
  }, [targetArrivalTime, selectedCity, originNodeId, destinationNodeId]);

  const runLlmQuery=useCallback(async (customPrompt='')=>{
    const requestId=++llmRequestIdRef.current;
    const targetSensor=selectedNodeId !== null ? selectedNodeId : 0;
    const promptText=customPrompt || `Which way to avoid and use if starting now for Sensor #${targetSensor}?`;
    
    try {
      const data=await requestLlmReasoning({
          sensor_id: targetSensor,
          prompt: promptText,
          city: selectedCity,
          step,
          origin_id: originNodeId,
          destination_id: destinationNodeId
      });

      if(requestId!==llmRequestIdRef.current) return;

      if(data) {
        if(data.route_result || data.recommended_path_coords) {
          window.dispatchEvent(new CustomEvent('llm-route-result', {
            detail: data.route_result || {
              recommended_path_coords: data.recommended_path_coords,
              congested_avoid_coords: data.congested_avoid_coords || []
            }
          }));
        }
      }
    } catch (err) {
      if(requestId===llmRequestIdRef.current) {
        console.error('Failed to run LLM query:', err);
      }
    }
  }, [selectedNodeId, selectedCity, step, originNodeId, destinationNodeId]);

  const mapCenter=useMemo(()=>{
    switch (selectedCity) {
      case 'sd': return [32.7157, -117.1611];
      case 'pems04': return [37.7749, -122.4194];
      case 'pems08': return [34.1083, -117.2898];
      case 'pems_bay': return [37.3382, -121.8863];
      case 'pems03': return [38.5816, -121.4944];
      case 'pems07': return [37.3382, -121.8863];
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
        <MapContainer key={selectedCity} center={mapCenter} zoom={selectedCity === 'sd' ? 10 : 11} preferCanvas={false} style={{ height: "100%", width: "100%" }} zoomControl={false}>
          <MapResizeHandler isRightSidebarOpen={isRightSidebarOpen} />
          <TileLayer
            className={styles.darkTileLayer}
            attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>'
            url={CARTO_URL}
          />

          {edges.map((edge, idx)=>(
            <Polyline
              key={idx}
              positions={edge}
              pathOptions={{ color: '#38bdf8', weight: 1, opacity: 0.35 }}
            />
          ))}

          {/* HIGHLIGHTED RECOMMENDED ROUTE EDGES */}
          {routeResult && Array.isArray(routeResult.recommended_path_coords) && routeResult.recommended_path_coords.length > 0 && (
            <Pane name="recommendedRoute" style={{ zIndex: 410 }}>
              <Polyline
                positions={
                  Array.isArray(routeResult.recommended_path_coords[0]?.[0])
                    ? routeResult.recommended_path_coords.map(pair => pair).flat()
                    : routeResult.recommended_path_coords
                }
                pathOptions={{ color: '#102A43', weight: 11, opacity: 0.9, lineCap: 'round', lineJoin: 'round' }}
              />
              <Polyline
                positions={
                  Array.isArray(routeResult.recommended_path_coords[0]?.[0])
                    ? routeResult.recommended_path_coords.map(pair => pair).flat()
                    : routeResult.recommended_path_coords
                }
                pathOptions={{ color: '#2DD4BF', weight: 6, opacity: 1.0, lineCap: 'round', lineJoin: 'round' }}
              />
            </Pane>
          )}

          {/* HIGHLIGHTED CONGESTED BOTTLENECK EDGES TO AVOID (Pulsing Crimson Red) */}
          {routeResult && Array.isArray(routeResult.congested_avoid_coords) && routeResult.congested_avoid_coords.map((pair, idx)=>(
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
            futurePredictedSpeeds={futurePredictedSpeeds}
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

        {/* FEATURE 3: Replay & Calendar Engine Card */}
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