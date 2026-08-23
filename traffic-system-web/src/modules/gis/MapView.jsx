import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, Polyline, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Play, Pause, RotateCcw, ShieldCheck, FileText, Globe, Calendar, Clock, Sliders, ChevronRight, Bot, Send, Navigation, AlertTriangle, Eye, Sparkles, MapPin } from 'lucide-react';
import styles from './MapView.module.css';

import simulationData from '../../core/simulationData.json';
const { empiricalProfiles } = simulationData;
const hasEmpiricalData = Object.keys(empiricalProfiles || {}).length > 0;

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

  // Time formatting
  const getDisplayTime = (currentStep) => {
    const totalMinutes = currentStep * 5;
    const hours = Math.floor(totalMinutes / 60);
    const mins = totalMinutes % 60;
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 === 0 ? 12 : hours % 12;
    const displayMins = mins < 10 ? '0' + mins : mins;
    return `${displayHours}:${displayMins} ${ampm}`;
  };

  useEffect(() => {
    let interval;
    if (isPlaying) {
      const intervalMs = Math.max(30, Math.floor(600 / speedMultiplier));
      interval = setInterval(() => {
        setStep(prev => (prev + 1) % 288);
      }, intervalMs);
    }
    return () => clearInterval(interval);
  }, [isPlaying, speedMultiplier]);

  // Speed Classification & Future Vision Mode Logic
  useEffect(() => {
    if (baseNodes.length === 0) return;

    const dateObj = new Date(date || '2012-03-15');
    const dayOfWeek = dateObj.getDay(); 
    const dayOfMonth = dateObj.getDate();
    const isWeekend = (dayOfWeek === 0 || dayOfWeek === 6);
    
    const strKey = String(dayOfWeek % (Object.keys(empiricalProfiles || {}).length || 7));
    const empDayData = (hasEmpiricalData && (empiricalProfiles[strKey] || empiricalProfiles[dayOfWeek])) ? (empiricalProfiles[strKey] || empiricalProfiles[dayOfWeek]) : null;
    
    // Future Vision: look 3 steps (15 mins) ahead!
    const targetStep = isFutureVisionActive ? (step + 3) % 288 : step;
    const empStepSpeeds = (empDayData && empDayData[targetStep]) ? empDayData[targetStep] : null;

    const dateMult = isWeekend ? 0.45 : 1.0;
    const dateOffset = (dayOfMonth % 5 - 2) * 0.8;
    
    const totalMins = targetStep * 5;
    const hour = totalMins / 60.0;
    const rushFactor = (Math.exp(-Math.pow(hour - 8.0, 2) / 4.0) + Math.exp(-Math.pow(hour - 17.5, 2) / 4.0)) * dateMult;

    const updated = baseNodes.map((sensor, idx) => {
      let speed = sensor.speed || 55.0;

      if (empStepSpeeds && idx < empStepSpeeds.length) {
        speed = empStepSpeeds[idx];
      } else {
        const noise = (Math.sin(idx * 1.5 + targetStep * 0.2) * 4.0 + Math.cos(idx * 0.8 - targetStep * 0.1) * 3.0);
        const drop = (idx % 7 === 0) ? (rushFactor * 26.0) : (idx % 3 === 0) ? (rushFactor * 16.0) : (rushFactor * 8.0);
        speed = Math.max(0.0, Math.min(70.0, 62.0 - drop + noise + dateOffset));
      }

      let rel = (speed / 70.0) + 0.15;
      if (activePolicy && activePolicy.reliability_paradigm === 'reliability_equal') {
        rel = Math.max(rel, 0.88);
      }
      rel = Math.max(0.45, Math.min(0.98, rel));

      let color = '#2ecc71';
      let status = 'FREE_FLOW';

      if (speed < 1.0) {
        color = '#38bdf8';
        status = 'ZERO_FLOW';
      } else if (speed < 25.0) {
        color = '#e74c3c';
        status = 'BOTTLENECK';
      } else if (speed < 50.0) {
        color = '#f1c40f';
        status = 'DEGRADING';
      }

      return {
        ...sensor,
        speed: parseFloat(speed.toFixed(1)),
        reliability: parseFloat(rel.toFixed(3)),
        color,
        status
      };
    });

    setNodes(updated);
  }, [step, baseNodes, date, isFutureVisionActive, activePolicy]);

  // Execute Smart Route Planning (Origin -> Destination -> Target Time)
  const calculateSmartRoute = async (customOrigin = null, customDest = null) => {
    const oId = customOrigin !== null ? customOrigin : originNodeId;
    const dId = customDest !== null ? customDest : destinationNodeId;
    
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
  };

  const applyParetoPolicy = async (goalType) => {
    setPolicyGoal(goalType);
    try {
      const response = await fetch('/api/policy/pareto', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal: goalType })
      });
      if (response.ok) {
        const data = await response.json();
        setActivePolicy(data);
      }
    } catch (err) {
      console.error('Failed to apply Pareto policy:', err);
    }
  };

  const runLlmQuery = async (customPrompt = '') => {
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
  };

  const healthyCount = nodes.filter(n => n.color === '#2ecc71').length;
  const driftCount = nodes.filter(n => n.color === '#f1c40f').length;
  const bottleneckCount = nodes.filter(n => n.color === '#e74c3c').length;
  const zeroFlowCount = nodes.filter(n => n.color === '#38bdf8').length;

  const mapCenter = selectedCity === 'sd' ? [32.7157, -117.1611] : (selectedCity === 'pems04' ? [37.7749, -122.4194] : (selectedCity === 'pems08' ? [34.1083, -117.2898] : (selectedCity === 'pems_bay' ? [37.3382, -121.8863] : (selectedCity === 'pems03' ? [38.5816, -121.4944] : [34.0522, -118.2437]))));
  const selectedNode = nodes.find(n => n.id === selectedNodeId) || nodes[0];

  return (
    <div className={styles.mapContainer}>
      
      {/* Header Overlay Bar - Single Line */}
      <div className={`${styles.mapHeaderOverlay} ${!isRightSidebarOpen ? styles.mapHeaderOverlayFull : ''}`}>
        <div className={styles.singleHeaderBar}>
          <Globe size={15} color="#38bdf8" />
          <span className={styles.headerTitle}>EquiTraffic Console:</span>
          
          <select 
            value={selectedCity} 
            onChange={(e) => setSelectedCity(e.target.value)}
            className={styles.headerSelect}
          >
            <option value="la">Los Angeles METR-LA (207 Mapped Sensors)</option>
            <option value="sd">San Diego SD400 (716 Mapped Sensors)</option>
            <option value="pems04">PeMS04 Bay Area (307 Mapped Sensors)</option>
            <option value="pems08">PeMS08 San Bernardino (170 Mapped Sensors)</option>
            <option value="pems_bay">PeMS-BAY San Jose (325 Mapped Sensors)</option>
            <option value="pems03">PeMS03 Sacramento (358 Mapped Sensors)</option>
            <option value="pems07">PeMS07 Greater LA (883 Mapped Sensors)</option>
          </select>

          <span className={styles.divider}>|</span>

          {/* Feature 3 ("Something Interesting"): 🔮 15-Min Future Vision Toggle */}
          <button 
            onClick={() => setIsFutureVisionActive(!isFutureVisionActive)}
            className={styles.btn}
            style={{ 
              backgroundColor: isFutureVisionActive ? '#9333ea' : '#1e293b', 
              color: '#ffffff',
              border: '1px solid #a855f7',
              fontSize: '11px',
              padding: '4px 10px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
            title="Toggle 15-Min Future Congestion Simulator"
          >
            <Eye size={14} color={isFutureVisionActive ? '#ffffff' : '#c084fc'} />
            <span>{isFutureVisionActive ? '🔮 15-Min Future Vision ACTIVE' : '🔮 Enable 15-Min Future Vision'}</span>
          </button>

          <span className={styles.divider}>|</span>

          <button 
            onClick={() => setIsRightSidebarOpen(!isRightSidebarOpen)} 
            className={styles.toggleRightBtn}
            title={isRightSidebarOpen ? "Hide Right Control Panel" : "Open Right Control Panel"}
          >
            {isRightSidebarOpen ? <ChevronRight size={14} /> : <Sliders size={14} />}
            <span>{isRightSidebarOpen ? 'Hide Panel' : 'Controls'}</span>
          </button>
        </div>
      </div>

      <div className={styles.mapWrapper}>
        <MapContainer key={selectedCity} center={mapCenter} zoom={selectedCity === 'sd' ? 10 : 11} style={{ height: "100%", width: "100%" }} zoomControl={false}>
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

          {nodes.map(node => {
            const isWarnedInFuture = isFutureVisionActive && upcoming15MinWarnings.some(w => w.id === node.id || w.sensor_id === node.sensor_id);
            const isOrigin = originNodeId === node.id;
            const isDest = destinationNodeId === node.id;
            const isSelected = selectedNodeId === node.id;

            return (
              <CircleMarker
                key={`${node.id}-${node.color}-${isWarnedInFuture}`}
                center={[node.lat, node.lon]}
                radius={isSelected ? 9 : (isOrigin || isDest || isWarnedInFuture ? 9 : 6)}
                pathOptions={{
                  fillColor: isOrigin ? '#00ffcc' : (isDest ? '#ff0055' : (isWarnedInFuture ? '#ef4444' : (node.color || '#2ecc71'))),
                  color: isWarnedInFuture ? '#a855f7' : (isSelected || isOrigin || isDest ? '#ffffff' : '#1e293b'),
                  weight: isWarnedInFuture ? 4 : (isSelected || isOrigin || isDest ? 3 : 1),
                  opacity: 0.95,
                  fillOpacity: isWarnedInFuture ? 1.0 : 0.85
                }}
                eventHandlers={{
                  click: () => {
                    setSelectedNodeId(node.id);
                    runLlmQuery(`Which way to avoid & use if starting now for ${node.location_label || ('Sensor #' + node.sensor_id)}?`);
                  }
                }}
              >
              <Popup>
                <div style={{ fontSize: '11px', color: '#0f172a', padding: '2px' }}>
                  <strong style={{ color: '#0284c7' }}>Sensor #{node.sensor_id || node.id}</strong><br/>
                  <span>{node.location_label || 'Mapped Highway Segment'}</span><br/>
                  Speed: <strong>{node.speed} mph</strong> ({node.status})<br/>
                  
                  {/* Click-to-set Origin & Destination Buttons */}
                  <div style={{ display: 'flex', gap: '4px', marginTop: '6px' }}>
                    <button 
                      onClick={() => {
                        setOriginNodeId(node.id);
                        calculateSmartRoute(node.id, destinationNodeId);
                      }}
                      style={{ background: '#00ffcc', color: '#0f172a', border: 'none', borderRadius: '4px', padding: '3px 6px', fontSize: '10px', fontWeight: 800, cursor: 'pointer' }}
                    >
                      🟢 Set as Origin
                    </button>
                    <button 
                      onClick={() => {
                        setDestinationNodeId(node.id);
                        calculateSmartRoute(originNodeId, node.id);
                      }}
                      style={{ background: '#ff0055', color: '#ffffff', border: 'none', borderRadius: '4px', padding: '3px 6px', fontSize: '10px', fontWeight: 800, cursor: 'pointer' }}
                    >
                      🔴 Set as Destination
                    </button>
                  </div>
                </div>
              </Popup>
            </CircleMarker>
            );
          })}
        </MapContainer>
        
        {/* Map Legend Floating */}
        <div className={styles.mapLegend}>
          <div className={styles.legendTitle}>Visual Map Legend</div>
          <div className={styles.legendItem}>
            <span className={styles.dot} style={{ backgroundColor: '#2ecc71' }}></span>
            <span>Free-Flow Traffic (&ge; 50 mph)</span>
          </div>
          <div className={styles.legendItem}>
            <span className={styles.dot} style={{ backgroundColor: '#f1c40f' }}></span>
            <span>Moderate Congestion (25 - 49 mph)</span>
          </div>
          <div className={styles.legendItem}>
            <span className={styles.dot} style={{ backgroundColor: '#e74c3c' }}></span>
            <span>Severe Bottleneck Jam (&lt; 25 mph)</span>
          </div>
          <div className={styles.legendItem}>
            <span className={styles.dot} style={{ backgroundColor: '#38bdf8' }}></span>
            <span>Zero-Flow / Off-Peak (&lt; 1 mph)</span>
          </div>
          
          {/* Edge Color Highlights Legend */}
          <div style={{ marginTop: '8px', borderTop: '1px solid #334155', paddingTop: '6px' }}>
            <div className={styles.legendItem}>
              <span style={{ display: 'inline-block', width: '16px', height: '4px', backgroundColor: '#00ffcc', marginRight: '6px' }}></span>
              <span style={{ color: '#00ffcc', fontWeight: 700 }}>Optimal Route Path</span>
            </div>
            <div className={styles.legendItem}>
              <span style={{ display: 'inline-block', width: '16px', height: '4px', backgroundColor: '#ff0055', marginRight: '6px' }}></span>
              <span style={{ color: '#ff0055', fontWeight: 700 }}>Congested Link to Avoid</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Control Drawer */}
      <div className={`${styles.sidebar} ${!isRightSidebarOpen ? styles.sidebarClosed : ''}`}>
        
        {/* FEATURE 1: Smart Mapped Sensor Origin, Destination & Arrival Time Route Planner Card */}
        <div className={styles.card} style={{ borderColor: '#00ffcc', background: 'linear-gradient(180deg, #0f172a, #0b253a)' }}>
          <div className={styles.cardTitle} style={{ color: '#00ffcc' }}>
            <Navigation size={16} color="#00ffcc" />
            <span>Mapped Sensor Route Planner</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div>
              <label style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 700 }}>Starting Sensor (Origin):</label>
              <select 
                value={originNodeId} 
                onChange={(e) => setOriginNodeId(parseInt(e.target.value))}
                className={styles.headerSelect}
                style={{ width: '100%', fontSize: '11px', marginTop: '2px' }}
              >
                {nodes.map(n => (
                  <option key={n.id} value={n.id}>
                    Sensor #{n.sensor_id || n.id} — {n.location_label || `Corridor Node #${n.id}`} ({n.speed} mph)
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 700 }}>Destination Sensor:</label>
              <select 
                value={destinationNodeId} 
                onChange={(e) => setDestinationNodeId(parseInt(e.target.value))}
                className={styles.headerSelect}
                style={{ width: '100%', fontSize: '11px', marginTop: '2px' }}
              >
                {nodes.map(n => (
                  <option key={n.id} value={n.id}>
                    Sensor #{n.sensor_id || n.id} — {n.location_label || `Corridor Node #${n.id}`} ({n.speed} mph)
                  </option>
                ))}
              </select>
            </div>

            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <div style={{ flex: 1 }}>
                <label style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 700 }}>Target Arrival Time:</label>
                <input 
                  type="text" 
                  value={targetArrivalTime} 
                  onChange={(e) => setTargetArrivalTime(e.target.value)}
                  className={styles.headerSelect}
                  style={{ width: '100%', fontSize: '11px', marginTop: '2px' }}
                />
              </div>
              <button 
                onClick={() => calculateSmartRoute()}
                className={styles.btn}
                style={{ backgroundColor: '#00ffcc', color: '#0f172a', fontWeight: 800, marginTop: '14px', flex: 1 }}
                disabled={isRouting}
              >
                📍 Find Best Route
              </button>
            </div>
          </div>

          {routeResult && (
            <div className={styles.logBox} style={{ borderColor: '#00ffcc', marginTop: '10px' }}>
              <div style={{ fontSize: '11px', color: '#00ffcc', fontWeight: 800 }}>
                ✅ Route Calculated ({routeResult.target_arrival_time})
              </div>
              <div style={{ fontSize: '11px', color: '#f8fafc', marginTop: '4px' }}>
                • Departure: <strong>{routeResult.recommended_departure_time}</strong><br/>
                • Travel Time: <strong>{routeResult.estimated_travel_time_mins} mins</strong><br/>
                • Time Saved: <strong style={{ color: '#34d399' }}>{routeResult.estimated_time_saved_mins} mins!</strong>
              </div>
            </div>
          )}
        </div>

        {/* FEATURE 2: 15-Minute Congestion Warning Alerts */}
        <div className={styles.card} style={{ borderColor: 'rgba(239, 68, 68, 0.5)' }}>
          <div className={styles.cardTitle} style={{ color: '#ef4444' }}>
            <AlertTriangle size={15} color="#ef4444" />
            <span>15-Min Upcoming Bottleneck Alerts ({getDisplayTime(step)})</span>
          </div>

          {upcoming15MinWarnings.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '120px', overflowY: 'auto' }}>
              {upcoming15MinWarnings.map((w, idx) => (
                <div key={idx} style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '6px 8px', borderRadius: '6px', fontSize: '11px', color: '#f8fafc' }}>
                  🚨 <strong>{w.location_label}</strong><br/>
                  <span style={{ color: '#ef4444' }}>Predicted Drop: {w.predicted_speed} mph in 15 mins</span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ fontSize: '11px', color: '#94a3b8', fontStyle: 'italic' }}>
              No severe bottleneck spikes predicted for the next 15 minutes.
            </div>
          )}
        </div>

        {/* Replay & Calendar Engine Card */}
        <div className={`${styles.card} ${styles.cardPrimary}`}>
          <div className={`${styles.cardTitle} ${styles.cardTitlePrimary}`}>
            <Calendar size={14} color="#38bdf8" />
            <span>Click to Open Calendar (2012 Period):</span>
          </div>
          
          <div className={styles.controlsGroup}>
            <input 
              type="date" 
              value={date} 
              onChange={(e) => setDate(e.target.value)}
              className={styles.headerSelect}
              style={{ width: '100%', marginBottom: '8px' }}
            />

            <div className={styles.timeDisplay}>
              <span>Time: <span className={styles.timeText}>{getDisplayTime(step)}</span></span>
              <span className={styles.stepText}>(Step {step}/288)</span>
            </div>

            <input 
              type="range" 
              min="0" 
              max="287" 
              value={step} 
              onChange={(e) => setStep(parseInt(e.target.value))}
              className={styles.timeSlider}
            />

            <div className={styles.buttonGroup}>
              {!isPlaying ? (
                <button onClick={() => setIsPlaying(true)} className={`${styles.btn} ${styles.btnPlay}`}>
                  <Play size={14} /> Play
                </button>
              ) : (
                <button onClick={() => setIsPlaying(false)} className={`${styles.btn} ${styles.btnPause}`}>
                  <Pause size={14} /> Pause
                </button>
              )}
              <button onClick={() => { setIsPlaying(false); setStep(96); }} className={`${styles.btn} ${styles.btnReset}`}>
                <RotateCcw size={14} /> Reset
              </button>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '8px' }}>
              <span style={{ fontSize: '11px', color: '#94a3b8', fontWeight: 600 }}>Playback Speed:</span>
              <select 
                value={speedMultiplier} 
                onChange={(e) => setSpeedMultiplier(parseInt(e.target.value))}
                className={styles.headerSelect}
                style={{ fontSize: '11px', padding: '3px 6px' }}
              >
                <option value={1}>1x Speed</option>
                <option value={5}>5x Speed</option>
                <option value={10}>10x Speed</option>
                <option value={60}>60x Speed</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
