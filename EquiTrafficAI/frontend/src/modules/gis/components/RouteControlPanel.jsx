import React from 'react';
import { Navigation, Clock } from 'lucide-react';
import styles from '../MapView.module.css';

const RouteControlPanel = ({
  nodes = [],
  originNodeId,
  setOriginNodeId,
  destinationNodeId,
  setDestinationNodeId,
  targetArrivalTime,
  setTargetArrivalTime,
  calculateSmartRoute,
  isRouting,
  routeResult
}) => {
  return (
    <div className={`${styles.card} ui-card-cyan`}>
      <div className={`${styles.cardTitle} text-cyan`}>
        <Navigation size={16} color="#00ffcc" />
        <span>Mapped Sensor Route Planner</span>
      </div>

      <div className="flex-col-gap8">
        <div>
          <label htmlFor="select-route-origin" className="ui-label-sm">Starting Sensor (Origin):</label>
          <select 
            id="select-route-origin"
            value={originNodeId} 
            onChange={(e) => {
              const newOrigin = parseInt(e.target.value);
              setOriginNodeId(newOrigin);
              if (newOrigin === destinationNodeId) {
                const altDest = nodes.find(n => n.id !== newOrigin)?.id || (newOrigin + 1) % nodes.length;
                setDestinationNodeId(altDest);
              }
            }}
            className="ui-select-dark mt-2"
            aria-label="Select Route Starting Sensor"
          >
            {nodes.map(n => (
              <option key={n.id} value={n.id}>
                Sensor #{n.sensor_id || n.id} — {n.location_label || `Corridor Node #${n.id}`} ({n.speed} mph)
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="select-route-dest" className="ui-label-sm">Destination Sensor:</label>
          <select 
            id="select-route-dest"
            value={destinationNodeId} 
            onChange={(e) => {
              const newDest = parseInt(e.target.value);
              setDestinationNodeId(newDest);
              if (newDest === originNodeId) {
                const altOrigin = nodes.find(n => n.id !== newDest)?.id || (newDest + 1) % nodes.length;
                setOriginNodeId(altOrigin);
              }
            }}
            className="ui-select-dark mt-2"
            aria-label="Select Route Destination Sensor"
          >
            {nodes.map(n => (
              <option key={n.id} value={n.id}>
                Sensor #{n.sensor_id || n.id} — {n.location_label || `Corridor Node #${n.id}`} ({n.speed} mph)
              </option>
            ))}
          </select>
        </div>

        <div className="flex-row-gap8">
          <div className="flex-1">
            <label className="ui-label-sm">Target Arrival Time:</label>
            <input 
              type="text" 
              value={targetArrivalTime} 
              onChange={(e) => setTargetArrivalTime(e.target.value)}
              className="ui-select-dark mt-2"
            />
          </div>
          <button 
            onClick={() => calculateSmartRoute()}
            className="ui-btn-cyan flex-1 mt-14"
            disabled={isRouting}
          >
            📍 Find Best Route
          </button>
        </div>

        {routeResult && routeResult.primary_route && (
          <div className="ui-result-card">
            <div className="text-cyan mb-4 flex-row-gap8">
              <Clock size={13} /> {routeResult.recommended_departure_time || 'Depart in 5 mins'}
            </div>
            <div className="ui-pareto-stat">Distance: <strong>{routeResult.primary_route.distance_miles ?? 4.2} miles</strong></div>
            <div className="ui-pareto-stat">Est. Travel Time: <strong>{routeResult.primary_route.travel_time_minutes ?? 14} mins</strong></div>
            <div className="ui-text-savings">
              {routeResult.recommended_alternate_route?.estimated_time_saved_minutes > 0 ? (
                <>⚡ {routeResult.time_saved_msg || `Saves ${routeResult.recommended_alternate_route.estimated_time_saved_minutes} mins by avoiding bottleneck links!`}</>
              ) : (
                <span className="text-emerald">✨ Route is clear. No alternate reroute needed!</span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RouteControlPanel;
