import React from 'react';
import { CircleMarker, Pane, Popup } from 'react-leaflet';
import styles from '../MapView.module.css';

const MapMarkerLayer = ({
  nodes = [],
  isFutureVisionActive,
  upcoming15MinWarnings = [],
  futurePredictedSpeeds = {},
  originNodeId,
  destinationNodeId,
  selectedNodeId,
  setSelectedNodeId,
  runLlmQuery,
  setOriginNodeId,
  setDestinationNodeId,
  calculateSmartRoute
}) => {
  return (
    <Pane name="trafficNodes" style={{ zIndex: 420 }}>
      {nodes.map(node => {
        const isWarnedInFuture = isFutureVisionActive && upcoming15MinWarnings.some((warning) => {
          const warningNodeId = warning.id ?? warning.node_id;
          const warningSensorId = warning.sensor_id;
          return (warningNodeId != null && String(warningNodeId) === String(node.id))
            || (warningSensorId != null && String(warningSensorId) === String(node.sensor_id));
        });
        const predictedSpeed = Number(futurePredictedSpeeds[String(node.id)] ?? futurePredictedSpeeds[String(node.sensor_id)]);
        const displaySpeed = isFutureVisionActive && Number.isFinite(predictedSpeed) ? predictedSpeed : node.speed;
        const displayStatus = displaySpeed < 25 ? 'Congested' : displaySpeed < 50 ? 'Moderate' : 'Clear';
        const futureColor = displaySpeed < 25 ? '#e74c3c' : displaySpeed < 50 ? '#f1c40f' : '#2ecc71';
        const isOrigin = originNodeId === node.id;
        const isDest = destinationNodeId === node.id;
        const isSelected = selectedNodeId === node.id;

        const markerRadius = isSelected || isOrigin || isDest || isWarnedInFuture ? 9 : 6;
        const markerFill = isOrigin ? '#00ffcc' : isDest ? '#ff0055' : isFutureVisionActive && Number.isFinite(predictedSpeed) ? futureColor : (node.color || '#2ecc71');
        const markerBorder = isWarnedInFuture ? '#a855f7' : (isSelected || isOrigin || isDest ? '#ffffff' : '#1e293b');
        const markerWeight = isWarnedInFuture ? 4 : (isSelected || isOrigin || isDest ? 3 : 1);

        return (
          <CircleMarker
            key={`${node.id}-${node.color}-${displaySpeed}-${isWarnedInFuture}`}
            center={[node.lat, node.lon]}
            radius={markerRadius}
            pathOptions={{
              fillColor: markerFill,
              color: markerBorder,
              weight: markerWeight,
              opacity: 0.95,
              fillOpacity: isWarnedInFuture ? 1.0 : 0.85
            }}
            eventHandlers={{
              click: () => {
                setSelectedNodeId(node.id);
                runLlmQuery(`Which way to avoid & use if starting now for Node #${node.id}, Sensor #${node.sensor_id || 'unknown'} (${node.location_label || 'this corridor'})?`);
              }
            }}
          >
            <Popup>
              <div className={styles.popupCard}>
                <strong className={styles.popupTitle}>Node #{node.id} / Sensor #{node.sensor_id || 'unknown'}</strong><br/>
                <span>{node.location_label || 'Mapped Highway Segment'}</span><br/>
                Speed: <strong>{displaySpeed.toFixed(1)} mph</strong> ({isFutureVisionActive ? `${displayStatus}, forecast` : node.status})<br/>
                
                <div className={styles.popupBtnGroup}>
                  <button 
                    type="button"
                    onMouseDown={(event) => event.stopPropagation()}
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      event.nativeEvent.stopImmediatePropagation();
                      setOriginNodeId(node.id);
                      let targetDest = destinationNodeId;
                      if (node.id === destinationNodeId) {
                        targetDest = nodes.find(n => n.id !== node.id)?.id || (node.id + 1) % nodes.length;
                        setDestinationNodeId(targetDest);
                      }
                      calculateSmartRoute(node.id, targetDest);
                    }}
                    className={styles.popupBtnOrigin}
                  >
                    🟢 Set as Origin
                  </button>
                  <button 
                    type="button"
                    onMouseDown={(event) => event.stopPropagation()}
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      event.nativeEvent.stopImmediatePropagation();
                      setDestinationNodeId(node.id);
                      let targetOrigin = originNodeId;
                      if (node.id === originNodeId) {
                        targetOrigin = nodes.find(n => n.id !== node.id)?.id || (node.id + 1) % nodes.length;
                        setOriginNodeId(targetOrigin);
                      }
                      calculateSmartRoute(targetOrigin, node.id);
                    }}
                    className={styles.popupBtnDest}
                  >
                    🔴 Set as Destination
                  </button>
                </div>
              </div>
            </Popup>
          </CircleMarker>
        );
      })}
    </Pane>
  );
};

export default React.memo(MapMarkerLayer);
