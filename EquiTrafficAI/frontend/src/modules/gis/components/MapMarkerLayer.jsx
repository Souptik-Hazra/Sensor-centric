import React from 'react';
import { CircleMarker, Popup } from 'react-leaflet';
import styles from '../MapView.module.css';

const MapMarkerLayer = ({
  nodes = [],
  isFutureVisionActive,
  upcoming15MinWarnings = [],
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
    <>
      {nodes.map(node => {
        const isWarnedInFuture = isFutureVisionActive && upcoming15MinWarnings.some(w => w.id === node.id || w.sensor_id === node.sensor_id);
        const isOrigin = originNodeId === node.id;
        const isDest = destinationNodeId === node.id;
        const isSelected = selectedNodeId === node.id;

        const markerRadius = isSelected || isOrigin || isDest || isWarnedInFuture ? 9 : 6;
        const markerFill = isOrigin ? '#00ffcc' : isDest ? '#ff0055' : isWarnedInFuture ? '#ef4444' : (node.color || '#2ecc71');
        const markerBorder = isWarnedInFuture ? '#a855f7' : (isSelected || isOrigin || isDest ? '#ffffff' : '#1e293b');
        const markerWeight = isWarnedInFuture ? 4 : (isSelected || isOrigin || isDest ? 3 : 1);

        return (
          <CircleMarker
            key={`${node.id}-${node.color}-${isWarnedInFuture}`}
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
                runLlmQuery(`Which way to avoid & use if starting now for ${node.location_label || ('Sensor #' + node.sensor_id)}?`);
              }
            }}
          >
            <Popup>
              <div className={styles.popupCard}>
                <strong className={styles.popupTitle}>Sensor #{node.sensor_id || node.id}</strong><br/>
                <span>{node.location_label || 'Mapped Highway Segment'}</span><br/>
                Speed: <strong>{node.speed} mph</strong> ({node.status})<br/>
                
                <div className={styles.popupBtnGroup}>
                  <button 
                    onClick={() => {
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
                    onClick={() => {
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
    </>
  );
};

export default React.memo(MapMarkerLayer);
