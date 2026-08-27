import React, { useEffect, useMemo } from 'react';
import useTrafficStore from '../../store/useTrafficStore';
import styles from './MonitoringView.module.css';

const MonitoringView = () => {
  // Optimized Atomic Zustand Selectors to prevent unnecessary component re-renders
  const sensors = useTrafficStore((state) => state.sensors);
  const trafficData = useTrafficStore((state) => state.trafficData);
  const currentTimestampIndex = useTrafficStore((state) => state.currentTimestampIndex);
  const initializeData = useTrafficStore((state) => state.initializeData);

  useEffect(() => {
    if (sensors.length === 0) {
      initializeData();
    }
  }, [sensors.length, initializeData]);

  // Memoized speed lookup map for O(1) rendering speed
  const speedMap = useMemo(() => {
    const map = new Map();
    trafficData.forEach((d) => map.set(String(d.sensor_id), d));
    return map;
  }, [trafficData]);

  const getStatusBadge = (status) => {
    switch (status) {
      case 'fast': return <span className={`${styles.badge} ${styles.fast}`}>Clear</span>;
      case 'medium': return <span className={`${styles.badge} ${styles.medium}`}>Moderate</span>;
      case 'slow': return <span className={`${styles.badge} ${styles.slow}`}>Congested</span>;
      default: return <span className={styles.badge}>Unknown</span>;
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.headerRow}>
        <h2 className={styles.title}>Real-time Sensor Monitoring</h2>
        <span className={styles.timeIndex}>
          Time Index: {currentTimestampIndex}
        </span>
      </div>
      
      <div className={styles.tableContainer}>
        <table className={styles.table} aria-label="Real-time Highway Sensor Telemetry Table">
          <thead>
            <tr>
              <th>Sensor ID</th>
              <th>Location Name</th>
              <th>Current Speed</th>
              <th>Status</th>
              <th>Coordinates</th>
            </tr>
          </thead>
          <tbody>
            {sensors.map(sensor => {
              const data = speedMap.get(String(sensor.sensor_id));
              const currentSpeed = data?.speed || sensor.speed || 58.5;
              const statusType = data?.status || (currentSpeed >= 50 ? 'fast' : currentSpeed >= 25 ? 'medium' : 'slow');
              const locationName = sensor.location_label || sensor.name || `Corridor Sensor #${sensor.sensor_id}`;

              return (
                <tr key={sensor.sensor_id}>
                  <td className={styles.sensorIdCell}>{sensor.sensor_id}</td>
                  <td>{locationName}</td>
                  <td className={styles.speedCell}>{currentSpeed.toFixed(1)} mph</td>
                  <td>{getStatusBadge(statusType)}</td>
                  <td className={styles.coordCell}>
                    {(sensor.lat || 34.05).toFixed(4)}, {(sensor.lon || sensor.lng || -118.24).toFixed(4)}
                  </td>
                </tr>
              );
            })}
            {sensors.length === 0 && (
              <tr>
                <td colSpan="5" className={styles.loadingCell}>
                  Loading sensor data...
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default React.memo(MonitoringView);
