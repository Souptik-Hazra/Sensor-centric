import React, { useEffect } from 'react';
import useTrafficStore from '../../store/useTrafficStore';
import styles from './MonitoringView.module.css';

const MonitoringView = () => {
  const { sensors, trafficData, initializeData, currentTimestampIndex } = useTrafficStore();

  useEffect(() => {
    if (sensors.length === 0) {
      initializeData();
    }
  }, [sensors.length, initializeData]);

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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 className={styles.title}>Real-time Sensor Monitoring</h2>
        <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
          Time Index: {currentTimestampIndex}
        </span>
      </div>
      
      <div className={styles.tableContainer}>
        <table className={styles.table}>
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
              const data = trafficData.find(d => String(d.sensor_id) === String(sensor.sensor_id));
              const currentSpeed = data?.speed || sensor.speed || 58.5;
              const statusType = data?.status || (currentSpeed >= 50 ? 'fast' : currentSpeed >= 25 ? 'medium' : 'slow');
              const locationName = sensor.location_label || sensor.name || `Corridor Sensor #${sensor.sensor_id}`;

              return (
                <tr key={sensor.sensor_id}>
                  <td style={{ fontWeight: 600, color: '#38bdf8' }}>{sensor.sensor_id}</td>
                  <td>{locationName}</td>
                  <td style={{ fontWeight: 700 }}>{currentSpeed.toFixed(1)} mph</td>
                  <td>{getStatusBadge(statusType)}</td>
                  <td style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                    {(sensor.lat || 34.05).toFixed(4)}, {(sensor.lon || sensor.lng || -118.24).toFixed(4)}
                  </td>
                </tr>
              );
            })}
            {sensors.length === 0 && (
              <tr>
                <td colSpan="5" style={{ textAlign: 'center', padding: '24px', color: 'var(--text-secondary)' }}>
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

export default MonitoringView;
