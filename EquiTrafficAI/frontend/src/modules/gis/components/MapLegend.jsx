import React from 'react';
import styles from '../MapView.module.css';

const MapLegend = () => {
  return (
    <div className={styles.mapLegend}>
      <div className={styles.legendTitle}>Visual Map Legend</div>
      <div className={styles.legendItem}>
        <span className={`${styles.dot} ${styles.dotGreen}`}></span>
        <span>Free-Flow Traffic (&ge; 50 mph)</span>
      </div>
      <div className={styles.legendItem}>
        <span className={`${styles.dot} ${styles.dotYellow}`}></span>
        <span>Moderate Congestion (25 - 49 mph)</span>
      </div>
      <div className={styles.legendItem}>
        <span className={`${styles.dot} ${styles.dotRed}`}></span>
        <span>Severe Bottleneck Jam (&lt; 25 mph)</span>
      </div>
      <div className={styles.legendItem}>
        <span className={`${styles.dot} ${styles.dotBlue}`}></span>
        <span>Zero-Flow / Off-Peak (&lt; 1 mph)</span>
      </div>
      
      {/* Edge Color Highlights Legend */}
      <div className={styles.edgeLegendDivider}>
        <div className={styles.legendItem}>
          <span className={styles.routeLineCyan}></span>
          <span className="text-cyan">Optimal Route Path</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.routeLineRed}></span>
          <span className="text-rose">Congested Link to Avoid</span>
        </div>
      </div>
    </div>
  );
};

export default MapLegend;
