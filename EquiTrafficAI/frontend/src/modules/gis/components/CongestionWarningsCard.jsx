import React from 'react';
import { AlertTriangle } from 'lucide-react';
import styles from '../MapView.module.css';

const CongestionWarningsCard = ({ upcoming15MinWarnings, getDisplayTime, step }) => {
  return (
    <div className={`${styles.card} ui-card-danger`}>
      <div className={`${styles.cardTitle} text-rose`}>
        <AlertTriangle size={15} color="#ef4444" />
        <span>15-Min Upcoming Bottleneck Alerts ({getDisplayTime(step)})</span>
      </div>

      {upcoming15MinWarnings.length > 0 ? (
        <div className="ui-scroll-list">
          {upcoming15MinWarnings.map((w, idx) => (
            <div key={idx} className="ui-badge-danger">
              🚨 <strong>{w.location_label}</strong><br/>
              <span className="text-rose">Predicted Drop: {w.predicted_speed} mph in 15 mins</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="ui-section-desc">
          ✅ All freeway corridors free-flowing for next 15 mins.
        </div>
      )}
    </div>
  );
};

export default CongestionWarningsCard;
