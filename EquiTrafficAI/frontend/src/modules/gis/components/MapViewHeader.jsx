import React from 'react';
import { Globe, Eye, ChevronRight, Sliders } from 'lucide-react';
import styles from '../MapView.module.css';

const MapViewHeader = ({
  isRightSidebarOpen,
  selectedCity,
  setSelectedCity,
  isFutureVisionActive,
  setIsFutureVisionActive,
  setIsRightSidebarOpen
}) => {
  return (
    <div className={`${styles.mapHeaderOverlay} ${!isRightSidebarOpen ? styles.mapHeaderOverlayFull : ''}`}>
      <div className={styles.singleHeaderBar}>
        <Globe size={15} color="#38bdf8" />
        <span className={styles.headerTitle}>EquiTraffic Console:</span>
        
        <select 
          value={selectedCity} 
          onChange={(e) => setSelectedCity(e.target.value)}
          className="ui-select-dark"
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
          className={isFutureVisionActive ? 'ui-btn-purple-active' : 'ui-btn-purple'}
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
  );
};

export default MapViewHeader;
