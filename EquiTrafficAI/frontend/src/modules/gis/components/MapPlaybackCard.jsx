import React from 'react';
import { Calendar, Play, Pause, RotateCcw } from 'lucide-react';
import styles from '../MapView.module.css';

const MapPlaybackCard = ({
  date,
  setDate,
  step,
  setStep,
  isPlaying,
  setIsPlaying,
  getDisplayTime,
  speedMultiplier,
  setSpeedMultiplier
}) => {
  return (
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
          className="ui-select-dark mb-8"
          aria-label="Select Historical Date"
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
          aria-label="Map Timeline Scrubber Slider"
        />

        <div className={styles.buttonGroup}>
          {!isPlaying ? (
            <button onClick={() => setIsPlaying(true)} className={`${styles.btn} ${styles.btnPlay}`} aria-label="Play 24-Hour Cycle">
              <Play size={14} /> Play
            </button>
          ) : (
            <button onClick={() => setIsPlaying(false)} className={`${styles.btn} ${styles.btnPause}`} aria-label="Pause 24-Hour Cycle">
              <Pause size={14} /> Pause
            </button>
          )}
          <button onClick={() => { setIsPlaying(false); setStep(96); }} className={`${styles.btn} ${styles.btnReset}`} aria-label="Reset Timeline to 08:00 AM">
            <RotateCcw size={14} /> Reset
          </button>
        </div>

        <div className="flex-between mt-8">
          <span className="ui-label-sm">Playback Speed:</span>
          <select 
            value={speedMultiplier} 
            onChange={(e) => setSpeedMultiplier(parseInt(e.target.value))}
            className="ui-select-dark"
            aria-label="Select Playback Speed"
          >
            <option value={1}>1x Speed</option>
            <option value={5}>5x Speed</option>
            <option value={10}>10x Speed</option>
            <option value={60}>60x Speed</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default React.memo(MapPlaybackCard);
