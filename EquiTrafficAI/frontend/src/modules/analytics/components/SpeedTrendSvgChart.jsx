import React from 'react';
import { TrendingUp, Activity, Play, Pause } from 'lucide-react';

const SpeedTrendSvgChart = ({
  step,
  setStep,
  isPlaying,
  setIsPlaying,
  formatTime,
  selectedCity,
  generateSpeedSvgPoints,
  MAX_STEPS = 288
}) => {
  return (
    <>
      {/* 24-Hour Telemetry Playback Scrubber Card */}
      <div className="ui-card-panel">
        <div className="ui-header-flex">
          <h2 className="ui-section-title text-cyan">
            <Activity size={18} /> 24-Hour Telemetry & GWNet 15-min Forecast Playback
          </h2>
          <span className="ui-time-badge">
            Current: {formatTime(step)} <span className="ui-metric-unit">(Step {step}/288)</span>
          </span>
        </div>

        <input 
          type="range" 
          min="0" 
          max={MAX_STEPS - 1} 
          value={step} 
          onChange={(e) => setStep(parseInt(e.target.value))}
          className="ui-range-slider"
          aria-label="Timeline Scrubber Slider"
        />

        <div className="ui-legend-item">
          <button 
            onClick={() => setIsPlaying(!isPlaying)}
            className={isPlaying ? 'ui-btn-pause' : 'ui-btn-play'}
          >
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
            {isPlaying ? 'Pause Playback' : 'Play 24-Hour Cycle'}
          </button>
          <span className="ui-metric-subtext">
            Scrub timeline slider to evaluate peak rush hour speed drops and GNN forecast convergence.
          </span>
        </div>
      </div>

      {/* Network Speed Trend SVG Chart */}
      <div className="ui-card-panel">
        <div className="ui-header-flex">
          <h2 className="ui-section-title text-primary">
            <TrendingUp size={18} color="#38bdf8" /> Network Speed Curve vs. Graph WaveNet 15-min Forecast
          </h2>

          <div className="ui-legend-flex">
            <span className="ui-legend-item text-cyan">
              <span className="ui-legend-line-cyan"></span> Ground Truth
            </span>
            <span className="ui-legend-item text-purple">
              <span className="ui-legend-line-purple"></span> GWNet 15-min Forecast
            </span>
          </div>
        </div>

        <div className="ui-chart-box">
          <svg viewBox="0 0 780 260" className="ui-svg-canvas">
            <line x1="40" y1="40" x2="750" y2="40" stroke="#334155" strokeDasharray="4" />
            <line x1="40" y1="100" x2="750" y2="100" stroke="#334155" strokeDasharray="4" />
            <line x1="40" y1="160" x2="750" y2="160" stroke="#334155" strokeDasharray="4" />
            <line x1="40" y1="220" x2="750" y2="220" stroke="#334155" strokeLinecap="round" />

            <text x="10" y="45" fill="#94a3b8" fontSize="11">60 mph</text>
            <text x="10" y="105" fill="#94a3b8" fontSize="11">50 mph</text>
            <text x="10" y="165" fill="#94a3b8" fontSize="11">40 mph</text>
            <text x="10" y="225" fill="#94a3b8" fontSize="11">30 mph</text>

            <polyline
              fill="none"
              stroke="#38bdf8"
              strokeWidth="3"
              points={generateSpeedSvgPoints(false)}
            />

            <polyline
              fill="none"
              stroke="#c084fc"
              strokeWidth="2"
              strokeDasharray="5 3"
              points={generateSpeedSvgPoints(true)}
            />

            <line 
              x1={(step / 288) * 700 + 40} 
              y1="20" 
              x2={(step / 288) * 700 + 40} 
              y2="230" 
              stroke="#f1c40f" 
              strokeWidth="2" 
              strokeDasharray="4" 
            />
            <circle cx={(step / 288) * 700 + 40} cy={220 - (((62.0 - (Math.exp(-Math.pow((step*5/60.0) - 8.0, 2) / 4.0) + Math.exp(-Math.pow((step*5/60.0) - 17.5, 2) / 4.0)) * 30.0) - 20) / 50) * 180} r="6" fill="#f1c40f" />

            <text x="40" y="248" fill="#94a3b8" fontSize="11">12 AM</text>
            <text x="215" y="248" fill="#94a3b8" fontSize="11">06 AM (Morning Rush)</text>
            <text x="390" y="248" fill="#94a3b8" fontSize="11">12 PM</text>
            <text x="565" y="248" fill="#94a3b8" fontSize="11">06 PM (Evening Rush)</text>
            <text x="720" y="248" fill="#94a3b8" fontSize="11">11 PM</text>
          </svg>
        </div>
      </div>
    </>
  );
};

export default React.memo(SpeedTrendSvgChart);
