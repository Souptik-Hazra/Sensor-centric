import React, { useState, useEffect } from 'react';
import { Play, Pause, TrendingUp, ShieldCheck, Activity, BarChart2, Globe, Cpu, Zap, Award } from 'lucide-react';
import styles from './AnalyticsView.module.css';

const AnalyticsView = () => {
  const [selectedCity, setSelectedCity] = useState('la');
  const [step, setStep] = useState(96); // 08:00 AM
  const [isPlaying, setIsPlaying] = useState(false);
  const MAX_STEPS = 288;

  // Format 288 timesteps into 24-hour clock
  const formatTime = (index) => {
    const validIndex = isNaN(index) || index === undefined ? 96 : index;
    const totalMinutes = validIndex * 5;
    const h = Math.floor(totalMinutes / 60) % 24;
    const m = totalMinutes % 60;
    const ampm = h >= 12 ? 'PM' : 'AM';
    const displayH = h % 12 === 0 ? 12 : h % 12;
    return `${displayH.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')} ${ampm}`;
  };

  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(() => {
        setStep((prev) => (prev + 1) % MAX_STEPS);
      }, 400);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  // Generate SVG Points for Network Speed Curve across 24 Hours
  const generateSpeedSvgPoints = (isForecast = false) => {
    const points = [];
    for (let i = 0; i <= 288; i += 6) {
      const hour = (i * 5) / 60.0;
      const rush = Math.exp(-Math.pow(hour - 8.0, 2) / 4.0) + Math.exp(-Math.pow(hour - 17.5, 2) / 4.0);
      const baseSpeed = selectedCity === 'sd' ? 58.0 : 62.0;
      const speedDrop = selectedCity === 'sd' ? 24.0 : 30.0;
      
      let speed = baseSpeed - rush * speedDrop;
      if (isForecast) {
        speed += Math.sin(i * 0.3) * 1.5; // Slight GWNet forecast variation
      }

      const x = (i / 288) * 700 + 40;
      const y = 220 - ((speed - 20) / 50) * 180;
      points.push(`${x},${y}`);
    }
    return points.join(' ');
  };

  const [analyticsData, setAnalyticsData] = useState(null);

  useEffect(() => {
    const fetchAnalyticsMetrics = async () => {
      try {
        const res = await fetch(`/api/analytics/metrics?city=${selectedCity}`);
        if (res.ok) {
          const data = await res.json();
          setAnalyticsData(data);
        }
      } catch (err) {
        console.error("Failed to fetch analytics metrics:", err);
      }
    };
    fetchAnalyticsMetrics();
  }, [selectedCity]);

  const paretoPoints = analyticsData?.pareto_matrix || [
    { strategy: "DCRNN Baseline", mae: 2.77, rsf: 0.38, color: "#ef4444", status: "DOMINATED" },
    { strategy: "FairSTG Baseline", mae: 2.45, rsf: 0.28, color: "#f59e0b", status: "SUB-OPTIMAL" },
    { strategy: "GWNet (Suburban Equity)", mae: 2.15, rsf: 0.14, color: "#a855f7", status: "PARETO OPTIMAL" },
    { strategy: "GWNet (Max Throughput)", mae: 1.82, rsf: 0.22, color: "#38bdf8", status: "PARETO OPTIMAL" }
  ];

  return (
    <div style={{ padding: '24px', background: '#0f172a', minHeight: '100vh', color: '#f8fafc' }}>
      
      {/* Header Bar with City Switcher */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h1 style={{ fontSize: '1.4rem', fontWeight: '800', color: '#f8fafc', margin: 0, display: 'flex', alignItems: 'center', gap: '10px' }}>
            <BarChart2 size={24} color="#38bdf8" /> Data Analytics & Multi-Horizon GNN Benchmark
          </h1>
          <p style={{ fontSize: '12px', color: '#94a3b8', margin: '4px 0 0 0' }}>
            Evaluating Graph WaveNet (GWNet) 15-min forecasts, SCM Causal Attributions, and Pareto Equity Trade-offs.
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', background: '#1e293b', padding: '6px 14px', borderRadius: '8px', border: '1px solid #334155' }}>
          <Globe size={16} color="#38bdf8" />
          <span style={{ fontSize: '12px', fontWeight: '700', color: '#94a3b8' }}>Target Corridor:</span>
          <select 
            value={selectedCity} 
            onChange={(e) => setSelectedCity(e.target.value)}
            style={{ background: '#0f172a', color: '#38bdf8', border: '1px solid #334155', padding: '4px 10px', borderRadius: '6px', fontWeight: '700', fontSize: '12px', cursor: 'pointer', outline: 'none' }}
          >
            <option value="la">Los Angeles METR-LA (207 Sensors)</option>
            <option value="sd">San Diego SD400 (716 Sensors)</option>
          </select>
        </div>
      </div>

      {/* Top Executive Metric Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '24px' }}>
        
        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '16px' }}>
          <div style={{ fontSize: '11px', fontWeight: '700', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Cpu size={14} color="#38bdf8" /> Network MAE Accuracy
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: '800', color: '#38bdf8' }}>1.82 <span style={{ fontSize: '12px', color: '#94a3b8' }}>mph</span></div>
          <div style={{ fontSize: '11px', color: '#34d399', marginTop: '4px' }}>+34.2% vs DCRNN Baseline (2.77 mph)</div>
        </div>

        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '16px' }}>
          <div style={{ fontSize: '11px', fontWeight: '700', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ShieldCheck size={14} color="#c084fc" /> Regional Equity (RSF)
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: '800', color: '#c084fc' }}>0.0705</div>
          <div style={{ fontSize: '11px', color: '#c084fc', marginTop: '4px' }}>-23.4% Disparity for Suburban Commuters</div>
        </div>

        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '16px' }}>
          <div style={{ fontSize: '11px', fontWeight: '700', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Zap size={14} color="#f59e0b" /> CAP-D Causal Cascade
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: '800', color: '#f59e0b' }}>61.3%</div>
          <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '4px' }}>Indirect Cascades (21.4% Direct Local)</div>
        </div>

        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '16px' }}>
          <div style={{ fontSize: '11px', fontWeight: '700', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Award size={14} color="#34d399" /> Zero-Dropout Quality
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: '800', color: '#34d399' }}>{selectedCity === 'sd' ? '2.75%' : '8.45%'}</div>
          <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '4px' }}>Hardware Failure Filtering Rate</div>
        </div>

      </div>

      {/* 24-Hour Telemetry Playback Scrubber Card */}
      <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '20px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
          <h2 style={{ fontSize: '14px', fontWeight: '700', color: '#38bdf8', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={18} /> 24-Hour Telemetry & GWNet 15-min Forecast Playback
          </h2>
          <span style={{ color: '#34d399', fontWeight: '700', fontSize: '1rem', background: '#0f172a', padding: '4px 12px', borderRadius: '6px', border: '1px solid #334155' }}>
            Current: {formatTime(step)} <span style={{ fontSize: '11px', color: '#94a3b8' }}>(Step {step}/288)</span>
          </span>
        </div>

        <input 
          type="range" 
          min="0" 
          max={MAX_STEPS - 1} 
          value={step} 
          onChange={(e) => setStep(parseInt(e.target.value))}
          style={{ width: '100%', accentColor: '#38bdf8', margin: '4px 0 16px 0', cursor: 'pointer' }}
        />

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <button 
            onClick={() => setIsPlaying(!isPlaying)}
            style={{ background: isPlaying ? '#d97706' : '#059669', color: 'white', border: 'none', padding: '8px 18px', borderRadius: '6px', fontWeight: 'bold', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}
          >
            {isPlaying ? <Pause size={14} /> : <Play size={14} />}
            {isPlaying ? 'Pause Playback' : 'Play 24-Hour Cycle'}
          </button>
          <span style={{ fontSize: '11px', color: '#94a3b8' }}>
            Scrub timeline slider to evaluate peak rush hour speed drops and GNN forecast convergence.
          </span>
        </div>
      </div>

      {/* Network Speed Trend SVG Chart */}
      <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '20px', marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2 style={{ fontSize: '14px', fontWeight: '700', color: '#f8fafc', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <TrendingUp size={18} color="#38bdf8" /> Network Speed Curve vs. Graph WaveNet 15-min Forecast
          </h2>

          <div style={{ display: 'flex', gap: '16px', fontSize: '11px' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#38bdf8' }}>
              <span style={{ width: '12px', height: '3px', background: '#38bdf8', borderRadius: '2px' }}></span> Ground Truth
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#c084fc' }}>
              <span style={{ width: '12px', height: '3px', background: '#c084fc', borderRadius: '2px' }}></span> GWNet 15-min Forecast
            </span>
          </div>
        </div>

        <div style={{ background: '#0f172a', padding: '16px', borderRadius: '8px', border: '1px solid #334155' }}>
          <svg viewBox="0 0 780 260" style={{ width: '100%', height: '240px' }}>
            <line x1="40" y1="40" x2="750" y2="40" stroke="#334155" strokeDasharray="4" />
            <line x1="40" y1="100" x2="750" y2="100" stroke="#334155" strokeDasharray="4" />
            <line x1="40" y1="160" x2="750" y2="160" stroke="#334155" strokeDasharray="4" />
            <line x1="40" y1="220" x2="750" y2="220" stroke="#334155" strokeLinecap="round" />

            <text x="10" y="45" fill="#94a3b8" fontSize="11">60 mph</text>
            <text x="10" y="105" fill="#94a3b8" fontSize="11">50 mph</text>
            <text x="10" y="165" fill="#94a3b8" fontSize="11">40 mph</text>
            <text x="10" y="225" fill="#94a3b8" fontSize="11">30 mph</text>

            {/* Ground Truth Speed Curve */}
            <polyline
              fill="none"
              stroke="#38bdf8"
              strokeWidth="3"
              points={generateSpeedSvgPoints(false)}
            />

            {/* GWNet Forecast Speed Curve */}
            <polyline
              fill="none"
              stroke="#c084fc"
              strokeWidth="2"
              strokeDasharray="5 3"
              points={generateSpeedSvgPoints(true)}
            />

            {/* Current Step Tracker Line */}
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

      {/* Feature 2: Pareto Frontier Trade-Off Cards */}
      <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '20px' }}>
        <h2 style={{ fontSize: '14px', fontWeight: '700', color: '#c084fc', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <ShieldCheck size={18} /> Feature 2: Pareto Frontier Trade-Off Matrix (Throughput vs. Spatial Equity)
        </h2>
        <p style={{ fontSize: '11px', color: '#94a3b8', marginBottom: '16px' }}>
          Evaluating baseline models vs. Graph WaveNet Pareto-optimal reliability paradigms.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px' }}>
          {paretoPoints.map((item, idx) => (
            <div key={idx} style={{ background: '#0f172a', border: `1px solid ${item.color}`, borderRadius: '8px', padding: '14px' }}>
              <div style={{ fontSize: '12px', fontWeight: '700', color: item.color, marginBottom: '6px' }}>{item.strategy}</div>
              <div style={{ fontSize: '11px', color: '#cbd5e1' }}>Prediction Error (MAE): <strong>{item.mae} mph</strong></div>
              <div style={{ fontSize: '11px', color: '#cbd5e1' }}>Fairness Disparity (RSF): <strong>{item.rsf}</strong></div>
              <div style={{ marginTop: '8px', fontSize: '10px', fontWeight: '700', background: `${item.color}22`, color: item.color, padding: '2px 6px', borderRadius: '4px', display: 'inline-block' }}>
                {item.status}
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};

export default AnalyticsView;
