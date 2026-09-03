import React, { useState, useEffect } from 'react';
import { BarChart2, Globe } from 'lucide-react';
import styles from './AnalyticsView.module.css';
import ExecutiveMetricsGrid from './components/ExecutiveMetricsGrid';
import ParetoFrontierMatrix from './components/ParetoFrontierMatrix';
import SpeedTrendSvgChart from './components/SpeedTrendSvgChart';
import { fetchAnalyticsMetrics, fetchModelHealth, diagnoseSensor, fetchPolicy } from '../../services/apiService';

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
  const [modelHealth, setModelHealth] = useState(null);
  const [causalData, setCausalData] = useState(null);
  const [policyData, setPolicyData] = useState(null);

  useEffect(() => {
    const fetchAnalyticsMetrics = async () => {
      try {
        const [metricsResult, healthResult, causalResult, policyResult] = await Promise.allSettled([
          fetchAnalyticsMetrics(selectedCity),
          fetchModelHealth(),
          diagnoseSensor(0, selectedCity),
          fetchPolicy('equity')
        ]);
        if (metricsResult.status === 'fulfilled') setAnalyticsData(metricsResult.value);
        if (healthResult.status === 'fulfilled') setModelHealth(healthResult.value);
        if (causalResult.status === 'fulfilled') setCausalData(causalResult.value);
        if (policyResult.status === 'fulfilled') setPolicyData(policyResult.value);
      } catch (err) {
        console.error("Failed to fetch analytics metrics:", err);
      }
    };
    fetchAnalyticsMetrics();
  }, [selectedCity]);

  return (
    <div className={styles.pageContainer}>
      
      {/* Header Bar with City Switcher */}
      <div className={styles.headerBar}>
        <div>
          <h1 className={styles.pageTitle}>
            <BarChart2 size={24} color="#38bdf8" /> Data Analytics & Multi-Horizon GNN Benchmark
          </h1>
          <p className={styles.pageSubtitle}>
            Evaluating Graph WaveNet (GWNet) 15-min forecasts, SCM Causal Attributions, and Pareto Equity Trade-offs.
          </p>
        </div>

        <div className={styles.switcherBox}>
          <Globe size={16} color="#38bdf8" />
          <label htmlFor="analytics-city-select" className={styles.switcherLabel}>Target Corridor:</label>
          <select 
            id="analytics-city-select"
            value={selectedCity} 
            onChange={(e) => setSelectedCity(e.target.value)}
            className="ui-select-dark"
            aria-label="Select Target Freeway Corridor Dataset"
          >
            <option value="la">Los Angeles METR-LA (207 Sensors)</option>
            <option value="sd">San Diego SD400 (716 Sensors)</option>
          </select>
        </div>
      </div>

      {/* Top Executive Metric Cards Grid Sub-Component */}
      <ExecutiveMetricsGrid selectedCity={selectedCity} />

      {/* 24-Hour Telemetry Scrubber & Speed Curve SVG Chart Sub-Component */}
      <SpeedTrendSvgChart 
        step={step}
        setStep={setStep}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
        formatTime={formatTime}
        selectedCity={selectedCity}
        generateSpeedSvgPoints={generateSpeedSvgPoints}
        MAX_STEPS={MAX_STEPS}
      />

      {/* Feature 2: Pareto Frontier Trade-Off Cards Sub-Component */}
      <ParetoFrontierMatrix points={analyticsData?.pareto_matrix} />

      <div className="ui-pareto-grid mt-20">
        <div className="ui-card-glass">
          <h3 className="ui-section-title text-cyan">Model Readiness</h3>
          {modelHealth ? Object.entries(modelHealth).map(([name, model]) => (
            <div key={name} className="ui-pareto-stat">
              <strong>{name.toUpperCase()}</strong>: {model.checkpoint_loaded ? 'Ready' : 'Unavailable'}
              {model.error ? <div className="ui-section-desc">{model.error}</div> : null}
            </div>
          )) : <div className="ui-section-desc">Loading model status...</div>}
        </div>
        <div className="ui-card-glass">
          <h3 className="ui-section-title text-purple">Causal Diagnosis</h3>
          {causalData ? (
            <>
              <div className="ui-pareto-stat">Sensor: <strong>{causalData.sensor_id}</strong></div>
              <div className="ui-pareto-stat">Current speed: <strong>{causalData.current_speed_mph} mph</strong></div>
              <div className="ui-pareto-stat">Indirect reliability effect: <strong>{causalData.causal_scm_breakdown?.indirect_reliability_contribution_pct ?? causalData.causal_scm_breakdown?.ctf_indirect_effect_reliability_ctf_ie_r}</strong></div>
            </>
          ) : <div className="ui-section-desc">Loading causal diagnosis...</div>}
        </div>
        <div className="ui-card-glass">
          <h3 className="ui-section-title text-emerald">Active Policy</h3>
          {policyData ? (
            <>
              <div className="ui-pareto-stat">Goal: <strong>{policyData.user_goal}</strong></div>
              <div className="ui-pareto-stat">Paradigm: <strong>{policyData.reliability_paradigm}</strong></div>
              <div className="ui-section-desc">{policyData.policy_explanation}</div>
            </>
          ) : <div className="ui-section-desc">Loading policy...</div>}
        </div>
      </div>

    </div>
  );
};

export default AnalyticsView;
