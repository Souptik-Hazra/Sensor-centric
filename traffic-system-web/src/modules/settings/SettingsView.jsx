import React, { useState } from 'react';
import { Sliders, Server, ShieldCheck, Database, Save, CheckCircle2 } from 'lucide-react';

const SettingsView = () => {
  const [defaultCity, setDefaultCity] = useState('la');
  const [defaultHorizon, setDefaultHorizon] = useState(30);
  const [reliabilityVariant, setReliabilityVariant] = useState('reliability_equal');
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');
  const [cctvEndpoint, setCctvEndpoint] = useState('https://cctv.dot.ca.gov');
  const [savedSuccess, setSavedSuccess] = useState(false);

  const handleSave = (e) => {
    e.preventDefault();
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 3000);
  };

  return (
    <div style={{ padding: '24px', background: '#0f172a', minHeight: '100vh', color: '#f8fafc' }}>
      
      {/* Header */}
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: '700', color: '#38bdf8', marginBottom: '4px' }}>
          System Settings & AI Configuration
        </h2>
        <p style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
          Configure default datasets, GWNet forecasting horizons, Pareto policy rules, and API endpoints.
        </p>
      </div>

      {savedSuccess && (
        <div style={{ background: '#05966922', border: '1px solid #10b981', color: '#34d399', padding: '12px 16px', borderRadius: '8px', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', fontWeight: 'bold' }}>
          <CheckCircle2 size={18} /> System configurations saved successfully!
        </div>
      )}

      <form onSubmit={handleSave} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '20px' }}>
        
        {/* Card 1: Dataset & Forecasting Defaults */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '20px' }}>
          <h3 style={{ fontSize: '14px', fontWeight: '700', color: '#38bdf8', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            <Database size={16} /> Dataset & GWNet Forecast Defaults
          </h3>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px', fontWeight: 'bold' }}>Default City Dataset:</label>
            <select 
              value={defaultCity} 
              onChange={(e) => setDefaultCity(e.target.value)}
              style={{ width: '100%', background: '#0f172a', color: '#f8fafc', border: '1px solid #334155', padding: '8px 12px', borderRadius: '6px', fontSize: '13px' }}
            >
              <option value="la">Los Angeles METR-LA (207 Sensors)</option>
              <option value="sd">San Diego SD400 (716 Sensors)</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px', fontWeight: 'bold' }}>Default GWNet Forecast Horizon:</label>
            <select 
              value={defaultHorizon} 
              onChange={(e) => setDefaultHorizon(parseInt(e.target.value))}
              style={{ width: '100%', background: '#0f172a', color: '#f8fafc', border: '1px solid #334155', padding: '8px 12px', borderRadius: '6px', fontSize: '13px' }}
            >
              <option value={15}>15-minute Short Horizon</option>
              <option value={30}>30-minute Medium Horizon (Recommended)</option>
              <option value={60}>60-minute Long Horizon</option>
            </select>
          </div>
        </div>

        {/* Card 2: Feature 2 - Pareto Policy Advisor Defaults */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '20px' }}>
          <h3 style={{ fontSize: '14px', fontWeight: '700', color: '#c084fc', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            <ShieldCheck size={16} /> Feature 2: Civic Policy Defaults
          </h3>

          <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px', fontWeight: 'bold' }}>Active Reliability Variant:</label>
            <select 
              value={reliabilityVariant} 
              onChange={(e) => setReliabilityVariant(e.target.value)}
              style={{ width: '100%', background: '#0f172a', color: '#f8fafc', border: '1px solid #334155', padding: '8px 12px', borderRadius: '6px', fontSize: '13px' }}
            >
              <option value="reliability_equal">reliability_equal (Suburban Equity Priority)</option>
              <option value="reliability_pca">reliability_pca (Maximum Throughput Priority)</option>
              <option value="reliability_original">reliability_original (Original Baseline)</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px', fontWeight: 'bold' }}>Baseline Regional Fairness (RSF) Target:</label>
            <input 
              type="text" 
              value="0.0920" 
              disabled 
              style={{ width: '100%', background: '#0f172a', color: '#94a3b8', border: '1px solid #334155', padding: '8px 12px', borderRadius: '6px', fontSize: '13px' }}
            />
          </div>
        </div>

        {/* Card 3: Backend API & Camera Endpoint Settings */}
        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '10px', padding: '20px', gridColumn: '1 / -1' }}>
          <h3 style={{ fontSize: '14px', fontWeight: '700', color: '#34d399', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            <Server size={16} /> FastAPI Backend & Camera Endpoints
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px', marginBottom: '20px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px', fontWeight: 'bold' }}>FastAPI Backend Base URL:</label>
              <input 
                type="text" 
                value={apiUrl} 
                onChange={(e) => setApiUrl(e.target.value)}
                style={{ width: '100%', background: '#0f172a', color: '#f8fafc', border: '1px solid #334155', padding: '8px 12px', borderRadius: '6px', fontSize: '13px' }}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '12px', color: '#cbd5e1', marginBottom: '6px', fontWeight: 'bold' }}>Caltrans Public CCTV Endpoint:</label>
              <input 
                type="text" 
                value={cctvEndpoint} 
                onChange={(e) => setCctvEndpoint(e.target.value)}
                style={{ width: '100%', background: '#0f172a', color: '#f8fafc', border: '1px solid #334155', padding: '8px 12px', borderRadius: '6px', fontSize: '13px' }}
              />
            </div>
          </div>

          <button 
            type="submit"
            style={{ background: '#0284c7', color: 'white', border: 'none', padding: '10px 24px', borderRadius: '6px', fontWeight: '700', fontSize: '13px', display: 'inline-flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}
          >
            <Save size={16} /> Save System Settings
          </button>
        </div>

      </form>
    </div>
  );
};

export default SettingsView;
