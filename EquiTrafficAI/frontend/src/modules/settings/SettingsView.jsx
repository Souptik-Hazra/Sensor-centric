import React, { useState } from 'react';
import { Server, ShieldCheck, Database, Save, CheckCircle2 } from 'lucide-react';

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
    <div className="ui-page-wrapper ui-card-panel mb-0">
      
      {/* Header */}
      <div className="mb-24">
        <h2 className="ui-section-title text-cyan">
          System Settings & AI Configuration
        </h2>
        <p className="ui-section-desc">
          Configure default datasets, GWNet forecasting horizons, Pareto policy rules, and API endpoints.
        </p>
      </div>

      {savedSuccess && (
        <div className="ui-alert-success">
          <CheckCircle2 size={18} /> System configurations saved successfully!
        </div>
      )}

      <form onSubmit={handleSave} className="ui-form-grid">
        
        {/* Card 1: Dataset & Forecasting Defaults */}
        <div className="ui-card-panel mb-0">
          <h3 className="ui-section-title text-cyan mb-16">
            <Database size={16} /> Dataset & GWNet Forecast Defaults
          </h3>

          <div className="ui-form-group">
            <label htmlFor="select-default-city" className="ui-label-sm ui-label-block">Default City Dataset:</label>
            <select 
              id="select-default-city"
              value={defaultCity} 
              onChange={(e) => setDefaultCity(e.target.value)}
              className="ui-select-dark"
            >
              <option value="la">Los Angeles METR-LA (207 Sensors)</option>
              <option value="sd">San Diego SD400 (716 Sensors)</option>
            </select>
          </div>

          <div>
            <label htmlFor="select-default-horizon" className="ui-label-sm ui-label-block">Default GWNet Forecast Horizon:</label>
            <select 
              id="select-default-horizon"
              value={defaultHorizon} 
              onChange={(e) => setDefaultHorizon(parseInt(e.target.value))}
              className="ui-select-dark"
            >
              <option value={15}>15-minute Short Horizon</option>
              <option value={30}>30-minute Medium Horizon (Recommended)</option>
              <option value={60}>60-minute Long Horizon</option>
            </select>
          </div>
        </div>

        {/* Card 2: Feature 2 - Pareto Policy Advisor Defaults */}
        <div className="ui-card-panel mb-0">
          <h3 className="ui-section-title text-purple mb-16">
            <ShieldCheck size={16} /> Feature 2: Civic Policy Defaults
          </h3>

          <div className="ui-form-group">
            <label htmlFor="select-reliability-variant" className="ui-label-sm ui-label-block">Active Reliability Variant:</label>
            <select 
              id="select-reliability-variant"
              value={reliabilityVariant} 
              onChange={(e) => setReliabilityVariant(e.target.value)}
              className="ui-select-dark"
            >
              <option value="reliability_equal">reliability_equal (Suburban Equity Priority)</option>
              <option value="reliability_pca">reliability_pca (Maximum Throughput Priority)</option>
              <option value="reliability_original">reliability_original (Original Baseline)</option>
            </select>
          </div>

          <div>
            <label htmlFor="input-fairness-target" className="ui-label-sm ui-label-block">Baseline Regional Fairness (RSF) Target:</label>
            <input 
              id="input-fairness-target"
              type="text" 
              value="0.0920" 
              disabled 
              className="ui-select-dark text-secondary"
              aria-disabled="true"
            />
          </div>
        </div>

        {/* Card 3: Backend API & Camera Endpoint Settings */}
        <div className="ui-card-panel ui-col-full mb-0">
          <h3 className="ui-section-title text-emerald mb-16">
            <Server size={16} /> FastAPI Backend & Camera Endpoints
          </h3>

          <div className="ui-pareto-grid mb-20">
            <div>
              <label htmlFor="input-api-url" className="ui-label-sm ui-label-block">FastAPI Backend Base URL:</label>
              <input 
                id="input-api-url"
                type="text" 
                value={apiUrl} 
                onChange={(e) => setApiUrl(e.target.value)}
                className="ui-select-dark"
              />
            </div>
            <div>
              <label htmlFor="input-cctv-endpoint" className="ui-label-sm ui-label-block">Caltrans Public CCTV Endpoint:</label>
              <input 
                id="input-cctv-endpoint"
                type="text" 
                value={cctvEndpoint} 
                onChange={(e) => setCctvEndpoint(e.target.value)}
                className="ui-select-dark"
              />
            </div>
          </div>

          <button 
            type="submit"
            className="ui-btn-blue"
          >
            <Save size={16} /> Save System Settings
          </button>
        </div>

      </form>
    </div>
  );
};

export default SettingsView;
