import React from 'react';
import { Cpu, ShieldCheck, Zap, Award } from 'lucide-react';

const ExecutiveMetricsGrid = ({ selectedCity }) => {
  return (
    <div className="ui-grid-cards">
      
      <div className="ui-metric-card">
        <div className="ui-label-sm ui-metric-header">
          <Cpu size={14} color="#38bdf8" /> Network MAE Accuracy
        </div>
        <div className="ui-metric-value text-cyan">
          1.82 <span className="ui-metric-unit">mph</span>
        </div>
        <div className="ui-metric-subtext text-emerald">+34.2% vs DCRNN Baseline (2.77 mph)</div>
      </div>

      <div className="ui-metric-card">
        <div className="ui-label-sm ui-metric-header">
          <ShieldCheck size={14} color="#c084fc" /> Regional Equity (RSF)
        </div>
        <div className="ui-metric-value text-purple">0.0705</div>
        <div className="ui-metric-subtext text-purple">-23.4% Disparity for Suburban Commuters</div>
      </div>

      <div className="ui-metric-card">
        <div className="ui-label-sm ui-metric-header">
          <Zap size={14} color="#f59e0b" /> CAP-D Causal Cascade
        </div>
        <div className="ui-metric-value text-amber">61.3%</div>
        <div className="ui-metric-subtext">Indirect Cascades (21.4% Direct Local)</div>
      </div>

      <div className="ui-metric-card">
        <div className="ui-label-sm ui-metric-header">
          <Award size={14} color="#34d399" /> Zero-Dropout Quality
        </div>
        <div className="ui-metric-value text-emerald">{selectedCity === 'sd' ? '2.75%' : '8.45%'}</div>
        <div className="ui-metric-subtext">Hardware Failure Filtering Rate</div>
      </div>

    </div>
  );
};

export default React.memo(ExecutiveMetricsGrid);
