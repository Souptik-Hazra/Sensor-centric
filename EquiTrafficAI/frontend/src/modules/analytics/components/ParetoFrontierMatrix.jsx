import React from 'react';
import { ShieldCheck } from 'lucide-react';

const paretoPoints = [
  { strategy: 'DCRNN Baseline', mae: '3.06', rsf: '0.644', status: 'DOMINATED', color: '#ef4444' },
  { strategy: 'FairSTG Baseline', mae: '2.69', rsf: '0.477', status: 'SUB-OPTIMAL', color: '#f59e0b' },
  { strategy: 'GWNet (Suburban Equity)', mae: '2.37', rsf: '0.238', status: 'PARETO OPTIMAL', color: '#a855f7' },
  { strategy: 'GWNet (Max Throughput)', mae: '2.01', rsf: '0.37', status: 'PARETO OPTIMAL', color: '#38bdf8' },
];

const ParetoFrontierMatrix = () => {
  return (
    <div className="ui-card-glass">
      <h2 className="ui-section-title text-purple">
        <ShieldCheck size={18} /> Feature 2: Pareto Frontier Trade-Off Matrix (Throughput vs. Spatial Equity)
      </h2>
      <p className="ui-section-desc">
        Evaluating baseline models vs. Graph WaveNet Pareto-optimal reliability paradigms.
      </p>

      <div className="ui-pareto-grid">
        {paretoPoints.map((item, idx) => (
          <div key={idx} className="ui-pareto-card" style={{ borderColor: item.color, borderWidth: '1px', borderStyle: 'solid' }}>
            <div className="ui-pareto-title" style={{ color: item.color }}>{item.strategy}</div>
            <div className="ui-pareto-stat">Prediction Error (MAE): <strong>{item.mae} mph</strong></div>
            <div className="ui-pareto-stat">Fairness Disparity (RSF): <strong>{item.rsf}</strong></div>
            <div className="ui-pareto-badge" style={{ backgroundColor: `${item.color}22`, color: item.color }}>
              {item.status}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default React.memo(ParetoFrontierMatrix);
