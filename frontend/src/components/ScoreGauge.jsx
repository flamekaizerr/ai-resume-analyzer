import React, { useEffect, useState } from 'react';

const ScoreGauge = ({ score }) => {
  const normalized = Math.max(0, Math.min(1, score || 0));
  const target = Math.round(normalized * 100);
  const [displayed, setDisplayed] = useState(0);

  useEffect(() => {
    let animationFrame = null;
    const duration = 900;
    const start = performance.now();

    const animate = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      setDisplayed(Math.round(target * progress));
      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      }
    };

    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, [target]);

  const percentage = displayed;

  // Simple circular gauge using SVG
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  let color = 'var(--danger)';
  if (percentage >= 75) color = 'var(--success)';
  else if (percentage >= 50) color = 'var(--warning)';

  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <h3 style={{ marginBottom: '20px' }}>Overall Fit Score</h3>
      <div style={{ position: 'relative', width: '150px', height: '150px' }}>
        <svg width="150" height="150" viewBox="0 0 150 150">
          <circle 
            cx="75" cy="75" r={radius} 
            fill="none" 
            stroke="var(--panel-border)" 
            strokeWidth="12" 
          />
          <circle 
            cx="75" cy="75" r={radius} 
            fill="none" 
            stroke={color} 
            strokeWidth="12" 
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            transform="rotate(-90 75 75)"
            style={{ transition: 'stroke-dashoffset 1s ease-in-out' }}
          />
        </svg>
        <div style={{ 
          position: 'absolute', 
          top: '50%', 
          left: '50%', 
          transform: 'translate(-50%, -50%)',
          fontSize: '2rem',
          fontWeight: 'bold'
        }}>
          {percentage}%
        </div>
      </div>
    </div>
  );
};

export default ScoreGauge;
