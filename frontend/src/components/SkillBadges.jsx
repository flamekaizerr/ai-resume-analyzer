import React from 'react';

const SkillBadges = ({ matched, missing, transferable }) => {
  return (
    <div className="glass-panel">
      <h3 style={{ marginBottom: '16px' }}>Skills Analysis</h3>
      
      <div style={{ marginBottom: '16px' }}>
        <h4 style={{ color: 'var(--text-secondary)', marginBottom: '8px', fontSize: '0.9rem' }}>Matched Skills</h4>
        <div>
          {matched?.length > 0 ? (
            matched.map((skill, i) => (
              <span key={i} className="chip success">{skill}</span>
            ))
          ) : <span style={{ color: 'var(--text-secondary)' }}>None identified</span>}
        </div>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <h4 style={{ color: 'var(--text-secondary)', marginBottom: '8px', fontSize: '0.9rem' }}>Missing Skills (Gap)</h4>
        <div>
          {missing?.length > 0 ? (
            missing.map((skill, i) => (
              <span key={i} className="chip danger">{skill}</span>
            ))
          ) : <span style={{ color: 'var(--text-secondary)' }}>No gaps identified!</span>}
        </div>
      </div>

      {transferable?.length > 0 && (
        <div>
          <h4 style={{ color: 'var(--text-secondary)', marginBottom: '8px', fontSize: '0.9rem' }}>Transferable Skills</h4>
          <div>
            {transferable.map((skill, i) => (
              <span key={i} className="chip warning">{skill}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SkillBadges;
