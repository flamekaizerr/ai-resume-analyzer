import React from 'react';

const StatCard = ({ label, value }) => {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
};

const SkillPills = ({ title, skills, tone }) => {
  return (
    <div className="skill-list">
      <h4>{title}</h4>
      <div className="skill-pills">
        {skills.length === 0 && <span className="pill-empty">N/A</span>}
        {skills.map((item) => (
          <span key={item.skill} className={`chip ${tone}`}>
            {item.skill} <strong>({item.count})</strong>
          </span>
        ))}
      </div>
    </div>
  );
};

const StatsPanel = ({ stats, loading, error }) => {
  return (
    <div className="glass-panel stats-panel">
      <div className="panel-header">
        <div>
          <h3>Aggregate Insights</h3>
          <p>Track overall fit and skill trends.</p>
        </div>
      </div>

      {loading && <div className="panel-state">Loading stats...</div>}
      {!loading && error && <div className="panel-state error">{error}</div>}

      {!loading && !error && stats && (
        <>
          <div className="stats-grid">
            <StatCard label="Total Analyses" value={stats.total_analyses} />
            <StatCard label="Average Fit" value={`${Math.round(stats.avg_score * 100)}%`} />
            <StatCard label="Best Fit" value={`${Math.round(stats.best_score * 100)}%`} />
            <StatCard label="Recent Avg" value={`${Math.round(stats.recent_avg * 100)}%`} />
          </div>

          <div className="stats-lists">
            <SkillPills title="Top Missing Skills" skills={stats.top_missing_skills || []} tone="danger" />
            <SkillPills title="Top Matched Skills" skills={stats.top_matched_skills || []} tone="success" />
          </div>
        </>
      )}
    </div>
  );
};

export default StatsPanel;
