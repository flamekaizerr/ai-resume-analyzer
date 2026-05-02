import React from 'react';
import { Radar, RadarChart as RechartsRadar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

const RadarChart = ({ scores }) => {
  if (!scores) return null;

  const data = [
    { subject: 'Technical', A: Math.round(scores.technical_skills * 100), fullMark: 100 },
    { subject: 'Soft Skills', A: Math.round(scores.soft_skills * 100), fullMark: 100 },
    { subject: 'Experience', A: Math.round(scores.experience_match * 100), fullMark: 100 },
    { subject: 'Education', A: Math.round(scores.education * 100), fullMark: 100 },
    { subject: 'Keywords', A: Math.round(scores.keywords * 100), fullMark: 100 },
    { subject: 'Overall', A: Math.round(scores.overall * 100), fullMark: 100 },
  ];

  return (
    <div className="glass-panel" style={{ height: '360px' }}>
      <h3 style={{ marginBottom: '16px' }}>Fit Dimensions</h3>
      <ResponsiveContainer width="100%" height="100%">
        <RechartsRadar cx="50%" cy="50%" outerRadius="80%" data={data}>
          <PolarGrid stroke="var(--panel-border)" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
          <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
          <Radar name="Score" dataKey="A" stroke="var(--accent-color)" fill="var(--accent-color)" fillOpacity={0.55} />
        </RechartsRadar>
      </ResponsiveContainer>
    </div>
  );
};

export default RadarChart;
