import React from 'react';

const formatDate = (value) => {
  if (!value) return 'N/A';
  const date = new Date(value);
  return new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short'
  }).format(date);
};

const formatSkills = (skills) => {
  if (!skills || skills.length === 0) return 'N/A';
  return skills.slice(0, 3).join(', ');
};

const HistoryTable = ({ items, loading, error }) => {
  return (
    <div className="glass-panel history-panel">
      <div className="panel-header">
        <div>
          <h3>Analysis History</h3>
          <p>Review your recent resume-to-job fits.</p>
        </div>
      </div>

      {loading && <div className="panel-state">Loading history...</div>}
      {!loading && error && <div className="panel-state error">{error}</div>}
      {!loading && !error && (!items || items.length === 0) && (
        <div className="panel-state">No analyses yet. Run your first analysis to populate this list.</div>
      )}

      {!loading && !error && items && items.length > 0 && (
        <div className="table-wrapper">
          <table className="history-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Similarity</th>
                <th>Matched Skills</th>
                <th>Missing Skills</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.id}>
                  <td data-label="Date">{formatDate(item.timestamp)}</td>
                  <td data-label="Similarity">
                    {Math.round((item.similarity_score || 0) * 100)}%
                  </td>
                  <td data-label="Matched">{formatSkills(item.matched_skills)}</td>
                  <td data-label="Missing">{formatSkills(item.missing_skills)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default HistoryTable;
