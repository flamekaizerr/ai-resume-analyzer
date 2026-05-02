import React from 'react';

const FeedbackPanel = ({ feedback }) => {
  return (
    <div className="glass-panel">
      <h3 style={{ marginBottom: '16px' }}>AI Improvement Suggestions</h3>
      <ul className="feedback-list">
        {feedback && feedback.length > 0 ? (
          feedback.map((item, i) => (
            <li key={i}>{item}</li>
          ))
        ) : (
          <li>No feedback generated yet.</li>
        )}
      </ul>
    </div>
  );
};

export default FeedbackPanel;
