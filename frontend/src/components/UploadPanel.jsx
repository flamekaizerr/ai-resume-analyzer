import React, { useState } from 'react';
import { analyzeResume } from '../api';

const UploadPanel = ({ onAnalysisComplete, onRunAgent, initialJd = '', initialResume = '' }) => {
  const [jdText, setJdText] = useState(initialJd);
  const [resumeText, setResumeText] = useState(initialResume);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!jdText) {
      setError("Job Description is required");
      return;
    }
    if (!file && !resumeText) {
      setError("Either Resume file or text is required");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('jd_text', jdText);
    if (file) {
      formData.append('resume_file', file);
    }
    if (resumeText) {
      formData.append('resume_text', resumeText);
    }

    try {
      const data = await analyzeResume(formData);
      onAnalysisComplete({
        ...data,
        rawResumeText: resumeText,
        rawJdText: jdText
      });
    } catch (err) {
      console.error(err);
      const message = err?.response?.data?.detail || err?.message || "Analysis failed. Please try again.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleAgentClick = () => {
    if (!jdText || !jdText.trim()) {
      setError("Job Description is required for AI Agent Suggestions");
      return;
    }
    if (!resumeText || !resumeText.trim()) {
      setError("Please paste resume text into 'Or Paste Resume Text' to run the AI Agent");
      return;
    }
    if (onRunAgent) {
      onRunAgent({ jdText, resumeText });
    }
  };

  return (
    <div className="glass-panel">
      <h2 style={{ marginBottom: '20px' }}>New Analysis</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Job Description</label>
          <textarea 
            placeholder="Paste the JD here..."
            value={jdText}
            onChange={(e) => setJdText(e.target.value)}
          />
        </div>
        
        <div className="form-group">
          <label>Resume (PDF)</label>
          <input 
            type="file" 
            accept=".pdf"
            onChange={(e) => setFile(e.target.files[0])}
          />
        </div>

        <div className="form-group">
          <label>Or Paste Resume Text</label>
          <textarea 
            placeholder="Paste resume text if you don't have a PDF..."
            value={resumeText}
            onChange={(e) => setResumeText(e.target.value)}
          />
        </div>

        {error && <div style={{ color: 'var(--danger)', marginBottom: '16px' }}>{error}</div>}

        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          <button type="submit" className="btn-primary" disabled={loading} style={{ flex: '1 1 200px' }}>
            {loading ? 'Analyzing...' : 'Analyze Fit'}
          </button>
          <button 
            type="button" 
            className="tab-btn" 
            onClick={handleAgentClick}
            disabled={loading}
            style={{ 
              flex: '1 1 200px', 
              borderColor: 'var(--accent-color)', 
              color: 'var(--text-primary)',
              fontWeight: 600
            }}
          >
            AI Agent Suggestions (LangGraph)
          </button>
        </div>
      </form>
    </div>
  );
};

export default UploadPanel;
