import React, { useState } from 'react';
import { analyzeResume } from '../api';

const UploadPanel = ({ onAnalysisComplete }) => {
  const [jdText, setJdText] = useState('');
  const [resumeText, setResumeText] = useState('');
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
      onAnalysisComplete(data);
    } catch (err) {
      console.error(err);
      const message = err?.response?.data?.detail || err?.message || "Analysis failed. Please try again.";
      setError(message);
    } finally {
      setLoading(false);
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

        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze Fit'}
        </button>
      </form>
    </div>
  );
};

export default UploadPanel;
