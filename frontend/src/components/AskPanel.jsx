import React, { useState } from 'react';
import { MessageSquare, FileText, AlertCircle } from 'lucide-react';
import { askResume } from '../api';

const ACCEPTED = '.txt,.md,.pdf,.docx';

const AskPanel = ({ initialResume = '', initialJd = '' }) => {
  const [resumeText, setResumeText] = useState(initialResume);
  const [jdText, setJdText] = useState(initialJd);
  const [files, setFiles] = useState([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) {
      setError('Type a question first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('question', question.trim());
    if (resumeText.trim()) formData.append('resume_text', resumeText);
    if (jdText.trim()) formData.append('jd_text', jdText);
    files.forEach((f) => formData.append('files', f));

    try {
      const data = await askResume(formData);
      setResult(data);
    } catch (err) {
      console.error(err);
      if (err?.response?.status === 404) {
        setError('The backend does not have the /rag/ask route yet. Make sure it is running the latest backend code.');
        return;
      }
      const message = err?.response?.data?.detail || err?.message || 'Request failed. Please try again.';
      setError(typeof message === 'string' ? message : 'Request failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const isExtractive = result?.mode === 'extractive';

  return (
    <div className="glass-panel">
      <div className="panel-header" style={{ marginBottom: '20px' }}>
        <h3>Ask Your Resume</h3>
        <p>Paste or upload your resume and job descriptions, then ask a question about them. Answers point to the file and chunk they came from.</p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="grid-layout" style={{ gap: '16px' }}>
          <div className="form-group">
            <label>Resume Text</label>
            <textarea
              placeholder="Paste your resume here..."
              value={resumeText}
              onChange={(e) => setResumeText(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label>Job Description Text</label>
            <textarea
              placeholder="Paste one or more JDs here..."
              value={jdText}
              onChange={(e) => setJdText(e.target.value)}
            />
          </div>
        </div>

        <div className="form-group">
          <label>Or Upload Files (.txt, .md, .pdf, .docx, up to 5)</label>
          <input
            type="file"
            accept={ACCEPTED}
            multiple
            onChange={(e) => setFiles(Array.from(e.target.files || []))}
          />
        </div>

        <div className="form-group">
          <label>Question</label>
          <input
            type="text"
            placeholder="e.g. Which of my projects match this JD's requirements?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
        </div>

        {error && <div style={{ color: 'var(--danger)', marginBottom: '16px' }}>{error}</div>}

        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? 'Searching...' : 'Ask'}
        </button>
      </form>

      {result && (
        <div style={{ marginTop: '24px' }}>
          {result.index === 'sample' && (
            <p className="ask-note">No documents were given, so this searched the built-in sample resume and JDs.</p>
          )}

          {isExtractive ? (
            <div className="ask-banner">
              <AlertCircle size={18} style={{ color: 'var(--warning)', flexShrink: 0, marginTop: '2px' }} />
              <span>
                The server has no OPENROUTER_API_KEY set, so there is no written answer. These are the closest passages from your documents.
              </span>
            </div>
          ) : (
            <div className="ask-answer">
              <h4 className="ask-heading"><MessageSquare size={16} /> Answer</h4>
              <p style={{ whiteSpace: 'pre-wrap' }}>{result.answer}</p>
            </div>
          )}

          <h4 className="ask-heading" style={{ marginTop: '20px' }}>
            <FileText size={16} /> {isExtractive ? 'Top Passages' : 'Sources'} ({result.sources.length})
          </h4>
          {result.sources.length === 0 && <p className="ask-note">Nothing relevant found.</p>}
          <ul className="feedback-list">
            {result.sources.map((s, i) => (
              <li key={`${s.source}-${s.chunk_id}-${i}`}>
                <div className="ask-source-meta">
                  <span className="chip success" style={{ margin: 0 }}>{s.source} #{s.chunk_id}</span>
                  <span>score {s.score.toFixed(3)}</span>
                </div>
                <div style={{ fontSize: '0.92rem' }}>{s.text}</div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default AskPanel;
