import React, { useState } from 'react';
import { Bot, Sparkles, RefreshCw, AlertCircle, CheckCircle2, Copy, Check } from 'lucide-react';
import { analyzeWithAgent } from '../api';

const AgentPanel = ({ resumeText, jdText, defaultGaps = [] }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isKeyMissing, setIsKeyMissing] = useState(false);
  const [agentData, setAgentData] = useState(null);
  const [copiedIndex, setCopiedIndex] = useState(null);

  const runAgent = async () => {
    if (!resumeText || !resumeText.trim()) {
      setError("Please provide resume text to run the AI Agent. If you uploaded a PDF, paste the text into 'Or Paste Resume Text'.");
      return;
    }
    if (!jdText || !jdText.trim()) {
      setError("Job Description text is required for agent analysis.");
      return;
    }

    setLoading(true);
    setError(null);
    setIsKeyMissing(false);

    try {
      const data = await analyzeWithAgent({
        resume_text: resumeText.trim(),
        jd_text: jdText.trim(),
      });
      setAgentData(data);
    } catch (err) {
      console.error("Agent error:", err);
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail || err?.message || "Agent analysis failed";

      if (status === 503 && detail.toLowerCase().includes("google_api_key")) {
        setIsKeyMissing(true);
        setError("AI Agent requires a Google AI Studio API key. Set GOOGLE_API_KEY in your backend environment (.env) to enable LangGraph suggestions.");
      } else if (status === 429) {
        setError("Google AI Studio rate limit reached. Please wait a minute and retry.");
      } else if (status === 504) {
        setError("Agent analysis timed out. Try using a more concise resume or JD excerpt.");
      } else {
        setError(detail);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = (text, index) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const suggestions = agentData?.suggestions?.suggestions || [];
  const generalTips = agentData?.suggestions?.general_tips || [];
  const agentGaps = agentData?.gaps?.missing_skills || defaultGaps;
  const nodeErrors = agentData?.node_errors;

  return (
    <div className="glass-panel" style={{ marginTop: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            background: 'var(--accent-muted)',
            color: 'var(--accent-color)',
            padding: '8px',
            borderRadius: 'var(--radius-sm)',
            display: 'flex',
            alignItems: 'center'
          }}>
            <Bot size={22} />
          </div>
          <div>
            <h3 style={{ fontSize: '1.25rem', margin: 0, fontWeight: 700 }}>AI Agent Suggestions</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', margin: 0 }}>
              Stateful LangGraph agent: parses JD &bull; extracts skills &bull; scores gaps &bull; writes bullet rewrites
            </p>
          </div>
        </div>

        <button
          type="button"
          className="btn-primary"
          onClick={runAgent}
          disabled={loading}
          style={{ width: 'auto', padding: '10px 20px', display: 'inline-flex', alignItems: 'center', gap: '8px' }}
        >
          {loading ? (
            <>
              <RefreshCw size={16} className="spin-icon" style={{ animation: 'spin 1s linear infinite' }} />
              <span>Analyzing with Agent...</span>
            </>
          ) : (
            <>
              <Sparkles size={16} />
              <span>{agentData ? 'Re-run AI Agent' : 'Generate AI Agent Suggestions'}</span>
            </>
          )}
        </button>
      </div>

      {/* Missing Key Banner */}
      {isKeyMissing && (
        <div style={{
          background: 'rgba(245, 158, 11, 0.12)',
          border: '1px solid rgba(245, 158, 11, 0.4)',
          borderRadius: 'var(--radius-md)',
          padding: '16px',
          marginBottom: '20px',
          display: 'flex',
          gap: '12px',
          alignItems: 'flex-start'
        }}>
          <AlertCircle size={20} style={{ color: 'var(--warning)', flexShrink: 0, marginTop: '2px' }} />
          <div>
            <div style={{ fontWeight: 600, color: 'var(--warning)', marginBottom: '4px' }}>
              GOOGLE_API_KEY Not Configured
            </div>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', margin: 0 }}>
              The multi-step LangGraph agent uses Google Gemini via Google AI Studio. 
              To activate it, obtain a free API key at{' '}
              <a 
                href="https://aistudio.google.com/" 
                target="_blank" 
                rel="noreferrer" 
                style={{ color: 'var(--accent-color)', textDecoration: 'underline' }}
              >
                aistudio.google.com
              </a>{' '}
              and set <code>GOOGLE_API_KEY=your_key</code> in your backend environment.
            </p>
          </div>
        </div>
      )}

      {/* General Error */}
      {error && !isKeyMissing && (
        <div style={{
          background: 'rgba(239, 68, 68, 0.12)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: 'var(--radius-md)',
          padding: '14px 16px',
          marginBottom: '20px',
          color: 'var(--danger)',
          fontSize: '0.9rem',
          display: 'flex',
          gap: '10px',
          alignItems: 'center'
        }}>
          <AlertCircle size={18} style={{ flexShrink: 0 }} />
          <span>{error}</span>
        </div>
      )}

      {/* Node error notice if partial results */}
      {nodeErrors && Object.keys(nodeErrors).length > 0 && (
        <div style={{
          background: 'rgba(245, 158, 11, 0.08)',
          border: '1px solid rgba(245, 158, 11, 0.25)',
          borderRadius: 'var(--radius-sm)',
          padding: '10px 14px',
          marginBottom: '16px',
          fontSize: '0.85rem',
          color: 'var(--text-secondary)'
        }}>
          Note: Some agent steps used graceful fallbacks ({Object.keys(nodeErrors).join(', ')}).
        </div>
      )}

      {/* Prompt to click if not yet run */}
      {!agentData && !loading && !error && (
        <div style={{
          padding: '30px 20px',
          textAlign: 'center',
          background: 'var(--field-bg)',
          borderRadius: 'var(--radius-md)',
          border: '1px dashed var(--panel-border)'
        }}>
          <Sparkles size={32} style={{ color: 'var(--accent-color)', marginBottom: '10px', opacity: 0.8 }} />
          <p style={{ fontWeight: 600, marginBottom: '6px' }}>Ready for deep agent analysis</p>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '540px', margin: '0 auto' }}>
            Click &ldquo;Generate AI Agent Suggestions&rdquo; above to run the 4-node LangGraph pipeline. 
            It extracts structured JD requirements, analyzes skill gaps using LangChain tools, and generates tailored resume rewrite bullets.
          </p>
        </div>
      )}

      {/* Agent Results Display */}
      {agentData && (
        <div>
          {/* Skill Gaps Identified */}
          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Targeted Skill Gaps ({agentGaps.length})
            </h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {agentGaps.length > 0 ? (
                agentGaps.map((skill, i) => (
                  <span key={i} className="chip danger">
                    {skill}
                  </span>
                ))
              ) : (
                <span className="chip success">No critical skill gaps detected!</span>
              )}
            </div>
          </div>

          {/* Rewrite Suggestions */}
          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              Recommended Resume Bullet Rewrites
            </h4>

            {suggestions.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
                {suggestions.map((item, idx) => (
                  <div
                    key={idx}
                    style={{
                      background: 'var(--item-bg)',
                      border: '1px solid var(--panel-border)',
                      borderLeft: '4px solid var(--accent-color)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '16px'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span className="chip warning" style={{ margin: 0 }}>
                        Gap: {item.skill_gap}
                      </span>
                      <button
                        type="button"
                        onClick={() => handleCopy(item.suggested_bullet, idx)}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: copiedIndex === idx ? 'var(--success)' : 'var(--text-secondary)',
                          cursor: 'pointer',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px',
                          fontSize: '0.8rem'
                        }}
                        title="Copy bullet"
                      >
                        {copiedIndex === idx ? <Check size={14} /> : <Copy size={14} />}
                        <span>{copiedIndex === idx ? 'Copied' : 'Copy'}</span>
                      </button>
                    </div>

                    <div style={{
                      fontFamily: 'var(--font-body)',
                      fontSize: '0.96rem',
                      color: 'var(--text-primary)',
                      lineHeight: '1.5',
                      marginBottom: item.reasoning ? '8px' : '0'
                    }}>
                      &bull; {item.suggested_bullet}
                    </div>

                    {item.reasoning && (
                      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        <strong>Why:</strong> {item.reasoning}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                No specific bullet rewrites needed. Your resume matches the key technical requirements.
              </div>
            )}
          </div>

          {/* General Tips */}
          {generalTips.length > 0 && (
            <div style={{
              background: 'var(--card-bg)',
              borderRadius: 'var(--radius-sm)',
              padding: '14px 16px',
              border: '1px solid var(--panel-border)'
            }}>
              <h4 style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Agent Strategic Advice
              </h4>
              <ul style={{ paddingLeft: '20px', color: 'var(--text-primary)', fontSize: '0.9rem' }}>
                {generalTips.map((tip, i) => (
                  <li key={i} style={{ marginBottom: '4px' }}>{tip}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Workflow Footer */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            marginTop: '16px',
            fontSize: '0.78rem',
            color: 'var(--text-secondary)'
          }}>
            <CheckCircle2 size={14} style={{ color: 'var(--success)' }} />
            <span>Workflow completed: parse_jd &rarr; extract_resume_skills &rarr; analyze_gaps (@tool) &rarr; suggest_rewrites</span>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default AgentPanel;
