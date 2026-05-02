import React, { lazy, Suspense, useCallback, useEffect, useState } from 'react';
import { Moon, Sun } from 'lucide-react';
import { getHistory, getStats } from './api';

const UploadPanel = lazy(() => import('./components/UploadPanel'));
const ScoreGauge = lazy(() => import('./components/ScoreGauge'));
const SkillBadges = lazy(() => import('./components/SkillBadges'));
const RadarChart = lazy(() => import('./components/RadarChart'));
const FeedbackPanel = lazy(() => import('./components/FeedbackPanel'));
const HistoryTable = lazy(() => import('./components/HistoryTable'));
const StatsPanel = lazy(() => import('./components/StatsPanel'));

const TabFallback = ({ label }) => (
  <div className="glass-panel panel-state">{label}</div>
);

function App() {
  const [activeTab, setActiveTab] = useState('analyze');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [insightsError, setInsightsError] = useState(null);
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('theme');
    if (saved) return saved;
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  });

  useEffect(() => {
    document.body.classList.remove('theme-dark', 'theme-light');
    document.body.classList.add(`theme-${theme}`);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const refreshInsights = useCallback(async () => {
    setInsightsLoading(true);
    setInsightsError(null);
    try {
      const [historyData, statsData] = await Promise.all([getHistory(), getStats()]);
      setHistory(historyData);
      setStats(statsData);
    } catch (err) {
      console.error(err);
      setInsightsError('Failed to load history and stats.');
    } finally {
      setInsightsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (activeTab === 'tracker') {
      refreshInsights();
    }
  }, [activeTab, refreshInsights]);

  const handleAnalysisComplete = (data) => {
    setResult(data);
    setActiveTab('result');
    refreshInsights();
  };

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-top">
          <div>
            <h1>AI Resume Screener</h1>
            <p>Job-Fit Analyzer & Skill Gap Detection</p>
          </div>
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            aria-label="Toggle dark mode"
          >
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            <span>{theme === 'dark' ? 'Light' : 'Dark'}</span>
          </button>
        </div>
      </header>

      <div className="tabs">
        <button
          className={`tab-btn ${activeTab === 'analyze' ? 'active' : ''}`}
          onClick={() => setActiveTab('analyze')}
        >
          New Analysis
        </button>
        <button
          className={`tab-btn ${activeTab === 'result' ? 'active' : ''}`}
          onClick={() => setActiveTab('result')}
          disabled={!result}
        >
          Results
        </button>
        <button
          className={`tab-btn ${activeTab === 'tracker' ? 'active' : ''}`}
          onClick={() => setActiveTab('tracker')}
        >
          History & Stats
        </button>
      </div>

      <main>
        {activeTab === 'analyze' && (
          <Suspense fallback={<TabFallback label="Loading analysis form..." />}>
            <UploadPanel onAnalysisComplete={handleAnalysisComplete} />
          </Suspense>
        )}

        {activeTab === 'result' && result && (
          <Suspense fallback={<TabFallback label="Loading results..." />}>
            <div>
              <div className="grid-layout" style={{ marginBottom: '24px' }}>
                <ScoreGauge score={result.similarity_score} />
                <RadarChart scores={result.section_scores} />
              </div>
              <div className="grid-layout">
                <SkillBadges 
                  matched={result.matched_skills}
                  missing={result.missing_skills}
                  transferable={result.transferable_skills}
                />
                <FeedbackPanel feedback={result.feedback} />
              </div>
            </div>
          </Suspense>
        )}

        {activeTab === 'tracker' && (
          <Suspense fallback={<TabFallback label="Loading history and stats..." />}>
            <div className="tracker-layout">
              <StatsPanel stats={stats} loading={insightsLoading} error={insightsError} />
              <HistoryTable items={history} loading={insightsLoading} error={insightsError} />
            </div>
          </Suspense>
        )}
      </main>
    </div>
  );
}

export default App;
