import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || (
    import.meta.env.PROD
      ? 'https://flamekaizerr-ai-resume-analyzer-api.hf.space'
      : '/api'
  )
});

export const analyzeResume = async (formData) => {
  const response = await api.post('/analyze', formData);
  return response.data;
};

export const analyzeWithAgent = async (payload) => {
  const response = await api.post('/agent/analyze', payload);
  return response.data;
};

export const askResume = async (formData) => {
  const response = await api.post('/rag/ask', formData);
  return response.data;
};

export const getHistory = async () => {
  const response = await api.get('/history');
  return response.data;
};

export const getStats = async () => {
  const response = await api.get('/stats');
  return response.data;
};

export const getAnalysis = async (id) => {
  const response = await api.get(`/analysis/${id}`);
  return response.data;
};

export default api;
