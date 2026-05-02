import axios from 'axios';

const api = axios.create({
  baseURL: '/api' // Uses vite proxy
});

export const analyzeResume = async (formData) => {
  const response = await api.post('/analyze', formData);
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
