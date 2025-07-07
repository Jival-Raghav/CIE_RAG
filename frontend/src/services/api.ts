import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Your FastAPI backend URL

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout for AI responses
});

export interface ChatRequest {
  query: string;
  username: string;
}

export interface ChatResponse {
  response: string;
}

export const chatAPI = {
  sendMessage: async (query: string, username: string): Promise<ChatResponse> => {
    try {
      const response = await api.post<ChatResponse>('/chat', {
        query,
        username
      });
      return response.data;
    } catch (error) {
      console.error('API Error:', error);
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.detail || 'Failed to send message');
      }
      throw error;
    }
  }
};

export default api;
