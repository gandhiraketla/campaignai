import React, { useState } from 'react';
import Header from './components/Header';
import QueryInput from './components/QueryInput';
import ResponseDisplay from './components/ResponseDisplay';

function App() {
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (query) => {
    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const apiResponse = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query: query
        }),
      });

      if (!apiResponse.ok) {
        // Try to parse error response
        const errorData = await apiResponse.json().catch(() => null);
        throw new Error(
          errorData?.detail || 
          `HTTP error! status: ${apiResponse.status}`
        );
      }

      const data = await apiResponse.json();
      setResponse(data.response);
    } catch (err) {
      console.error('API Call Error:', err);
      setError(err.message || 'An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
      <Header />
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-2xl mx-auto">
          <QueryInput 
            onSubmit={handleSubmit} 
            isLoading={isLoading}
          />
          <ResponseDisplay 
            response={response} 
            error={error} 
            isLoading={isLoading} 
          />
        </div>
      </div>
      <footer className="text-center text-white/50 py-6">
        © 2025 CampaignAI. Powering Intelligent Marketing Insights.
      </footer>
    </div>
  );
}

export default App;