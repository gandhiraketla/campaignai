import React, { useState } from 'react';
import { Send, Loader2 } from 'lucide-react';

function QueryInput({ onSubmit, isLoading }) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSubmit(query);
    }
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-8 shadow-2xl">
      <h2 className="text-3xl font-bold text-center bg-clip-text text-transparent 
                     bg-gradient-to-r from-indigo-600 to-pink-600 mb-6">
        AI Campaign Insights
      </h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Describe your marketing query..."
          className="w-full px-4 py-3 bg-white/10 border border-white/20 
                     rounded-xl text-white placeholder-white/50 
                     focus:outline-none focus:ring-2 focus:ring-indigo-500 
                     resize-y min-h-[150px]"
          disabled={isLoading}
        />
        <button 
          type="submit" 
          disabled={isLoading || !query.trim()}
          className="w-full flex items-center justify-center 
                     bg-gradient-to-r from-indigo-600 to-pink-600 
                     text-white py-4 rounded-xl 
                     hover:from-indigo-700 hover:to-pink-700 
                     transition duration-300 
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <><Loader2 className="mr-2 animate-spin" /> Generating Insights...</>
          ) : (
            <><Send className="mr-2" /> Get Insights</>
          )}
        </button>
      </form>
    </div>
  );
}

export default QueryInput;