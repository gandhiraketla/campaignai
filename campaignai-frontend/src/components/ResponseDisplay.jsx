import React from 'react';
import { Loader2, AlertTriangle } from 'lucide-react';

function formatResponseToBulletPoints(response) {
  // Split the response into paragraphs
  const paragraphs = response.split(/\n\n/);
  
  return (
    <div className="space-y-4">
      {paragraphs.map((paragraph, index) => {
        // Check if the paragraph starts with a numbered list
        const numberedListMatch = paragraph.match(/^(\d+\.\s*.+)/gm);
        
        if (numberedListMatch) {
          // Convert numbered list to bullet points
          return (
            <ul key={index} className="list-disc list-inside space-y-2">
              {paragraph.split('\n').map((item, itemIndex) => (
                <li key={itemIndex} className="text-white/90">
                  {item.replace(/^\d+\.\s*/, '')}
                </li>
              ))}
            </ul>
          );
        }
        
        // For regular paragraphs
        return (
          <p key={index} className="text-white/90 leading-relaxed">
            {paragraph}
          </p>
        );
      })}
    </div>
  );
}

function ResponseDisplay({ response, error, isLoading }) {
  if (isLoading) {
    return (
      <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-8 mt-6 text-center">
        <div className="flex justify-center items-center text-white">
          <Loader2 className="mr-2 animate-spin" size={24} />
          <p>Analyzing your query...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 backdrop-blur-lg border border-red-500/20 rounded-2xl p-8 mt-6 text-center">
        <div className="flex justify-center items-center text-red-500">
          <AlertTriangle className="mr-2" size={24} />
          <p className="text-red-400">{error}</p>
        </div>
      </div>
    );
  }

  if (response) {
    return (
      <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-8 mt-6">
        <h3 className="text-3xl font-extrabold mb-4 
                       text-transparent bg-clip-text 
                       bg-gradient-to-r from-white via-white to-white 
                       drop-shadow-[0_2px_2px_rgba(0,0,0,0.8)]
                       tracking-wide">
          Campaign Insights
        </h3>
        <div className="leading-relaxed text-lg font-medium">
          {formatResponseToBulletPoints(response)}
        </div>
      </div>
    );
  }

  return null;
}

export default ResponseDisplay;