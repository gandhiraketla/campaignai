import React from 'react';
import { Rocket } from 'lucide-react';

function Header() {
  return (
    <header className="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-500 shadow-lg">
      <div className="container mx-auto flex items-center justify-between px-6 py-4">
        {/* Logo and Brand */}
        <div className="flex items-center space-x-4">
          <div className="bg-white/20 p-3 rounded-full backdrop-blur-md">
            <Rocket className="text-white" size={32} />
          </div>
          <h1 className="text-4xl font-extrabold text-white tracking-tight">
            CampaignAI
          </h1>
        </div>
        
        {/* Navigation */}
        <nav className="flex items-center space-x-6">
          <a 
            href="#" 
            className="text-white/80 hover:text-white transition-colors duration-300 
                       font-medium text-lg flex items-center space-x-2 
                       hover:bg-white/10 px-3 py-2 rounded-lg"
          >
            <span>Home</span>
          </a>
          <a 
            href="#" 
            className="text-white/80 hover:text-white transition-colors duration-300 
                       font-medium text-lg flex items-center space-x-2 
                       hover:bg-white/10 px-3 py-2 rounded-lg"
          >
            <span>Insights</span>
          </a>
        </nav>
      </div>
    </header>
  );
}

export default Header;