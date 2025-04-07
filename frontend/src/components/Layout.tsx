import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import Logo from './Logo';
import { NavLink } from 'react-router-dom';

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 text-white">
      {/* Animated background */}
      <div className="fixed inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-indigo-500/20 via-purple-500/20 to-pink-500/20 animate-pulse" />
      
      {/* Main container */}
      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Header with glassmorphism effect */}
        <header className="sticky top-0 z-50 backdrop-blur-lg bg-white/10 border-b border-white/20">
          <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center">
                <Link to="/" className="flex items-center space-x-2 group">
                  <Logo className="h-8 w-8 text-white group-hover:text-pink-400 transition-colors duration-300" />
                  <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-pink-400 to-purple-400 group-hover:from-purple-400 group-hover:to-pink-400 transition-all duration-300">
                    DOE Oracle
                  </span>
                </Link>
              </div>
              
              <div className="flex items-center space-x-4">
                <NavLink
                  to="/"
                  className={({ isActive }) =>
                    `text-white/70 hover:text-white transition-colors ${isActive ? 'text-white font-semibold' : ''}`
                  }
                >
                  Home
                </NavLink>
                <NavLink
                  to="/chat"
                  className={({ isActive }) =>
                    `text-white/70 hover:text-white transition-colors ${isActive ? 'text-white font-semibold' : ''}`
                  }
                >
                  Chat
                </NavLink>
                <NavLink
                  to="/research"
                  className={({ isActive }) =>
                    `text-white/70 hover:text-white transition-colors ${isActive ? 'text-white font-semibold' : ''}`
                  }
                >
                  Research
                </NavLink>
                <NavLink
                  to="/about"
                  className={({ isActive }) =>
                    `text-white/70 hover:text-white transition-colors ${isActive ? 'text-white font-semibold' : ''}`
                  }
                >
                  About
                </NavLink>
                <NavLink
                  to="/contact"
                  className={({ isActive }) =>
                    `text-white/70 hover:text-white transition-colors ${isActive ? 'text-white font-semibold' : ''}`
                  }
                >
                  Contact
                </NavLink>
              </div>
            </div>
          </nav>
        </header>

        {/* Main content */}
        <main className="flex-1">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            {children}
          </div>
        </main>

        {/* Footer with gradient border */}
        <footer className="relative z-10 border-t border-white/20 bg-gradient-to-r from-indigo-900/50 via-purple-900/50 to-pink-900/50 backdrop-blur-lg mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Logo className="h-6 w-6 text-white/70" />
                <span className="text-sm text-white/70">© 2024 DOE Oracle</span>
              </div>
              <div className="flex items-center space-x-4">
                <a href="#" className="text-white/70 hover:text-white transition-colors duration-300">
                  Privacy
                </a>
                <a href="#" className="text-white/70 hover:text-white transition-colors duration-300">
                  Terms
                </a>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default Layout; 