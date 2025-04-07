import React from 'react';

interface LogoProps {
  className?: string;
}

const Logo: React.FC<LogoProps> = ({ className = '' }) => {
  return (
    <svg 
      xmlns="http://www.w3.org/2000/svg" 
      viewBox="0 0 200 200"
      className={className}
    >
      <defs>
        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#ff6b6b" />
          <stop offset="50%" stopColor="#4ecdc4" />
          <stop offset="100%" stopColor="#45b7d1" />
        </linearGradient>
      </defs>
      <rect width="200" height="200" fill="url(#gradient)"/>
      <circle cx="100" cy="100" r="80" fill="white" opacity="0.2"/>
      <text 
        x="100" 
        y="115" 
        textAnchor="middle" 
        fontFamily="Arial" 
        fontSize="40" 
        fill="white" 
        fontWeight="bold"
      >
        DOE
      </text>
    </svg>
  );
};

export default Logo; 