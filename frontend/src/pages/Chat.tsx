import React from 'react';
import ChatInterface from '../components/ChatInterface';

const Chat: React.FC = () => {
  return (
    <div className="min-h-[calc(100vh-12rem)] flex items-center justify-center">
      <div className="w-full max-w-4xl lg:max-w-5xl xl:max-w-6xl">
        <ChatInterface />
      </div>
    </div>
  );
};

export default Chat; 