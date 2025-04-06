import { useState, useEffect, useRef } from 'react';

const ChatInterface = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [useRAG, setUseRAG] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Test connection on component mount
  useEffect(() => {
    testConnection();
  }, []);

  const testConnection = async () => {
    setConnectionStatus('Testing connection to backend...');
    try {
      // Simple fetch to test CORS and connectivity
      const response = await fetch('/prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: 'test',
          use_rag: false
        })
      });
      
      console.log('Test connection response:', response.status);
      
      if (response.ok) {
        setConnectionStatus('Connected to backend successfully! ✅');
      } else {
        setConnectionStatus(`Connection error: ${response.status} ${response.statusText} ❌`);
      }
    } catch (error) {
      console.error('Connection test error:', error);
      setConnectionStatus(`Connection failed: ${error} ❌`);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setIsLoading(true);

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);

    try {
      console.log('Sending request to /prompt via proxy');
      const response = await fetch('/prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: userMessage,
          use_rag: useRAG
        })
      });

      console.log('Response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`Network response was not ok: ${response.status} ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No reader available');
      }

      let assistantMessage = '';
      setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = new TextDecoder().decode(value);
        console.log('Received chunk:', text);
        assistantMessage += text;
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = {
            role: 'assistant',
            content: assistantMessage
          };
          return newMessages;
        });
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, there was an error processing your request. Check the console for details.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="flex flex-col h-[600px]">
        {/* Connection Status */}
        {connectionStatus && (
          <div className={`p-2 text-sm text-center ${connectionStatus.includes('✅') ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
            {connectionStatus}
            <button 
              onClick={testConnection}
              className="ml-2 underline"
            >
              Retry
            </button>
          </div>
        )}
        
        {/* RAG Toggle */}
        <div className="p-4 border-b bg-gray-50">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={useRAG}
              onChange={(e) => setUseRAG(e.target.checked)}
              className="h-4 w-4 text-blue-500 rounded focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700">
              Use RAG (Retrieval-Augmented Generation)
            </span>
          </label>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 ${
                  message.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
              </div>
            </div>
          ))}
          {messages.length === 0 && (
            <div className="text-center text-gray-500">
              <p>Start a conversation by sending a message!</p>
              <p className="text-sm mt-2">
                {useRAG 
                  ? "RAG is enabled - responses will be based on the research papers"
                  : "RAG is disabled - responses will be based on general knowledge"}
              </p>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <div className="border-t p-4">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading}
              className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Sending...' : 'Send'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface; 