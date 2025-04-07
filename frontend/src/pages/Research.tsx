import React from 'react';

const Research: React.FC = () => {
  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-pink-400 to-purple-400">
          Research & Publications
        </h1>
        <p className="mt-4 text-lg text-white/80 max-w-3xl mx-auto">
          Explore our latest research in combinatorial testing and software engineering methodologies.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Research Card 1 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 hover:border-pink-400/50 transition-all duration-300">
          <div className="h-48 bg-gradient-to-br from-pink-500/20 to-purple-500/20 rounded-lg mb-4 flex items-center justify-center">
            <svg className="w-16 h-16 text-white/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Combinatorial Testing</h3>
          <p className="text-white/70 mb-4">
            Advanced methodologies for efficient combinatorial test generation and analysis.
          </p>
          <a href="#" className="text-pink-400 hover:text-pink-300 transition-colors duration-300">
            Read More →
          </a>
        </div>

        {/* Research Card 2 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 hover:border-purple-400/50 transition-all duration-300">
          <div className="h-48 bg-gradient-to-br from-purple-500/20 to-indigo-500/20 rounded-lg mb-4 flex items-center justify-center">
            <svg className="w-16 h-16 text-white/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">Software Testing</h3>
          <p className="text-white/70 mb-4">
            Innovative approaches to software testing and quality assurance.
          </p>
          <a href="#" className="text-purple-400 hover:text-purple-300 transition-colors duration-300">
            Read More →
          </a>
        </div>

        {/* Research Card 3 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 hover:border-indigo-400/50 transition-all duration-300">
          <div className="h-48 bg-gradient-to-br from-indigo-500/20 to-blue-500/20 rounded-lg mb-4 flex items-center justify-center">
            <svg className="w-16 h-16 text-white/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-white mb-2">AI in Testing</h3>
          <p className="text-white/70 mb-4">
            Leveraging artificial intelligence for automated test generation and analysis.
          </p>
          <a href="#" className="text-indigo-400 hover:text-indigo-300 transition-colors duration-300">
            Read More →
          </a>
        </div>
      </div>

      <div className="mt-12 text-center">
        <h2 className="text-2xl font-bold text-white mb-4">Recent Publications</h2>
        <div className="space-y-4 max-w-3xl mx-auto">
          <div className="bg-white/5 backdrop-blur-lg rounded-lg p-4 border border-white/10">
            <h3 className="text-lg font-semibold text-white">"Advanced Combinatorial Testing Techniques"</h3>
            <p className="text-white/70">Journal of Software Testing, 2024</p>
          </div>
          <div className="bg-white/5 backdrop-blur-lg rounded-lg p-4 border border-white/10">
            <h3 className="text-lg font-semibold text-white">"AI-Powered Test Generation"</h3>
            <p className="text-white/70">International Conference on Software Engineering, 2023</p>
          </div>
          <div className="bg-white/5 backdrop-blur-lg rounded-lg p-4 border border-white/10">
            <h3 className="text-lg font-semibold text-white">"Optimizing Test Coverage"</h3>
            <p className="text-white/70">Software Quality Journal, 2023</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Research; 