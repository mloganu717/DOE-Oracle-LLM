import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import ChatInterface from './components/ChatInterface';
import './index.css';

// Placeholder components for other pages
const About = () => (
  <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
    <h1 className="text-3xl font-bold text-gray-900 mb-6">About CIO-Brain</h1>
    <div className="prose prose-blue max-w-none">
      <p className="text-lg text-gray-600">
        CIO-Brain is an intelligent research assistant powered by artificial intelligence that provides
        accurate information based on Dr. Wong's research on combinatorial testing.
      </p>
      <p className="text-lg text-gray-600 mt-4">
        Using advanced language models and retrieval-augmented generation (RAG), CIO-Brain can answer
        questions about software testing, fault detection, and other topics related to Dr. Wong's work.
      </p>
    </div>
  </div>
);

const Research = () => (
  <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
    <h1 className="text-3xl font-bold text-gray-900 mb-6">Research Papers</h1>
    <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      {[1, 2, 3].map((i) => (
        <div key={i} className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-6">
            <h3 className="text-lg font-medium text-gray-900">Research Paper {i}</h3>
            <p className="mt-2 text-gray-600">
              Combinatorial testing approaches for efficient fault detection and test case generation.
            </p>
            <div className="mt-4">
              <a href="#" className="text-blue-600 hover:text-blue-800">Read more &rarr;</a>
            </div>
          </div>
        </div>
      ))}
    </div>
  </div>
);

const Contact = () => (
  <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
    <h1 className="text-3xl font-bold text-gray-900 mb-6">Contact Us</h1>
    <div className="bg-white shadow overflow-hidden sm:rounded-lg">
      <div className="px-4 py-5 sm:p-6">
        <form className="space-y-6">
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-700">Name</label>
            <input type="text" id="name" className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500" />
          </div>
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">Email</label>
            <input type="email" id="email" className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500" />
          </div>
          <div>
            <label htmlFor="message" className="block text-sm font-medium text-gray-700">Message</label>
            <textarea id="message" rows={4} className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"></textarea>
          </div>
          <div>
            <button type="submit" className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
              Send Message
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
);

const Account = () => (
  <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
    <h1 className="text-3xl font-bold text-gray-900 mb-6">Your Account</h1>
    <div className="bg-white shadow overflow-hidden sm:rounded-lg">
      <div className="px-4 py-5 sm:p-6">
        <p className="text-gray-600">Account management features coming soon!</p>
      </div>
    </div>
  </div>
);

// Home page with chat interface
const Home = () => (
  <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
    <div className="mb-8 text-center">
      <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl">CIO-Brain Research Assistant</h1>
      <p className="mt-4 text-xl text-gray-600">
        Ask questions about Dr. Wong's research on combinatorial testing
      </p>
    </div>
    <div className="bg-white shadow-lg rounded-lg overflow-hidden h-[70vh]">
      <ChatInterface />
    </div>
  </div>
);

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/research" element={<Research />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/account" element={<Account />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
