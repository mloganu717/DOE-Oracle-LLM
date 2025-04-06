import os
import sys
import logging
from typing import List, Dict, Any, Optional
from flask import current_app
import json

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from ciobrain.admin.documents.rag_manager import RAGManager
import time
from mlx_lm import load, generate
import re

class Mediator:
    """Mediator class for handling interactions between frontend and backend services."""
    
    def __init__(self):
        """Initialize Mediator instance."""
        self.model = None
        self.rag_manager = None
        self.rag_chain = None
        self.model_initialized = False
        self.rag_initialized = False
        
        # Initialize RAG with Ollama
        self.rag_llm = ChatOllama(
            model="llama3.2",
            max_tokens=2048,
            temperature=0.7
        )
        
        # Initialize MLX model settings
        self.mlx_model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        self.adapter_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                       "fine_tuning", "adapters")
        
        # Track adapter usage
        self.using_adapter = False
        self.tokenizer = None
        
        # Try to load the fine-tuned model
        logging.info(f"Attempting to load MLX model")
        try:
            # Check if adapter directory exists and has files
            adapter_exists = os.path.exists(self.adapter_path) and len(os.listdir(self.adapter_path)) > 0
            
            if adapter_exists:
                # List files in adapter directory for debugging
                adapter_files = os.listdir(self.adapter_path)
                logging.info(f"Found adapter files: {adapter_files}")
                
                try:
                    # Try loading with adapter
                    logging.info(f"Loading adapter from: {self.adapter_path}")
                    self.model, self.tokenizer = load(
                        self.mlx_model_name,
                        adapter_path=self.adapter_path
                    )
                    self.using_adapter = True
                    logging.info("Successfully loaded MLX model with adapter")
                    
                    # Test the adapter with a simple prompt to see what it returns
                    test_prompt = "What is combinatorial testing?"
                    system_prompt = "You are an AI assistant with expertise in software testing, particularly Dr. Wong's research at UT Dallas on combinatorial testing. Provide accurate, technical responses based on academic research."
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": test_prompt}
                    ]
                    formatted_test = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    logging.info("Testing adapter with prompt: 'What is combinatorial testing?'")
                    test_response = generate(
                        self.model,
                        self.tokenizer,
                        prompt=formatted_test,
                        max_tokens=2048
                    )
                    logging.info(f"Test response: {test_response[:100]}...")
                    
                except Exception as adapter_error:
                    logging.error(f"Failed to load adapter: {str(adapter_error)}")
                    logging.exception("Adapter loading exception details:")
            else:
                logging.info("No adapter files found, using base model")
        except Exception as e:
            logging.error(f"Error checking adapter path: {str(e)}")
            
        # If adapter loading failed, use base model
        if self.model is None:
            logging.info("Loading base MLX model")
            try:
                self.model, self.tokenizer = load(
                    self.mlx_model_name
                )
                logging.info("Successfully loaded base MLX model")
            except Exception as e:
                logging.error(f"Error loading base MLX model: {str(e)}")
        
        # Initialize RAGManager
        self.rag_manager = RAGManager()
        self.vector_db = None
        self.retriever = None
        self.initialized = False

    def initialize_resources(self):
        """Initialize all necessary resources for the mediator"""
        try:
            # Initialize the LLM model and embeddings
            self.model_initialized = self._initialize_model()
            
            # Initialize the RAG resources
            self.rag_initialized = self._initialize_rag_resources()
            
            # Log status
            if self.model_initialized and self.rag_initialized:
                logging.info("All resources initialized successfully")
            elif self.model_initialized:
                logging.info("Model initialized successfully, but RAG resources failed")
            elif self.rag_initialized:
                logging.info("RAG resources initialized successfully, but model failed")
            else:
                logging.error("Failed to initialize any resources")
            
            return self.model_initialized
        except Exception as e:
            logging.error(f"Error initializing resources: {str(e)}")
            return False

    def _initialize_rag_resources(self):
        """Initialize RAG-specific resources"""
        try:
            # Create RAG manager instance
            from ciobrain.admin.documents.rag_manager import RAGManager
            self.rag_manager = RAGManager()
            
            # Check if vector store exists and initialize it if needed
            if not self.rag_manager.vector_store_exists():
                logging.info("Vector store doesn't exist, creating it...")
                self.rag_manager.create_vector_store()
            else:
                logging.info("Vector store exists, loading it...")
                self.rag_manager.load_vector_store()
            
            # Initialize RAG chain
            self.rag_chain = self.rag_manager.get_rag_chain()
            
            if self.rag_chain is None:
                logging.error("Failed to initialize RAG chain")
                return False
            
            logging.info("RAG resources initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Error initializing RAG resources: {str(e)}")
            return False

    def stream(self, messages, use_rag=False):
        """Stream a response to the given prompt."""
        try:
            # Check if RAG is explicitly enabled
            if use_rag:
                logging.info("Using RAG for response generation")
                
                # Extract the query from the messages
                try:
                    # Handle different message formats
                    if isinstance(messages, list):
                        if messages and isinstance(messages[-1], dict) and 'content' in messages[-1]:
                            query_str = messages[-1]['content']
                        else:
                            query_str = str(messages[-1])
                    elif isinstance(messages, dict) and 'content' in messages:
                        query_str = messages['content']
                    elif isinstance(messages, str):
                        query_str = messages
                    else:
                        query_str = str(messages)
                    
                    logging.info(f"Extracted query string for RAG: '{query_str}'")
                    
                    # Initialize RAG resources if not already done
                    if not hasattr(self, 'rag_manager') or self.rag_manager is None:
                        logging.info("Initializing RAG resources...")
                        try:
                            from ciobrain.admin.documents.rag_manager import RAGManager
                            self.rag_manager = RAGManager()
                        except Exception as e:
                            logging.error(f"Error initializing RAG resources: {str(e)}")
                            yield "I couldn't access the research papers database at this moment. Let me answer based on general knowledge instead.\n\n"
                            use_rag = False
                    
                    # If RAG is still enabled, use it
                    if use_rag:
                        try:
                            # Process the query directly using the RAGManager
                            logging.info(f"Processing query using RAGManager: '{query_str}'")
                            response = self.rag_manager.process_query(query_str)
                            
                            # Process the response
                            if response:
                                logging.info(f"RAG response received: {response[:100]}...")
                                
                                # Stream the response sentence by sentence
                                import re
                                sentences = re.split(r'(?<=[.!?])\s+', response)
                                for sentence in sentences:
                                    if sentence.strip():
                                        yield sentence + " "
                                return
                            else:
                                raise ValueError("Empty response from RAG chain")
                                
                        except Exception as e:
                            logging.error(f"Error using RAG chain: {str(e)}")
                            yield "I encountered an issue retrieving information from the research papers. Let me answer based on my general knowledge instead.\n\n"
                            use_rag = False
                except Exception as e:
                    logging.error(f"Error extracting query or initializing RAG: {str(e)}")
                    yield "I encountered a technical issue processing your request. Let me answer based on my general knowledge instead.\n\n"
                    use_rag = False
            
            # Fall back to the base model if not using RAG or if RAG failed
            logging.info("Using adapter model for response generation")
            
            # Extract the last message content if it's a list of messages
            if isinstance(messages, list) and len(messages) > 0:
                prompt = messages[-1]['content']
            else:
                prompt = messages
                
            # Load system prompt
            system_prompt = self._load_system_prompt()
            
            # Format prompt with system and user messages
            adapter_prompt = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"{system_prompt}\n"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{prompt}\n"
                "<|start_header_id|>assistant<|end_header_id|>\n"
            )
            
            logging.info(f"Adapter prompt: {adapter_prompt[:100]}...")
            
            # Generate response
            response = generate(
                self.model,
                self.tokenizer,
                prompt=adapter_prompt,
                max_tokens=2048
            )
            
            # Clean up response if needed
            if response:
                logging.info(f"Raw response: {response[:100]}...")
                yield response
            else:
                yield "No response generated."
                
        except Exception as e:
            logging.error(f"Error in stream method: {str(e)}")
            yield f"I encountered a technical issue processing your request. Please try again with a simpler question or without using the research papers database."

    def _load_system_prompt(self):
        """Load the system prompt for the assistant"""
        system_prompt = """
Cutting Knowledge Date: December 2023
The AI assistant is an expert in Dr. Wong's research on combinatorial testing at UT Dallas.
It provides accurate technical information about software testing techniques and applies
combinatorial testing concepts to help solve testing problems.

Key areas of expertise:
- Combinatorial testing theory and applications
- Software testing methodologies
- Test case generation and optimization
- Fault detection techniques
- Industrial applications of combinatorial testing

The assistant is helpful, clear, accurate, and provides responses based on academic
research in the field of software testing.
"""
        return system_prompt.strip()

    def _initialize_model(self):
        """Initialize the MLX model and tokenizer"""
        try:
            logging.info("Attempting to load MLX model")
            
            # Set up model paths
            adapter_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fine_tuning", "adapters")
            adapter_files = []
            
            if os.path.exists(adapter_path):
                adapter_files = os.listdir(adapter_path)
                logging.info(f"Found adapter files: {adapter_files}")
            else:
                logging.error(f"Adapter path not found: {adapter_path}")
                return False
            
            try:
                # Import here to avoid circular imports
                from mlx_lm import load, generate
                
                # Initialize model
                self.model, self.tokenizer = load(
                    self.mlx_model_name,
                    adapter_path=adapter_path
                )
                
                # Test the model with a simple prompt
                test_prompt = "What is combinatorial testing?"
                adapter_prompt = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                    f"{self._load_system_prompt()}\n"
                    "<|start_header_id|>user<|end_header_id|>\n"
                    f"{test_prompt}\n"
                    "<|start_header_id|>assistant<|end_header_id|>\n"
                )
                test_response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=adapter_prompt,
                    max_tokens=2048
                )
                logging.info(f"Test response: {test_response[:100]}...")
                
                logging.info("Successfully loaded MLX model with adapter")
                return True
            except Exception as e:
                logging.error(f"Error loading MLX model: {str(e)}")
                return False
        except Exception as e:
            logging.error(f"Error in model initialization: {str(e)}")
            return False