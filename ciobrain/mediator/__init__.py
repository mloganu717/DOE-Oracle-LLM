import os
from langchain_ollama.chat_models import ChatOllama
from ciobrain.admin.documents.rag_manager import RAGManager
from flask import current_app
import logging
import time
from mlx_lm import load, generate
import re

class Mediator:
    def __init__(self):
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
        self.model = None
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
        self.rag_chain = None
        self.initialized = False

    def initialize_resources(self):
        """Initialize all required resources."""
        try:
            # Initialize RAG manager
            self.rag_manager.initialize(current_app._get_current_object())
            
            # Process documents to create vector database
            self.vector_db = self.rag_manager.process_documents()
            if not self.vector_db:
                logging.error("Failed to create vector database from research papers")
                return False
            
            # Create RAG chain
            self.rag_chain = self.rag_manager.create_rag_chain()
            
            self.initialized = True
            logging.info("All resources initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Error initializing resources: {str(e)}")
            return False

    def stream(self, conversation, use_rag=True):
        """Stream responses from the model."""
        try:
            # Extract the last message from the conversation
            if isinstance(conversation, list) and conversation:
                last_message = conversation[-1]
                if isinstance(last_message, dict):
                    prompt = last_message.get('content', '')
                    # Check for RAG toggle in the message content
                    if "Use RAG: False" in prompt:
                        use_rag = False
                        prompt = prompt.replace("Use RAG: False", "").strip()
                    elif "Use RAG: True" in prompt:
                        use_rag = True
                        prompt = prompt.replace("Use RAG: True", "").strip()
                else:
                    prompt = str(last_message)
            else:
                prompt = str(conversation)

            if not prompt:
                yield "Error: No prompt provided"
                return

            if use_rag:
                # Initialize RAG resources if needed
                if not self.initialized:
                    if not self.initialize_resources():
                        yield "Error: Failed to initialize RAG resources. Please try again later."
                        return

                if not self.rag_chain:
                    yield "Error: RAG chain not properly initialized"
                    return

                try:
                    # Use RAG chain for response
                    response = self.rag_chain(prompt)
                    
                    # Handle different response types
                    if isinstance(response, dict):
                        # Extract the answer from the dictionary response
                        response_text = response.get('answer', '')
                        if not response_text:
                            response_text = str(response)
                    else:
                        response_text = str(response)
                    
                    if response_text:
                        # Split response into sentences for better streaming
                        sentences = response_text.split('. ')
                        for sentence in sentences:
                            if sentence.strip():
                                yield sentence.strip() + '. '
                    else:
                        yield "Error: RAG chain returned empty response"
                except Exception as e:
                    logging.error(f"Error in RAG processing: {str(e)}")
                    yield f"Error in RAG processing: {str(e)}"
            else:
                # Use direct MLX model response
                if not self.model or not self.tokenizer:
                    yield "Error: MLX model not properly initialized"
                    return

                try:
                    # Format prompt based on whether we're using adapter or base model
                    if self.using_adapter:
                        system_prompt = "You are an AI assistant with expertise in software testing, particularly Dr. Wong's research at UT Dallas on combinatorial testing. Provide accurate, technical responses based on academic research."
                        
                        # Format using structure from training data
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                        
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        
                        # Use smaller max_tokens for adapter model
                        max_tokens = 150
                        logging.info("Using adapter model for response generation")
                        logging.info(f"Adapter prompt: {formatted_prompt[:100]}...")
                        
                        # Try generating with adapter
                        adapter_response = generate(
                            self.model,
                            self.tokenizer,
                            prompt=formatted_prompt,
                            max_tokens=max_tokens
                        )
                        
                        # If response seems corrupted, fall back to base model
                        if "Journal of Music Health" in adapter_response or "Testosterone-Based Diet" in adapter_response:
                            logging.warning("Adapter response contains incorrect information. May be corrupted.")
                            logging.warning(f"Corrupt adapter response: {adapter_response[:100]}...")
                            
                            # Fall back to base model
                            logging.info("Falling back to base model")
                            formatted_prompt = f"Question: {prompt}\n\nAnswer:"
                            max_tokens = 250
                            
                            response = generate(
                                self.model,
                                self.tokenizer,
                                prompt=formatted_prompt,
                                max_tokens=max_tokens
                            )
                        else:
                            response = adapter_response
                    else:
                        # Simpler format for base model
                        formatted_prompt = f"Question: {prompt}\n\nAnswer:"
                        max_tokens = 250
                        logging.info("Using base model for response generation")
                        
                        # Generate response
                        response = generate(
                            self.model,
                            self.tokenizer,
                            prompt=formatted_prompt,
                            max_tokens=max_tokens
                        )
                    
                    # Process response regardless of model type
                    clean_response = response.strip()
                    logging.info(f"Raw response: {clean_response[:100]}...")
                    
                    # Try to find start of actual response
                    markers = ["Answer:", "<|assistant|>", "Assistant:"]
                    for marker in markers:
                        if marker in clean_response:
                            parts = clean_response.split(marker, 1)
                            if len(parts) > 1:
                                clean_response = parts[1].strip()
                                logging.info(f"Found marker: {marker}, extracted: {clean_response[:100]}...")
                                break
                    
                    # Simple response check
                    if clean_response and len(clean_response) > 10:
                        yield clean_response
                    else:
                        yield "I apologize, but I couldn't generate a proper response."
                            
                except Exception as e:
                    logging.error(f"Error in MLX processing: {str(e)}")
                    logging.exception("MLX processing exception details:")
                    yield f"Error in MLX processing: {str(e)}"

        except Exception as e:
            logging.error(f"Error in stream: {str(e)}")
            yield f"Error: {str(e)}"