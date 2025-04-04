import os
from langchain_ollama.chat_models import ChatOllama
from ciobrain.admin.documents.rag_manager import RAGManager
from flask import current_app
import logging
import time
from mlx_lm import load, generate

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
        self.adapter_path = "/Users/logan/DOE-ORACLE-LLM-1/ciobrain/fine_tuning/adapters"
        
        # Load the MLX model and tokenizer
        logging.info(f"Loading MLX model from {self.mlx_model_name}")
        try:
            self.model, self.tokenizer = load(
                self.mlx_model_name,
                adapter_path=self.adapter_path
            )
            logging.info("MLX model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading MLX model: {str(e)}")
            self.model = None
            self.tokenizer = None
        
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
            self.rag_chain = self.rag_manager.create_chain(
                self.rag_manager.create_retriever(self.vector_db, self.rag_llm),
                self.rag_llm
            )
            
            self.initialized = True
            logging.info("All resources initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Error initializing resources: {str(e)}")
            return False

    def stream(self, conversation, use_rag=True):
        """Stream responses from the model."""
        try:
            if not self.initialized:
                if not self.initialize_resources():
                    yield "Error: Failed to initialize resources. Please try again later."
                    return

            # Extract the last message from the conversation
            if isinstance(conversation, list) and conversation:
                last_message = conversation[-1]
                if isinstance(last_message, dict):
                    prompt = last_message.get('content', '')
                else:
                    prompt = str(last_message)
            else:
                prompt = str(conversation)

            if not prompt:
                yield "Error: No prompt provided"
                return

            if use_rag and self.rag_chain:
                # Use RAG chain for response
                for response in self.rag_chain(prompt):  # No longer passing max_tokens here
                    if isinstance(response, str):
                        yield response
                    else:
                        yield str(response)
            else:
                # Use direct MLX model response
                if not self.model or not self.tokenizer:
                    yield "Error: MLX model not properly initialized"
                    return

                # Load system prompt
                system_prompt_path = "/Users/logan/DOE-ORACLE-LLM-1/ciobrain/fine_tuning/system_prompt.txt"
                with open(system_prompt_path, 'r') as f:
                    system_prompt = f.read().strip()

                # Format the prompt using the tokenizer's chat template with system prompt
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Generate response using MLX's generate function
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=2048,
                    temperature=0.7,
                    verbose=True
                )
                
                # Split response into sentences for better formatting
                sentences = response.split('. ')
                for sentence in sentences:
                    if sentence.strip():
                        # Add newline after each sentence
                        yield sentence.strip() + '. \n'

        except Exception as e:
            logging.error(f"Error in stream: {str(e)}")
            yield f"Error: {str(e)}"
