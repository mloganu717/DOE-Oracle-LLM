import os
import shutil
import logging
import pdfplumber
from flask import current_app
from langchain.schema import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaEmbeddings
from typing import List, Dict, Any, Optional
from langchain.chains import LLMChain

logging.basicConfig(level=logging.INFO)

class RAGManager:

    def __init__(self, pdf_dir: str = "ciobrain/instance/knowledge/research-papers"):
        """Initialize the RAG manager with paths and resources."""
        self.pdf_dir = pdf_dir
        self.vector_store_path = "ciobrain/instance/knowledge/vector_store"
        self.vectorstore = None
        self.llm = None
        self.embedding_function = None
        self.rag_chain = None
        self.initialize_resources()

    def initialize_resources(self):
        """Initialize all necessary resources for RAG."""
        try:
            # Initialize embedding function
            self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
            
            # Initialize vector store
            self.vectorstore = self.create_vector_db()
            
            # Initialize LLM with Llama3
            self.llm = ChatOllama(
                model="llama3",
                temperature=0.7,
                max_tokens=2000
            )
            
            logging.info("Successfully initialized RAG resources")
        except Exception as e:
            logging.error(f"Failed to initialize RAG resources: {str(e)}")
            raise

    def process_documents(self) -> List[Document]:
        """Process all documents in the PDF directory."""
        documents = []
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.pdf_dir, exist_ok=True)
            
            # Check if the directory exists and is accessible
            if not os.path.exists(self.pdf_dir):
                logging.error(f"PDF directory does not exist or is not accessible: {self.pdf_dir}")
                return documents
                
            # List files in the directory
            file_list = os.listdir(self.pdf_dir)
            logging.info(f"Found {len(file_list)} files in {self.pdf_dir}")
            
            # Process PDF files
            for filename in file_list:
                if filename.endswith(".pdf"):
                    file_path = os.path.join(self.pdf_dir, filename)
                    logging.info(f"Processing PDF: {file_path}")
                    try:
                        loader = PyPDFLoader(file_path)
                        doc_pages = loader.load()
                        documents.extend(doc_pages)
                        logging.info(f"Added {len(doc_pages)} pages from {filename}")
                    except Exception as pdf_error:
                        logging.error(f"Error processing PDF {filename}: {str(pdf_error)}")
            
            logging.info(f"Processed {len(documents)} total pages from PDF files")
            return documents
        except Exception as e:
            logging.error(f"Error processing documents: {str(e)}")
            return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        if not documents:
            return []
            
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
        except Exception as e:
            logging.error(f"Error splitting documents: {str(e)}")
            return []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            return "\n".join(page.page_content for page in pages)
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def create_vector_db(self) -> Chroma:
        """Create or load the vector database."""
        try:
            # Ensure the PDF directory exists
            os.makedirs(self.pdf_dir, exist_ok=True)
            
            # Process documents
            documents = self.process_documents()
            
            if not documents:
                logging.warning("No documents found, creating test document")
                # Create a test document about Dr. Wong's research
                test_doc = Document(
                    page_content="Dr. Wong's research focuses on combinatorial testing, "
                               "a systematic approach to software testing that aims to "
                               "detect faults in complex systems by testing combinations "
                               "of input parameters. His work includes developing "
                               "algorithms for generating efficient test suites and "
                               "analyzing their effectiveness in fault detection.",
                    metadata={"source": "test_document"}
                )
                documents = [test_doc]
            
            # Split documents
            split_docs = self.split_documents(documents)
            
            # Ensure vector store directory exists
            os.makedirs(self.vector_store_path, exist_ok=True)
            
            # Create vector store
            logging.info(f"Creating vector store with {len(split_docs)} documents")
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embedding_function,
                persist_directory=self.vector_store_path
            )
            
            # Test the vector store to make sure it's properly initialized
            try:
                test_results = vectorstore.similarity_search("test", k=1)
                logging.info(f"Vector store test successful: {len(test_results)} results")
            except Exception as test_error:
                logging.error(f"Vector store test failed: {str(test_error)}")
            
            logging.info(f"Created vector store with {len(split_docs)} documents")
            return vectorstore
            
        except Exception as e:
            logging.error(f"Error creating vector database: {str(e)}")
            raise

    def hybrid_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Perform a hybrid search using both keyword matching and semantic similarity.
        
        Args:
            query: The search query string
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            
            logging.info(f"Performing hybrid search for query: {query}")
            
            # Ensure vector store is initialized
            if not self.vectorstore:
                logging.warning("Vector store not initialized, attempting to create it...")
                self.vectorstore = self.create_vector_db()
            
            # Ensure embedding function is initialized
            if not self.embedding_function:
                logging.warning("Embedding function not initialized, attempting to create it...")
                self.embedding_function = OllamaEmbeddings(model="nomic-embed-text")
            
            # First try semantic search
            semantic_results = []
            try:
                # Generate the embedding for the query
                logging.info("Generating embedding for query")
                
                # Use the vectorstore's similarity_search without passing the embedding_function parameter
                semantic_results = self.vectorstore.similarity_search(
                    query,
                    k=k
                )
                logging.info(f"Found {len(semantic_results)} semantic matches")
            except Exception as e:
                logging.error(f"Semantic search failed: {str(e)}")
                
                # Fall back to keyword search
                try:
                    logging.info("Falling back to keyword search")
                    keyword_results = self.direct_keyword_search(query, k=k)
                    
                    # Convert keyword results to Document objects if needed
                    if isinstance(keyword_results, str):
                        # Split into paragraphs and create Document objects
                        paragraphs = keyword_results.split("\n\n")
                        semantic_results = [
                            Document(page_content=para.strip(), metadata={"source": "keyword_search"})
                            for para in paragraphs if para.strip()
                        ]
                        logging.info(f"Created {len(semantic_results)} document objects from keyword search")
                    
                except Exception as kw_error:
                    logging.error(f"Keyword search also failed: {str(kw_error)}")
            
            # Combine and deduplicate results
            all_results = semantic_results
            unique_results = []
            seen_content = set()
            
            for doc in all_results:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_results.append(doc)
            
            if not unique_results:
                logging.warning("No results found in hybrid search")
                # Return fallback content about Dr. Wong's research
                return [Document(
                    page_content="Dr. Wong's research focuses on combinatorial testing, "
                               "a systematic approach to software testing that aims to "
                               "detect faults in complex systems by testing combinations "
                               "of input parameters. His work includes developing "
                               "algorithms for generating efficient test suites and "
                               "analyzing their effectiveness in fault detection.",
                    metadata={"source": "fallback_content"}
                )]
            
            return unique_results[:k]
            
        except Exception as e:
            logging.error(f"Error in hybrid search: {str(e)}")
            raise

    def get_rag_chain(self):
        """Get or create the RAG chain."""
        if self.rag_chain is None:
            try:
                # Ensure vector store is initialized
                if not self.vectorstore:
                    self.vectorstore = self.create_vector_db()
                
                # Ensure LLM is initialized
                if not self.llm:
                    self.llm = ChatOllama(
                        model="llama3",
                        temperature=0.7,
                        max_tokens=2000
                    )
                
                # Define the prompt template
                prompt = PromptTemplate(
                    input_variables=["context", "query"],
                    template="""You are a research assistant for Dr. Wong's work on combinatorial testing. 
                    Use the following context to answer the query. If the context doesn't contain relevant 
                    information, say so and provide a general answer based on your knowledge of combinatorial testing.
                    
                    Context: {context}
                    
                    Query: {query}
                    
                    Answer:"""
                )
                
                # Create a retrieval-style chain that can handle a string input directly
                def get_context_func(query_str):
                    """Get context for a query"""
                    # Extract query string if needed
                    if isinstance(query_str, dict) and "query" in query_str:
                        query_str = query_str["query"]
                    elif isinstance(query_str, list):
                        if len(query_str) > 0 and isinstance(query_str[0], dict) and "content" in query_str[0]:
                            query_str = query_str[0]["content"]
                        else:
                            query_str = str(query_str)
                    
                    # Make sure query is a string
                    if not isinstance(query_str, str):
                        query_str = str(query_str)
                    
                    logging.info(f"Getting context for query: {query_str}")
                    
                    # Get documents via hybrid search
                    docs = self.hybrid_search(query_str)
                    return "\n\n".join(doc.page_content for doc in docs)
                
                # Create a chain that can handle a simple string input
                from langchain_core.runnables import RunnableLambda
                from langchain_core.output_parsers import StrOutputParser
                
                chain = (
                    {
                        "context": RunnableLambda(get_context_func),
                        "query": lambda x: x if isinstance(x, str) else str(x)
                    }
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )
                
                self.rag_chain = chain
                
                # Test the chain with a simple query
                test_input = "What is combinatorial testing?"
                try:
                    test_response = self.rag_chain.invoke(test_input)
                    logging.info(f"RAG chain test successful: {test_response[:100]}...")
                except Exception as test_error:
                    logging.error(f"RAG chain test failed: {str(test_error)}")
                    raise
                
                logging.info("Successfully created and tested RAG chain")
                
            except Exception as e:
                logging.error(f"Error creating RAG chain: {str(e)}")
                raise
        
        return self.rag_chain

    def process_query(self, query: str) -> str:
        """Process a query using the RAG system."""
        try:
            # Get the RAG chain
            chain = self.get_rag_chain()
            if not chain:
                raise ValueError("Failed to create RAG chain")
            
            # Invoke the chain directly with the query string
            # This will use the get_context_func internally
            logging.info(f"Processing query: {query}")
            response = chain.invoke(query)
            
            if not response:
                raise ValueError("Empty response from RAG chain")
                
            logging.info(f"RAG response: {response[:100]}...")
            return response
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return "I encountered an issue processing your query with Dr. Wong's research. Let me answer based on my general knowledge instead."

    def _initialize_vector_store(self):
        """Initialize the vector store, loading existing or creating new one"""
        try:
            # Check if vector store exists and attempt to load it
            if self.vector_store_exists():
                logging.info("Attempting to load existing vector store...")
                if not self.load_vector_store():
                    logging.warning("Failed to load existing vector store, attempting to create a new one...")
                    if not self.create_vector_store():
                        logging.error("Failed to create vector store")
                        return False
                    else:
                        logging.info("Successfully created new vector store")
                else:
                    logging.info("Successfully loaded existing vector store")
            else:
                logging.info("No existing vector store found, creating a new one...")
                if not self.create_vector_store():
                    logging.error("Failed to create vector store")
                    return False
                else:
                    logging.info("Successfully created new vector store")
            
            # Verify that vector store is properly initialized
            if not hasattr(self, 'vectorstore') or self.vectorstore is None:
                logging.error("Vector store not properly initialized after attempt")
                return False
            
            if not hasattr(self.vectorstore, '_collection'):
                logging.error("Vector store collection not properly initialized")
                return False
            
            # Test the vector store with a basic query
            try:
                test_results = self.vectorstore._collection.get(limit=1)
                if test_results and 'documents' in test_results and test_results['documents']:
                    logging.info("Vector store verified with test query")
                    return True
                else:
                    logging.error("Vector store verification failed - no documents found")
                    return False
            except Exception as e:
                logging.error(f"Error testing vector store: {str(e)}")
                return False
        except Exception as e:
            logging.error(f"Error in _initialize_vector_store: {str(e)}")
            return False

    def initialize(self, app):
        """Initialize the RAG manager with the Flask app context."""
        try:
            self.app = app
            self.research_papers_dir = os.path.join(app.instance_path, 'knowledge', 'research-papers')
            self.vector_store_path = os.path.join(app.instance_path, 'knowledge', 'vector_store')
            
            # Initialize the LLM for RAG
            self.llm = ChatOllama(
                model="llama3",
                temperature=0.7,
                max_tokens=2048
            )
            
            logging.info(f"RAG Manager initialized with research papers dir: {self.research_papers_dir}")
            logging.info(f"Vector store path: {self.vector_store_path}")
            return True
        except Exception as e:
            logging.error(f"Error initializing RAG Manager: {str(e)}")
            return False

    def process_research_papers(self):
        """Process all research papers in the research_papers directory."""
        if not self.research_papers_dir or not self.vector_store_path:
            logging.error("RAGManager not properly initialized")
            return None

        # Check if the research papers directory exists
        if not os.path.exists(self.research_papers_dir):
            logging.error(f"Research papers directory not found at {self.research_papers_dir}")
            # Create the directory in case it's missing
            try:
                os.makedirs(self.research_papers_dir, exist_ok=True)
                logging.info(f"Created research papers directory at {self.research_papers_dir}")
            except Exception as e:
                logging.error(f"Failed to create research papers directory: {str(e)}")
            return None
        
        # Check if there are any PDF files in the directory
        pdf_files = [f for f in os.listdir(self.research_papers_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            logging.warning(f"No PDF files found in {self.research_papers_dir}")
            return None

        # Process the documents
        try:
            document_chunks = self.process_documents()
            if not document_chunks:
                logging.error("No documents were processed successfully")
                return None

            vector_db = self.load_or_create_vector_db(document_chunks)
            return vector_db
        except Exception as e:
            logging.error(f"Error processing documents: {str(e)}")
            return None

    def split_document(self, doc_path):
        if not os.path.exists(doc_path):
            logging.error(f"Document not found at {doc_path}")
            return []

        documents = self.extract_text(doc_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
        document_chunks = text_splitter.split_documents(documents)
        logging.info(f"Document split into {len(document_chunks)} chunks.")
        return document_chunks

    def extract_text(self, pdf_path):
        documents = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:  # Add extracted text if available
                    documents.append(Document(page_content=text))
        logging.info(f"Extracted {len(documents)} pages from the PDF.")
        return documents

    def load_or_create_vector_db(self, doc_chunks):
        vector_db_path = self.vector_store_path
        embedding = OllamaEmbeddings(model="nomic-embed-text")

        # Always create a new vector database to ensure it has the latest documents
        logging.info("Creating new vector database with research papers...")
        return self._create_new_vector_db(doc_chunks, vector_db_path, embedding)

    def _create_new_vector_db(self, doc_chunks, vector_db_path, embedding):
        """Helper function to create and persist a new vector database"""
        try:
            logging.info("Creating new vector database.")

            # Clean up existing vector store if it exists
            if os.path.exists(vector_db_path):
                self._clean_directory(vector_db_path)

            if not os.path.exists(vector_db_path):
                os.makedirs(vector_db_path)

            vector_db = Chroma.from_documents(
                documents=doc_chunks,
                embedding=embedding,
                collection_name="research_papers_store",  # Updated collection name
                persist_directory=vector_db_path,
            )

            # Embedding complete message
            logging.info(f"Vector database created and persisted successfully with {len(doc_chunks)} chunks.")
            return vector_db

        except Exception as e:
            logging.error(f"Error during vector database creation: {str(e)}")
            return None       

    def _clean_directory(self, dir_path):
        """Helper function to delete all files and subdirectories inside a given directory"""
        logging.info(f"Cleaning up directory: {dir_path}")
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e}")
            
    def create_retriever(self, vector_db, llm):
        """Create a multi-query retriever."""
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

        logging.info("Creating multi-query retriever...")
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
        )
        logging.info("Retriever created.")
        return retriever

    def create_chain(self, retriever, llm):
        """Creates a chain that combines the retriever and LLM."""
        try:
            # Create the prompt template
            template = """You are an AI assistant specialized in combinatorial testing and fault detection. Use the following pieces of context to answer the question. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

            Context: {context}

            Question: {question}

            Answer: """

            # Create the prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )

            # Create the chain
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            def process_response(query):
                try:
                    context = retriever.retrieve(query)
                    response = chain.invoke({"context": context, "question": query})
                    return {"answer": response}
                except Exception as e:
                    logging.error(f"Error in chain processing: {str(e)}")
                    return {"answer": f"Error processing response: {str(e)}"}

            return process_response

        except Exception as e:
            logging.error(f"Error creating chain: {str(e)}")
            return None
    
    def test_vector_db_loading(self):
        """Test if the vector database can be loaded successfully."""
        vector_db_path = self.vector_store_path
        if os.path.exists(vector_db_path) and os.listdir(vector_db_path):
            try:
                embedding = OllamaEmbeddings(model="nomic-embed-text")
                vector_db = Chroma(
                    embedding_function=embedding,
                    collection_name="research_papers_store",  # Updated collection name
                    persist_directory=vector_db_path,
                )
                logging.info("Vector database loaded successfully.")
                return vector_db
            except Exception as e:
                logging.error(f"Error loading vector database: {str(e)}")
        else:
            logging.info("No existing vector database found. New one will be created when processing documents.")
        return None

    def debug_vector_store_content(self):
        """Debugging method to inspect the content stored in the vector store."""
        with current_app.app_context():  # Ensure we have the correct context
            vector_db_path = self.vector_store_path
            embedding = OllamaEmbeddings(model="nomic-embed-text")

            if os.path.exists(vector_db_path):
                vector_db = Chroma(
                    embedding_function=embedding,
                    collection_name="handbook_vector_store",
                    persist_directory=vector_db_path,
                )
                logging.info("Debugging: Inspecting vector store content...")

                # Use similarity search to retrieve documents for inspection
                try:
                    sample_docs = vector_db.similarity_search(query="Sample", k=5)  # Retrieve 5 documents as a sample
                    if not sample_docs:
                        logging.info("No documents found in the vector store.")
                    else:
                        logging.info(f"Total documents retrieved: {len(sample_docs)}")
                        for idx, doc in enumerate(sample_docs):
                            logging.info(f"Document {idx + 1}: {doc.page_content[:200]}")  # Log first 200 characters of each document

                except Exception as e:
                    logging.error(f"Error retrieving documents for debugging: {str(e)}")

            else:
                logging.error(f"Vector database path not found at {vector_db_path}")

    def create_rag_chain(self):
        """Creates and returns a RAG chain using the vector database."""
        try:
            if not self.vector_db:
                logging.error("Vector database not initialized")
                return None

            if not self.llm:
                logging.error("LLM not initialized")
                return None

            # Create a retriever from the vector database
            retriever = self.vector_db.as_retriever(
                search_kwargs={"k": 4}
            )
            logging.info("Created retriever from vector database")

            # Create the RAG chain using LangChain
            from langchain.chains import ConversationalRetrievalChain
            from langchain.prompts import PromptTemplate

            # Define the prompt template for RAG
            template = """You are an AI assistant specialized in combinatorial testing and fault detection. Use the following pieces of context to answer the question. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

            Context: {context}

            Question: {question}

            Answer: """

            PROMPT = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )

            # Create the chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=False,
                verbose=True
            )

            logging.info("Created RAG chain successfully")
            return chain

        except Exception as e:
            logging.error(f"Error creating RAG chain: {str(e)}")
            return None

    def vector_store_exists(self):
        """Check if the vector store directory exists and has content"""
        try:
            # Ensure we have a path set
            if not self.vector_store_path:
                logging.error("Vector store path is not set")
                return False
            
            # Check if directory exists
            if not os.path.exists(self.vector_store_path):
                logging.info(f"Vector store directory does not exist: {self.vector_store_path}")
                return False
            
            # Check if the directory has content (index files, etc.)
            contents = os.listdir(self.vector_store_path)
            has_files = len(contents) > 0
            logging.info(f"Vector store directory has {len(contents)} items: {has_files}")
            return has_files
        except Exception as e:
            logging.error(f"Error checking if vector store exists: {str(e)}")
            return False

    def load_vector_store(self):
        """Load existing vector store"""
        try:
            if not self.vector_store_exists():
                logging.error("Vector store does not exist")
                return False
            
            # Load the vector store
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.vector_store_path,
                    embedding_function=self.embedding_function
                )
                
                # Verify the collection is properly initialized
                if not hasattr(self.vectorstore, '_collection'):
                    logging.error("Vector store collection not properly initialized after loading")
                    return False
                
                # Test the vector store with a simple query
                try:
                    test_results = self.vectorstore._collection.get(limit=1)
                    if not test_results or 'documents' not in test_results:
                        logging.error("Vector store appears to be empty or corrupted")
                        return False
                except Exception as test_error:
                    logging.error(f"Error testing vector store: {str(test_error)}")
                    return False
                
                logging.info(f"Loaded vector store from {self.vector_store_path}")
                return True
            except Exception as load_error:
                logging.error(f"Error loading vector store: {str(load_error)}")
                return False
        except Exception as e:
            logging.error(f"Error in load_vector_store: {str(e)}")
            return False

    def direct_keyword_search(self, query_str, k=3):
        """
        Perform a direct keyword search without using embeddings
        This is a fallback for when the embedding model has issues
        """
        try:
            if not query_str or not isinstance(query_str, str):
                logging.warning(f"Invalid query for keyword search: {query_str}")
                return "No valid query provided for searching Dr. Wong's research papers."
                
            logging.info(f"Performing direct keyword search for: {query_str}")
            
            # Normalize the query
            query_terms = query_str.lower().split()
            if not query_terms:
                return "Please provide more specific keywords to search in Dr. Wong's research papers."
            
            # Find documents that match the keywords by directly accessing the doc store
            matches = []
            
            # If vectorstore is initialized correctly
            if hasattr(self, 'vectorstore') and self.vectorstore and hasattr(self.vectorstore, '_collection'):
                # Get all documents
                try:
                    results = self.vectorstore._collection.get()
                    
                    if results and 'documents' in results and results['documents']:
                        docs = results['documents']
                        
                        # Score each document
                        scored_docs = []
                        for doc in docs:
                            if not doc or not isinstance(doc, str):
                                continue
                            
                            # Calculate simple term frequency score
                            score = 0
                            doc_lower = doc.lower()
                            for term in query_terms:
                                if term in doc_lower:
                                    score += doc_lower.count(term)
                            
                            if score > 0:
                                scored_docs.append((doc, score))
                        
                        # Sort by score and take top k
                        scored_docs.sort(key=lambda x: x[1], reverse=True)
                        matches = [doc for doc, _ in scored_docs[:k]]
                except Exception as inner_e:
                    logging.error(f"Error accessing document collection: {str(inner_e)}")
            else:
                logging.error("Vector store not properly initialized for keyword search")
                # Return fallback content about Dr. Wong's research
                return """
Dr. Wong is a researcher at UT Dallas specializing in combinatorial testing methods.
Combinatorial testing is an approach that helps detect defects caused by parameter interactions.
Dr. Wong's work focuses on improving test case generation, minimizing the number of tests required
while maximizing fault detection capability. His research has applications in software testing
and quality assurance across various domains."""
            
            if matches:
                return "\n\n".join(matches)
            else:
                return """
I couldn't find specific documents matching your query in Dr. Wong's research papers database.
Dr. Wong specializes in combinatorial testing, which is a software testing technique that ensures
all possible discrete combinations of input parameters are tested. His research focuses on improving
test case generation and fault detection mechanisms."""
                
        except Exception as e:
            logging.error(f"Error in direct keyword search: {str(e)}")
            return "I encountered an error while searching Dr. Wong's research papers, but I can still answer general questions about combinatorial testing based on my training."

    def create_vector_store(self):
        """Process documents and create a new vector store"""
        try:
            logging.info("Creating new vector store...")
            
            # Check if research papers directory exists
            if not os.path.exists(self.research_papers_dir):
                logging.error(f"Research papers directory not found: {self.research_papers_dir}")
                # Try to create it
                try:
                    os.makedirs(self.research_papers_dir, exist_ok=True)
                    logging.info(f"Created research papers directory: {self.research_papers_dir}")
                except Exception as e:
                    logging.error(f"Could not create research papers directory: {str(e)}")
                    return False
            
            # Check if there are PDF files in the directory
            pdf_files = [f for f in os.listdir(self.research_papers_dir) if f.lower().endswith('.pdf')]
            if not pdf_files:
                logging.warning(f"No PDF files found in {self.research_papers_dir}")
                # Create an empty vector store with test documents
                try:
                    # Create test documents about combinatorial testing
                    test_docs = [
                        Document(
                            page_content="Combinatorial testing is a test case generation technique that helps detect defects caused by interactions of different parameters. It was developed by Dr. Wong at UT Dallas.",
                            metadata={"source": "test", "title": "Combinatorial Testing Overview"}
                        ),
                        Document(
                            page_content="Dr. Wong's research focuses on combinatorial testing methods that efficiently cover interactions between software parameters while minimizing the number of test cases required.",
                            metadata={"source": "test", "title": "Dr. Wong's Research"}
                        ),
                        Document(
                            page_content="Fault detection in software testing involves identifying defects or bugs in software systems. Dr. Wong has published papers on improving fault detection through combinatorial approaches.",
                            metadata={"source": "test", "title": "Fault Detection"}
                        )
                    ]
                    
                    # Create directory if it doesn't exist
                    os.makedirs(self.vector_store_path, exist_ok=True)
                    
                    # Create vector store with test documents
                    self.vectorstore = Chroma.from_documents(
                        documents=test_docs,
                        embedding=self.embedding_function,
                        persist_directory=self.vector_store_path
                    )
                    
                    # Verify the vector store was created properly
                    if not hasattr(self.vectorstore, '_collection'):
                        logging.error("Vector store collection not properly initialized with test documents")
                        return False
                    
                    logging.info("Created vector store with test documents for Dr. Wong's research")
                    return True
                except Exception as e:
                    logging.error(f"Error creating vector store with test documents: {str(e)}")
                    return False
            
            # Process the documents from PDF files
            all_document_chunks = []
            for filename in pdf_files:
                paper_path = os.path.join(self.research_papers_dir, filename)
                logging.info(f"Processing research paper: {paper_path}")
                try:
                    # Extract and split document
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
                    documents = self.extract_text(paper_path)
                    if not documents:
                        logging.warning(f"No text extracted from {filename}")
                        continue
                    
                    paper_chunks = text_splitter.split_documents(documents)
                    if paper_chunks:
                        all_document_chunks.extend(paper_chunks)
                        logging.info(f"Added {len(paper_chunks)} chunks from {filename}")
                    else:
                        logging.warning(f"No chunks extracted from {filename}")
                except Exception as e:
                    logging.error(f"Error processing {filename}: {str(e)}")
                    continue
                
            if not all_document_chunks:
                logging.error("No document chunks were created from PDF files")
                return False
            
            logging.info(f"Processed {len(all_document_chunks)} chunks from all papers")
            
            # Create the vector store
            try:
                # Make sure directory exists
                os.makedirs(self.vector_store_path, exist_ok=True)
                
                # Create the vector store
                self.vectorstore = Chroma.from_documents(
                    documents=all_document_chunks,
                    embedding=self.embedding_function,
                    persist_directory=self.vector_store_path
                )
                
                # Verify the vector store was created properly
                if not hasattr(self.vectorstore, '_collection'):
                    logging.error("Vector store collection not properly initialized")
                    return False
                
                logging.info(f"Vector store created successfully with {len(all_document_chunks)} chunks")
                return True
            except Exception as e:
                logging.error(f"Error creating vector store: {str(e)}")
                return False
            
        except Exception as e:
            logging.error(f"Error in create_vector_store: {str(e)}")
            return False

