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

logging.basicConfig(level=logging.INFO)

class RAGManager:

    def __init__(self):
        self.app = None
        self.research_papers_dir = None
        self.vector_store_path = None
        self.vector_db = None
        self.llm = None

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

    def process_documents(self):
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
            document_chunks = self.process_research_papers()
            if not document_chunks:
                logging.error("No documents were processed successfully")
                return None

            vector_db = self.load_or_create_vector_db(document_chunks)
            return vector_db
        except Exception as e:
            logging.error(f"Error processing documents: {str(e)}")
            return None

    def process_research_papers(self):
        """Process all research papers in the research_papers directory."""
        if not os.path.exists(self.research_papers_dir):
            logging.error(f"Research papers directory not found at {self.research_papers_dir}")
            return None

        all_document_chunks = []
        
        # Process each research paper
        for filename in os.listdir(self.research_papers_dir):
            if filename.endswith('.pdf'):
                paper_path = os.path.join(self.research_papers_dir, filename)
                logging.info(f"Processing research paper: {paper_path}")
                try:
                    paper_chunks = self.split_document(paper_path)
                    if paper_chunks:
                        all_document_chunks.extend(paper_chunks)
                        logging.info(f"Added {len(paper_chunks)} chunks from {filename}")
                    else:
                        logging.warning(f"No chunks extracted from {filename}")
                except Exception as e:
                    logging.error(f"Error processing {filename}: {str(e)}")
                    continue

        if not all_document_chunks:
            logging.warning("No research papers were processed successfully")
            return None

        logging.info(f"Successfully processed {len(all_document_chunks)} total chunks from all papers")
        return all_document_chunks

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
                    response = chain.invoke(query)
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

