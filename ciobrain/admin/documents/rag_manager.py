import os
import shutil
import logging
import pdfplumber
from flask import current_app
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)

class RAGManager:

    def __init__(self):
        # Get the absolute path to the instance/knowledge/research-papers directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        instance_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), 'instance')
        self.research_papers_dir = os.path.join(instance_dir, 'knowledge', 'research-papers')
        self.vector_store_path = None
        os.makedirs(self.research_papers_dir, exist_ok=True)
        logging.info(f"Research papers directory set to: {self.research_papers_dir}")
        logging.info(f"Current directory contents: {os.listdir(self.research_papers_dir)}")

    def initialize(self, app):
        """Initialize paths using the Flask app context"""
        self.vector_store_path = app.config['VECTOR_STORE']
        logging.info(f"Vector store path set to: {self.vector_store_path}")

    def process_documents(self):
        """Process all research papers in the research_papers directory."""
        if not self.research_papers_dir or not self.vector_store_path:
            logging.error("RAGManager not properly initialized")
            return None

        # Process research papers
        document_chunks = self.process_research_papers()
        if not document_chunks:
            logging.error("No documents were processed successfully")
            return None

        vector_db = self.load_or_create_vector_db(document_chunks)
        return vector_db

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
        """Create the chain with preserved syntax and detailed progress updates."""
        # Improved RAG prompt template
        template = """You are an AI assistant helping with software testing research. 
        Use the following context to answer the question. If you don't know the answer, say so.
        Be specific and cite information from the research papers when possible.

        Context:
        {context}

        Question: {question}

        Answer: Let me help you with that based on the research papers."""

        prompt = ChatPromptTemplate.from_template(template)

        def chain_generator(question):
            try:
                # Use the new `.invoke()` method to get relevant documents
                retrieved_docs = retriever.invoke(question)

                # Handle if no documents were found
                if not retrieved_docs or len(retrieved_docs) == 0:
                    yield "I couldn't find any relevant information in the research papers for this question."
                    return

                # Convert the retrieved documents into a single string context
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                logging.info(f"Retrieved {len(retrieved_docs)} documents.")

                # Create a dictionary with context and question to pass through the chain
                input_dict = {"context": context, "question": question}

                # Create the chain by connecting all the Runnables
                chain = (
                    RunnablePassthrough()  # Accept the input dictionary
                    | prompt                # Use the prompt to format the response
                    | llm                   # Pass to the LLM to generate the output
                    | StrOutputParser()     # Parse the output into a readable string
                )

                # Invoke the chain with the correct dictionary format
                response = chain.invoke(input_dict)
                
                # Split the response into sentences for better streaming
                sentences = response.split('. ')
                for sentence in sentences:
                    if sentence.strip():
                        yield sentence.strip() + '. '
                
            except Exception as e:
                logging.error(f"Error in chain generation: {str(e)}")
                yield f"Error: {str(e)}"

        return chain_generator
    
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

