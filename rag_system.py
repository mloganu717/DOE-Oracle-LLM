"""
DOEOracle class for handling RAG queries with citations from Dr. Wong's research
"""

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import os
import re
from typing import List, Optional

class DOEOracle:
    def __init__(self, model_name: str = "CombinatorialExpert"):
        self.model_name = model_name
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.2  # Balanced for professional consistency with some variation
        )
        self.vector_store = None
        self.conversation_history = []
        
        # DOE Oracle prompt template - specialized for Dr. Wong's research
        self.prompt_template = """You are the DOE Oracle, a distinguished combinatorial testing expert who provides comprehensive, professional analysis based on Dr. Wong's research papers. Answer based ONLY on the provided Context from Dr. Wong's research papers.

COMMUNICATION STYLE:
- Maintain a formal, professional, and academic tone
- Provide detailed explanations with specific findings from the research
- Use precise terminology and technical language appropriate for experts
- Structure responses with clear examples and empirical evidence
- Present findings objectively and systematically
- Include relevant metrics, percentages, and comparative data when available

RULES:
1. Use ONLY information explicitly stated in the Context
2. If information is not in the Context, state "The provided research does not address this specific aspect"
3. Provide comprehensive responses of 3-5 sentences with detailed analysis
4. Do NOT include any line numbers (like 59| or 60|) in your response
5. Clean up any formatting artifacts from the source text
6. MANDATORY: Copy the exact citation shown after CITATION: and include it at the end of your response

Context with Source Information:
{context}

Question: {question}

Answer:"""
        
    def _clean_document_text(self, text: str) -> str:
        """Clean document text by removing line numbers and formatting artifacts."""
        # Remove line numbers at the start of lines (e.g., "59|", "123|")
        text = re.sub(r'^\s*\d+\|', '', text, flags=re.MULTILINE)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up common PDF artifacts
        text = re.sub(r'–', '-', text)  # Replace em-dash with regular dash
        text = re.sub(r'["""]', '"', text)  # Normalize quotes
        
        return text.strip()
    
    def _format_context_with_metadata(self, docs: List[Document]) -> str:
        """Format context including source metadata for proper citations."""
        formatted_chunks = []
        for doc in docs:
            # Get clean source name
            clean_source = doc.metadata.get('clean_source')
            if clean_source:
                source_name = clean_source
            else:
                source_name = os.path.basename(doc.metadata.get('source', 'Unknown')).replace('.pdf', '')
            
            # Convert 0-based page number to 1-based for human readability
            page_num = doc.metadata.get('page', 'unknown')
            if isinstance(page_num, int):
                page_num = page_num + 1  # Convert 0-based to 1-based
            
            # Format the chunk with source info prominently displayed
            chunk_text = f"{doc.page_content}\n\nCITATION: [Source: {source_name}, Page {page_num}]"
            formatted_chunks.append(chunk_text)
        
        return "\n\n".join(formatted_chunks)
    
    def load_papers(self, paper_paths: List[str]):
        """Load and process research papers from PDF files."""
        documents = []
        for paper_path in paper_paths:
            if not os.path.exists(paper_path):
                raise FileNotFoundError(f"Paper not found: {paper_path}")
            
            loader = PyPDFLoader(paper_path)
            loaded_docs = loader.load()
            
            # Clean the text content of each document
            for doc in loaded_docs:
                doc.page_content = self._clean_document_text(doc.page_content)
                # Ensure we have a clean filename for citations
                doc.metadata['clean_source'] = os.path.basename(paper_path).replace('.pdf', '')
            
            documents.extend(loaded_docs)
        
        # Split documents into chunks with more context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\nAbstract", "\n\n", "\n", ".", "!", "?", ",", " ", ""]  # Added Abstract as first separator
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store with metadata
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="instance/vector_store"
        )
        
        # Vector store is ready for custom querying with metadata
    
    def query(self, question: str) -> str:
        """Query the research expert with a question."""
        if not self.vector_store:
            raise ValueError("No papers loaded. Please load papers first.")
        
        # Retrieve relevant documents (get more context for richer responses)
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Get more context for detailed explanations
        )
        source_docs = retriever.invoke(question)
        
        # Format context with metadata
        context_with_metadata = self._format_context_with_metadata(source_docs)
        
        # Create prompt with formatted context
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        # Get response from LLM
        formatted_prompt = prompt.format(context=context_with_metadata, question=question)
        result = self.llm.invoke(formatted_prompt)
        
        # # Get the most relevant source document (Removed manual citation appending - relying on LLM prompt)
        # if source_docs:
        #     most_relevant_doc = source_docs[0]
        #     source_file = os.path.basename(most_relevant_doc.metadata.get('source', ''))
        #     page_num = most_relevant_doc.metadata.get('page', 1)
        #     
        #     # If the response doesn't already include a source citation, add it
        #     if not result.strip().endswith(']'):
        #         result = f"{result}\\n[Source: {source_file}, Page {page_num}]"
        
        # Format evidence for the frontend (using the first doc as primary evidence still)
        evidence = []
        if source_docs:
            primary_doc = source_docs[0]
            # Use clean source name if available, otherwise use basename
            clean_source = primary_doc.metadata.get('clean_source')
            if clean_source:
                source_file = clean_source
            else:
                source_file = os.path.basename(primary_doc.metadata.get('source', '')).replace('.pdf', '')
            
            # Convert 0-based page number to 1-based for human readability
            page_num = primary_doc.metadata.get('page', 'unknown')
            if isinstance(page_num, int):
                page_num = page_num + 1  # Convert 0-based to 1-based
            
            # Clean the quote content as well
            clean_quote = self._clean_document_text(primary_doc.page_content)
            # Truncate quote if too long for display
            if len(clean_quote) > 200:
                clean_quote = clean_quote[:200] + "..."
            
            evidence.append({
                'quote': clean_quote,
                'source': source_file,
                'page': str(page_num),
                'supports': 'Source Evidence'
            })

        # Update conversation history
        self.conversation_history.append((question, result))
        
        # Return both main content and evidence
        return {
            'text': result,
            'evidence': evidence
        }
    
    def save_vector_store(self):
        """Save the vector store to disk."""
        # Chroma now automatically persists documents
        pass
    
    def load_vector_store(self):
        """Load the vector store from disk."""
        self.vector_store = Chroma(
            persist_directory="instance/vector_store",
            embedding_function=self.embeddings
        )