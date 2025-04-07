# DOE RAG System Documentation

## Overview

DOE features a Retrieval-Augmented Generation (RAG) system that enhances the AI's responses by retrieving and incorporating information from Dr. Wong's research papers. This system combines both keyword search and semantic search techniques to provide comprehensive and accurate information.

## How It Works

The RAG system operates in the following steps:

1. **Query Processing**: When a user asks a question, their query is extracted and preprocessed.

2. **Hybrid Search**: The system performs a two-pronged search approach:
   - **Keyword Search**: Looks for exact keyword matches in the research papers.
   - **Semantic Search**: Uses embeddings to find conceptually related content even when exact keywords are not present.

3. **Result Combination**: Results from both search methods are combined and deduplicated to provide the most comprehensive set of relevant information.

4. **Response Generation**: The retrieved information is used as context for the AI model to generate a response that accurately addresses the user's question.

## Key Components

### 1. RAGManager

The `RAGManager` class handles:
- Vector store initialization and maintenance
- Document processing from PDF files
- Implementation of search algorithms (keyword, semantic, and hybrid)
- Creation of the RAG chain for response generation

### 2. Mediator

The `Mediator` class integrates the RAG functionality into the application by:
- Determining when to use RAG based on user preferences
- Handling error cases gracefully
- Providing fallback responses when RAG is unavailable
- Streaming responses to the user

## Robustness Features

The RAG system is designed to be robust against various failure modes:

1. **Vector Store Initialization**: Automatically creates test documents if no research papers are found.

2. **Error Handling**: Comprehensive error handling ensures the system degrades gracefully instead of failing completely.

3. **Fallback Responses**: If RAG fails, the system falls back to the base model with appropriate messaging to the user.

4. **Hybrid Search**: Combines the strengths of both keyword and semantic search to maximize the chances of finding relevant information.

## Adding Research Papers

To add research papers to the system:

1. Place PDF files of research papers in the `instance/knowledge/research-papers` directory.

2. Restart the application or trigger a reprocessing of documents.

The system will automatically extract text from the PDFs, chunk the content appropriately, and update the vector store.

## Troubleshooting

If you encounter issues with the RAG system:

1. Check the logs for specific error messages
2. Verify that the research papers directory contains valid PDF files
3. Ensure the embedding model is properly installed and accessible
4. If embeddings are failing, the system will still function using keyword search

## Future Improvements

Potential future enhancements include:

1. Supporting more document formats (beyond PDF)
2. Implementing multi-query retrieval for better search results
3. Adding document metadata extraction for more nuanced retrieval
4. Implementing user feedback mechanisms to improve search quality over time 