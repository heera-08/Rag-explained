
# =============================================================================
# 1. BASIC RAG ARCHITECTURE OVERVIEW
# =============================================================================

"""
RAG Flow:
1. Document Ingestion → Chunking → Embedding → Vector Store
2. Query → Embedding → Similarity Search → Context Retrieval
3. Context + Query → LLM → Enhanced Response
"""

# =============================================================================
# 2. DOCUMENT INGESTION AND CHUNKING
# =============================================================================

import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFLoader, TextLoader

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self, file_path: str) -> List[str]:
        """Load and chunk documents from various formats"""
        if file_path.endswith('.pdf'):
            loader = PDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return [chunk.page_content for chunk in chunks]

# Example usage
processor = DocumentProcessor()
document_chunks = processor.load_documents("company_docs.pdf")
print(f"Created {len(document_chunks)} chunks from document")

# =============================================================================
# 3. EMBEDDING GENERATION AND VECTOR STORAGE
# =============================================================================

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
    
    def add_documents(self, documents: List[str]):
        """Convert documents to embeddings and store in FAISS index"""
        print(f"Generating embeddings for {len(documents)} documents...")
        
        # Generate embeddings
        embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        self.documents = documents
        self.embeddings = embeddings
        
        print(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for most similar documents"""
        if self.index is None:
            raise ValueError("No documents in vector store")
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "document": self.documents[idx],
                "score": float(score),
                "index": int(idx)
            })
        
        return results

# Example usage
vector_store = VectorStore()
vector_store.add_documents(document_chunks)

# =============================================================================
# 4. QUERY PROCESSING AND RETRIEVAL
# =============================================================================

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, top_k: int = 3):
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve_context(self, query: str) -> str:
        """Retrieve relevant context for a query"""
        # Search for similar documents
        results = self.vector_store.search(query, k=self.top_k)
        
        # Filter by relevance score (threshold can be adjusted)
        relevant_docs = [r for r in results if r["score"] > 0.5]
        
        if not relevant_docs:
            return "No relevant context found."
        
        # Combine context with metadata
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            context_parts.append(f"Context {i+1} (Score: {doc['score']:.3f}):\n{doc['document']}")
        
        return "\n\n".join(context_parts)

# Example usage
retriever = RAGRetriever(vector_store)
context = retriever.retrieve_context("What is our company's security policy?")
print(f"Retrieved context:\n{context[:200]}...")

# =============================================================================
# 5. LLM INTEGRATION AND RESPONSE GENERATION
# =============================================================================

import openai
from typing import Optional

class RAGGenerator:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
    
    def generate_response(self, query: str, context: str, 
                         system_prompt: Optional[str] = None) -> str:
        """Generate response using retrieved context"""
        
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            Use the context to answer the user's question accurately. If the context doesn't contain 
            enough information, say so clearly. Always cite which part of the context you're using."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Please answer the question based on the provided context."""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=512,
                temperature=0.1  # Low temperature for factual responses
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Example usage (requires OpenAI API key)
# generator = RAGGenerator("your-api-key-here")
# response = generator.generate_response(query, context)

# =============================================================================
# 6. COMPLETE RAG PIPELINE
# =============================================================================

class RAGPipeline:
    def __init__(self, openai_api_key: str):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.retriever = RAGRetriever(self.vector_store)
        self.generator = RAGGenerator(openai_api_key)
        self.is_initialized = False
    
    def initialize(self, document_paths: List[str]):
        """Initialize the RAG system with documents"""
        print("Initializing RAG pipeline...")
        
        # Load and process all documents
        all_chunks = []
        for path in document_paths:
            chunks = self.document_processor.load_documents(path)
            all_chunks.extend(chunks)
        
        # Build vector store
        self.vector_store.add_documents(all_chunks)
        self.is_initialized = True
        
        print(f"RAG pipeline initialized with {len(all_chunks)} document chunks")
    
    def query(self, question: str) -> Dict:
        """Process a query through the complete RAG pipeline"""
        if not self.is_initialized:
            raise ValueError("RAG pipeline not initialized. Call initialize() first.")
        
        # Retrieve relevant context
        context = self.retriever.retrieve_context(question)
        
        # Generate response
        response = self.generator.generate_response(question, context)
        
        return {
            "question": question,
            "context": context,
            "response": response,
            "sources": len(context.split("Context")) - 1 if context != "No relevant context found." else 0
        }

# Example usage
# rag_system = RAGPipeline("your-openai-api-key OR google-gemini-api") whichever you have
# rag_system.initialize(["doc1.pdf", "doc2.txt", "doc3.pdf"])
# result = rag_system.query("What is our data retention policy?")
# print(f"Response: {result['response']}")

# =============================================================================
# 7. EVALUATION AND MONITORING
# =============================================================================

class RAGEvaluator:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
    
    def evaluate_retrieval(self, query: str, expected_docs: List[str]) -> Dict:
        """Evaluate retrieval quality"""
        results = self.rag_pipeline.vector_store.search(query, k=10)
        retrieved_docs = [r["document"] for r in results]
        
        # Calculate metrics
        relevant_retrieved = len(set(retrieved_docs) & set(expected_docs))
        precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
        recall = relevant_retrieved / len(expected_docs) if expected_docs else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "retrieved_count": len(retrieved_docs)
        }
    
    def benchmark_response_time(self, queries: List[str]) -> Dict:
        """Benchmark system performance"""
        import time
        
        times = []
        for query in queries:
            start_time = time.time()
            self.rag_pipeline.query(query)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "avg_response_time": sum(times) / len(times),
            "min_response_time": min(times),
            "max_response_time": max(times),
            "total_queries": len(queries)
        }
