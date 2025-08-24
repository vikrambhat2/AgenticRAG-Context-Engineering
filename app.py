import os
import tempfile
import shutil
from typing import List, Dict, Any, TypedDict
from pathlib import Path
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader,
    CSVLoader
)
from langgraph.graph import StateGraph, END
import pickle
import torch

# State definition for the RAG pipeline
class RAGState(TypedDict):
    """State for the RAG pipeline"""
    query: str
    documents: List[Document]
    compressed_documents: List[Document]
    context: str
    answer: str
    metadata: Dict[str, Any]
    error: str

class RAGPipeline:
    """Advanced RAG Pipeline using LangGraph with Contextual Compression"""
    
    def __init__(self, 
                 model_name: str = "llama3.2:latest",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        # Initialize LLMs using ChatOllama
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            max_tokens=2048
        )
        
        # Separate LLM for compression
        self.compression_llm = ChatOllama(
            model=model_name,
            temperature=0,
            max_tokens=1024
        )
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize HuggingFace embeddings with proper device handling
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={
                'device': device,
                'trust_remote_code': True,
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'device': device
            },
            show_progress=False
        )
        
        # Initialize components
        self.vectorstore = None
        self.base_retriever = None
        self.compression_retriever = None
        
        # Text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Prompts
        self.setup_prompts()
        
        # Build the LangGraph workflow
        self.workflow = self.build_workflow()
    
    def setup_prompts(self):
        """Setup prompt templates"""
        
        # Compression prompt for LLMChainExtractor
        self.compression_prompt = PromptTemplate(
            template="""Given the following question and document, extract only the parts of the document that are directly relevant to answering the question. 
            Be concise but preserve important context and details.
            
            Question: {question}
            
            Document: {context}
            
            Relevant extracted content:""",
            input_variables=["question", "context"]
        )
        
        # Final answer generation prompt
        self.answer_prompt = PromptTemplate(
            template="""You are a helpful AI assistant. Use the following context to answer the user's question accurately and comprehensively.
            If the context doesn't contain enough information to answer the question, clearly state what information is missing.
            Always cite your sources when making specific claims.
            
            Context:
            {context}
            
            Question: {query}
            
            Provide a detailed, well-structured answer:""",
            input_variables=["context", "query"]
        )
        
        # Query analysis prompt for routing
        self.query_analysis_prompt = PromptTemplate(
            template="""Analyze the following query and classify it:
            
            Query: {query}
            
            Classification options:
            - SIMPLE: Basic factual questions that need direct answers
            - COMPLEX: Multi-part questions requiring analysis or synthesis
            - COMPARISON: Questions asking to compare or contrast concepts
            
            Return only the classification (SIMPLE, COMPLEX, or COMPARISON):""",
            input_variables=["query"]
        )
    
    def load_documents_from_files(self, uploaded_files) -> List[Document]:
        """Load documents from uploaded files"""
        documents = []
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                # Save uploaded file to temp directory
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load document based on file type
                file_extension = Path(uploaded_file.name).suffix.lower()
                
                try:
                    if file_extension == '.pdf':
                        loader = PyPDFLoader(temp_path)
                    elif file_extension == '.docx':
                        loader = Docx2txtLoader(temp_path)
                    elif file_extension == '.txt':
                        loader = TextLoader(temp_path, encoding='utf-8')
                    elif file_extension == '.csv':
                        loader = CSVLoader(temp_path)
                    else:
                        st.warning(f"Unsupported file type: {file_extension}")
                        continue
                    
                    # Load and add source metadata
                    file_docs = loader.load()
                    for doc in file_docs:
                        doc.metadata['source'] = uploaded_file.name
                        doc.metadata['file_type'] = file_extension
                    
                    documents.extend(file_docs)
                    
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        
        return documents
    
    def setup_vectorstore(self, documents: List[Document]):
        """Setup vectorstore with documents"""
        if not documents:
            st.error("No documents provided!")
            return False
        
        try:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            st.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            
            # Create vectorstore
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
            # Setup base retriever
            self.base_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}  # Retrieve more initially for better compression
            )
            
            # Setup compression retriever
            compressor = LLMChainExtractor.from_llm(
                llm=self.compression_llm,
                prompt=self.compression_prompt
            )
            
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.base_retriever
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error setting up vectorstore: {str(e)}")
            return False
    
    def save_vectorstore(self, filepath: str):
        """Save vectorstore to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(filepath)
    
    def load_vectorstore(self, filepath: str):
        """Load vectorstore from disk"""
        try:
            self.vectorstore = FAISS.load_local(filepath, self.embeddings, allow_dangerous_deserialization=True)
            
            # Reinitialize retrievers
            self.base_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            )
            
            compressor = LLMChainExtractor.from_llm(
                llm=self.compression_llm,
                prompt=self.compression_prompt
            )
            
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.base_retriever
            )
            
            return True
        except Exception as e:
            st.error(f"Error loading vectorstore: {str(e)}")
            return False
    
    def analyze_query(self, state: RAGState) -> RAGState:
        """Analyze query complexity and type"""
        try:
            query = state["query"]
            
            # Analyze query type
            analysis_result = self.llm.invoke(
                self.query_analysis_prompt.format(query=query)
            )
            
            query_type = analysis_result.content.strip()
            
            # Update metadata
            metadata = state.get("metadata", {})
            metadata.update({
                "query_type": query_type,
                "analysis_timestamp": "now"
            })
            
            return {
                **state,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                **state,
                "error": f"Query analysis failed: {str(e)}"
            }
    
    def retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents"""
        try:
            query = state["query"]
            
            if not self.compression_retriever:
                raise ValueError("Vectorstore not initialized. Please upload and process documents first.")
            
            # Retrieve and compress documents
            compressed_docs = self.compression_retriever.get_relevant_documents(query)
            
            # Also get original documents for comparison/fallback
            original_docs = self.base_retriever.get_relevant_documents(query)
            
            return {
                **state,
                "documents": original_docs,
                "compressed_documents": compressed_docs
            }
            
        except Exception as e:
            return {
                **state,
                "error": f"Document retrieval failed: {str(e)}"
            }
    
    def prepare_context(self, state: RAGState) -> RAGState:
        """Prepare context from compressed documents"""
        try:
            compressed_docs = state["compressed_documents"]
            
            # Format context from compressed documents
            context_parts = []
            for i, doc in enumerate(compressed_docs):
                source = doc.metadata.get("source", f"Document {i+1}")
                content = doc.page_content.strip()
                
                context_part = f"[Source: {source}]\n{content}"
                context_parts.append(context_part)
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Update metadata with context stats
            metadata = state.get("metadata", {})
            original_length = sum(len(doc.page_content) for doc in state.get("documents", []))
            metadata.update({
                "num_documents": len(compressed_docs),
                "context_length": len(context),
                "compression_ratio": len(context) / original_length if original_length > 0 else 0
            })
            
            return {
                **state,
                "context": context,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                **state,
                "error": f"Context preparation failed: {str(e)}"
            }
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate final answer using the prepared context"""
        try:
            query = state["query"]
            context = state["context"]
            
            # Generate answer
            response = self.llm.invoke(
                self.answer_prompt.format(context=context, query=query)
            )
            
            answer = response.content.strip()
            
            # Update metadata
            metadata = state.get("metadata", {})
            metadata.update({
                "answer_length": len(answer),
                "generation_complete": True
            })
            
            return {
                **state,
                "answer": answer,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                **state,
                "error": f"Answer generation failed: {str(e)}"
            }
    
    def handle_error(self, state: RAGState) -> RAGState:
        """Handle errors in the pipeline"""
        error_msg = state.get("error", "Unknown error occurred")
        
        return {
            **state,
            "answer": f"I encountered an error while processing your request: {error_msg}. Please try again or rephrase your question."
        }
    
    def should_continue(self, state: RAGState) -> str:
        """Determine if we should continue or handle error"""
        if state.get("error"):
            return "error"
        return "continue"
    
    def build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("prepare_context", self.prepare_context)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("handle_error", self.handle_error)
        
        # Define the flow
        workflow.set_entry_point("analyze_query")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_query",
            self.should_continue,
            {
                "continue": "retrieve_documents",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "retrieve_documents",
            self.should_continue,
            {
                "continue": "prepare_context",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "prepare_context",
            self.should_continue,
            {
                "continue": "generate_answer",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_answer",
            self.should_continue,
            {
                "continue": END,
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a single query through the RAG pipeline"""
        
        initial_state = RAGState(
            query=query,
            documents=[],
            compressed_documents=[],
            context="",
            answer="",
            metadata={},
            error=""
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return final_state


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Advanced RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Advanced RAG Chatbot with Document Upload")
    st.markdown("Upload your documents and ask questions using advanced contextual compression!")
    
    # Sidebar for configuration and file upload
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            ["llama3.2:latest", "llama3.3", "mistral"],
            help="Choose the LLM model for generation"
        )
        
        # Embedding model selection with safer defaults
        embedding_model = st.selectbox(
            "Select Embedding Model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-MiniLM-L6-v2",
                "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
            ],
            help="Choose the embedding model (smaller models are more stable)"
        )
        
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv'],
            help="Upload PDF, DOCX, TXT, or CSV files"
        )
        
        # Process documents button
        process_docs = st.button("🔄 Process Documents", type="primary")
        
        # Vectorstore management
        st.header("💾 Vectorstore Management")
        
        vectorstore_name = st.text_input("Vectorstore Name", "my_vectorstore")
        
        col1, col2 = st.columns(2)
        with col1:
            save_vs = st.button("💾 Save")
        with col2:
            load_vs = st.button("📂 Load")
    
    # Initialize session state
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'vectorstore_ready' not in st.session_state:
        st.session_state.vectorstore_ready = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize RAG pipeline
    if st.session_state.rag_pipeline is None:
        try:
            with st.spinner("Initializing RAG pipeline..."):
                # Set environment variable to avoid meta tensor issues
                os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                
                st.session_state.rag_pipeline = RAGPipeline(
                    model_name=model_name,
                    embedding_model=embedding_model
                )
            st.success("RAG pipeline initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {str(e)}")
            # Show more detailed error information
            with st.expander("Error Details"):
                st.code(str(e))
    
    # Process documents
    if process_docs and uploaded_files and st.session_state.rag_pipeline:
        with st.spinner("Processing documents..."):
            # Load documents
            documents = st.session_state.rag_pipeline.load_documents_from_files(uploaded_files)
            
            if documents:
                # Setup vectorstore
                success = st.session_state.rag_pipeline.setup_vectorstore(documents)
                if success:
                    st.session_state.vectorstore_ready = True
                    st.success(f"Successfully processed {len(documents)} documents!")
                else:
                    st.error("Failed to setup vectorstore")
            else:
                st.error("No documents were loaded successfully")
    
    # Vectorstore management
    if save_vs and st.session_state.rag_pipeline and st.session_state.vectorstore_ready:
        try:
            st.session_state.rag_pipeline.save_vectorstore(vectorstore_name)
            st.success(f"Vectorstore saved as '{vectorstore_name}'")
        except Exception as e:
            st.error(f"Error saving vectorstore: {str(e)}")
    
    if load_vs and st.session_state.rag_pipeline and vectorstore_name:
        try:
            success = st.session_state.rag_pipeline.load_vectorstore(vectorstore_name)
            if success:
                st.session_state.vectorstore_ready = True
                st.success(f"Vectorstore '{vectorstore_name}' loaded successfully!")
            else:
                st.error(f"Failed to load vectorstore '{vectorstore_name}'")
        except Exception as e:
            st.error(f"Error loading vectorstore: {str(e)}")
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    for i, (question, answer, metadata) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**You:** {question}")
            st.markdown(f"**Assistant:** {answer}")
            
            # Show metadata in expander
            with st.expander(f"📊 Query Metadata #{i+1}"):
                st.json(metadata)
            
            st.divider()
    
    # Query input
    if st.session_state.vectorstore_ready and st.session_state.rag_pipeline:
        query = st.text_input(
            "Ask a question about your documents:",
            placeholder="What is the main topic discussed in the documents?"
        )
        
        if st.button("🚀 Ask Question", type="primary") and query:
            with st.spinner("Processing your question..."):
                try:
                    result = st.session_state.rag_pipeline.process_query(query)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((
                        query,
                        result['answer'],
                        result['metadata']
                    ))
                    
                    # Rerun to show the new message
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
    
    else:
        if not st.session_state.vectorstore_ready:
            st.warning("Please upload and process documents first, or load an existing vectorstore.")
    
    # Instructions
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **Select an LLM model** in the sidebar
        2. **Upload documents** (PDF, DOCX, TXT, CSV) using the file uploader
        3. **Click 'Process Documents'** to create the knowledge base
        4. **Ask questions** about your documents in the chat interface
        
        **Features:**
        - Advanced contextual compression for better relevance
        - Support for multiple document formats
        - Save and load vectorstores for reuse
        - Query analysis and metadata tracking
        - Chat history with expandable metadata
        """)

if __name__ == "__main__":
    main()