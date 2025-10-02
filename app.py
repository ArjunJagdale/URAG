import os
import re
import asyncio
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import gradio as gr

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# LangGraph imports
from langgraph.graph import StateGraph, END

# State definition for LangGraph (no serializable objects)
class RAGState(TypedDict):
    urls: List[str]
    raw_content: Dict[str, str]
    documents: List[Dict[str, Any]]  # Store as dict instead of Document objects
    embeddings_ready: bool
    query: str
    retrieved_docs: List[Dict[str, Any]]
    answer_length: str
    final_answer: str
    citations: List[Dict[str, str]]
    error_messages: List[str]

class MultiURLRAGSystem:
    def __init__(self, openrouter_api_key: str):
        self.openrouter_api_key = openrouter_api_key
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize LLM with OpenRouter
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            model="openai/gpt-3.5-turbo",
            temperature=0.1
        )
        
        # Store vector store as instance variable
        self.vector_store = None
        self.current_documents = []  # Store actual Document objects
        self.graph = self._create_graph()
        
    def _create_graph(self):
        """Create the LangGraph workflow without memory checkpointing"""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("fetch_content", self.fetch_content_node)
        workflow.add_node("process_documents", self.process_documents_node)
        workflow.add_node("create_vector_store", self.create_vector_store_node)
        workflow.add_node("retrieve_documents", self.retrieve_documents_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        
        # Define edges
        workflow.set_entry_point("fetch_content")
        workflow.add_edge("fetch_content", "process_documents")
        workflow.add_edge("process_documents", "create_vector_store")
        workflow.add_edge("create_vector_store", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        # Compile graph WITHOUT memory checkpointing
        return workflow.compile()
    
    def clean_html_content(self, html_content: str, base_url: str = "") -> str:
        """Clean and extract text from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
        except Exception as e:
            print(f"Error cleaning HTML: {e}")
            return ""
    
    def fetch_url_content(self, url: str) -> Dict[str, str]:
        """Fetch content from a single URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            content = self.clean_html_content(response.text, url)
            
            return {
                "url": url,
                "content": content,
                "status": "success"
            }
        except Exception as e:
            return {
                "url": url,
                "content": "",
                "status": "error",
                "error": str(e)
            }
    
    def fetch_content_node(self, state: RAGState) -> RAGState:
        """Node to fetch content from all URLs"""
        print("📥 Fetching content from URLs...")
        raw_content = {}
        error_messages = state.get("error_messages", [])
        
        for url in state["urls"]:
            result = self.fetch_url_content(url)
            if result["status"] == "success" and result["content"]:
                raw_content[url] = result["content"]
                print(f"✅ Successfully fetched: {url}")
            else:
                error_msg = f"Failed to fetch {url}: {result.get('error', 'Unknown error')}"
                error_messages.append(error_msg)
                print(f"❌ {error_msg}")
        
        return {
            **state,
            "raw_content": raw_content,
            "error_messages": error_messages
        }
    
    def process_documents_node(self, state: RAGState) -> RAGState:
        """Node to process and chunk documents"""
        print("📄 Processing and chunking documents...")
        documents = []
        actual_documents = []  # Store actual Document objects separately
        
        for url, content in state["raw_content"].items():
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                # Store serializable version in state
                doc_dict = {
                    "page_content": chunk,
                    "metadata": {
                        "source": url,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                }
                documents.append(doc_dict)
                
                # Store actual Document object separately
                actual_doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": url,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                )
                actual_documents.append(actual_doc)
        
        # Store actual documents in instance variable
        self.current_documents = actual_documents
        
        print(f"✅ Created {len(documents)} document chunks from {len(state['raw_content'])} URLs")
        
        return {
            **state,
            "documents": documents
        }
    
    def create_vector_store_node(self, state: RAGState) -> RAGState:
        """Node to create vector store from documents"""
        print("🔍 Creating vector embeddings and store...")
        
        if not state["documents"] or not self.current_documents:
            return {
                **state,
                "embeddings_ready": False,
                "error_messages": state.get("error_messages", []) + ["No documents to embed"]
            }
        
        try:
            # Use actual Document objects for vector store creation
            self.vector_store = FAISS.from_documents(
                self.current_documents, 
                self.embeddings
            )
            print(f"✅ Created vector store with {len(self.current_documents)} documents")
            
            return {
                **state,
                "embeddings_ready": True
            }
        except Exception as e:
            error_msg = f"Error creating vector store: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "embeddings_ready": False,
                "error_messages": state.get("error_messages", []) + [error_msg]
            }
    
    def retrieve_documents_node(self, state: RAGState) -> RAGState:
        """Node to retrieve relevant documents"""
        print(f"🔎 Retrieving documents for query: {state['query']}")
        
        if not state.get("embeddings_ready") or not self.vector_store:
            return {
                **state,
                "retrieved_docs": [],
                "error_messages": state.get("error_messages", []) + ["No vector store available"]
            }
        
        try:
            # Adjust k based on answer length
            k_map = {"short": 3, "medium": 6, "detailed": 10}
            k = k_map.get(state["answer_length"], 6)
            
            retrieved_docs = self.vector_store.similarity_search(
                state["query"], 
                k=k
            )
            
            # Convert Document objects to serializable dicts
            retrieved_docs_dict = []
            for doc in retrieved_docs:
                doc_dict = {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                retrieved_docs_dict.append(doc_dict)
            
            print(f"✅ Retrieved {len(retrieved_docs_dict)} relevant documents")
            
            return {
                **state,
                "retrieved_docs": retrieved_docs_dict
            }
        except Exception as e:
            error_msg = f"Error retrieving documents: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "retrieved_docs": [],
                "error_messages": state.get("error_messages", []) + [error_msg]
            }
    
    def generate_answer_node(self, state: RAGState) -> RAGState:
        """Node to generate final answer with citations"""
        print("🤖 Generating AI answer...")
        
        if not state["retrieved_docs"]:
            return {
                **state,
                "final_answer": "I couldn't find relevant information to answer your question.",
                "citations": []
            }
        
        try:
            # Prepare context from retrieved documents
            context_parts = []
            citations = []
            seen_urls = set()
            
            for i, doc_dict in enumerate(state["retrieved_docs"]):
                context_parts.append(f"Source {i+1}: {doc_dict['page_content']}")
                url = doc_dict["metadata"]["source"]
                if url not in seen_urls:
                    citations.append({
                        "url": url,
                        "title": f"Source {len(citations)+1}"
                    })
                    seen_urls.add(url)
            
            context = "\n\n".join(context_parts)
            
            # Create prompt based on answer length
            length_instructions = {
                "short": "Provide a concise answer in 1-2 paragraphs.",
                "medium": "Provide a comprehensive answer in 3-4 paragraphs.",
                "detailed": "Provide a detailed, thorough answer with multiple paragraphs covering all relevant aspects."
            }
            
            prompt_template = ChatPromptTemplate.from_template(
                """You are an AI assistant that provides accurate answers based on the given context.

Context from multiple sources:
{context}

Question: {question}

Instructions:
- {length_instruction}
- Base your answer strictly on the provided context
- If the context doesn't contain enough information, say so
- Be objective and factual
- Include specific details when available
- Organize your response clearly

Answer:"""
            )
            
            chain = prompt_template | self.llm
            
            response = chain.invoke({
                "context": context,
                "question": state["query"],
                "length_instruction": length_instructions[state["answer_length"]]
            })
            
            final_answer = response.content
            print("✅ Generated AI answer successfully")
            
            return {
                **state,
                "final_answer": final_answer,
                "citations": citations
            }
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "final_answer": f"Sorry, I encountered an error while generating the answer: {str(e)}",
                "citations": [],
                "error_messages": state.get("error_messages", []) + [error_msg]
            }
    
    def process_query(self, urls: List[str], query: str, answer_length: str = "medium"):
        """Process a query against multiple URLs"""
        print(f"\n🚀 Starting RAG pipeline with {len(urls)} URLs")
        print(f"Query: {query}")
        print(f"Answer length: {answer_length}")
        
        # Create initial state
        initial_state = RAGState(
            urls=urls,
            raw_content={},
            documents=[],
            embeddings_ready=False,
            query=query,
            retrieved_docs=[],
            answer_length=answer_length,
            final_answer="",
            citations=[],
            error_messages=[]
        )
        
        # Run the graph WITHOUT memory checkpointing
        result = self.graph.invoke(initial_state)
        
        return result

def create_gradio_interface():
    """Create Gradio interface"""
    
    # Global variable to store the RAG system
    rag_system = None

    def initialize_system():
        nonlocal rag_system
        api_key = os.getenv("OPENROUTER_API_KEY")  # <-- Fetch from HF Spaces secrets
        if not api_key:
            return "❌ OPENROUTER_API_KEY not found in environment. Please set it in HF Spaces Secrets."
        
        try:
            rag_system = MultiURLRAGSystem(api_key)
            return "✅ System initialized successfully!"
        except Exception as e:
            return f"❌ Error initializing system: {str(e)}"
    
    def process_urls_and_query(urls_text, query, answer_length):
        nonlocal rag_system
        
        if not rag_system:
            return "❌ Please initialize the system first (API key missing).", ""
        
        if not urls_text.strip():
            return "❌ Please provide at least one URL.", ""
        
        if not query.strip():
            return "❌ Please provide a query.", ""
        
        # Parse URLs
        urls = [url.strip() for url in urls_text.strip().split('\n') if url.strip()]
        
        if not urls:
            return "❌ No valid URLs found.", ""
        
        try:
            # Process the query
            result = rag_system.process_query(urls, query, answer_length)
            
            # Format response
            answer = result["final_answer"]
            
            # Add citations
            citations_text = ""
            if result["citations"]:
                citations_text = "\n\n**Sources:**\n"
                for i, citation in enumerate(result["citations"], 1):
                    citations_text += f"{i}. {citation['url']}\n"
            
            # Add error messages if any
            error_text = ""
            if result.get("error_messages"):
                error_text = "\n\n**Warnings/Errors:**\n"
                for error in result["error_messages"]:
                    error_text += f"• {error}\n"
            
            full_response = answer + citations_text + error_text
            
            return full_response, f"Processed {len(result.get('raw_content', {}))} URLs successfully"
            
        except Exception as e:
            return f"❌ Error processing query: {str(e)}", ""
    
    # Create Gradio interface
    with gr.Blocks(title="Multi-URL RAG System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Multi-URL RAG System")
        gr.Markdown("Ask questions across multiple websites using AI-powered search and retrieval.")
        
        with gr.Row():
            with gr.Column(scale=1):
                init_btn = gr.Button("Initialize System", variant="primary")
                init_status = gr.Textbox(label="Status", interactive=False)
                
        with gr.Row():
            with gr.Column(scale=2):
                urls_input = gr.Textbox(
                    label="Website URLs (one per line)",
                    placeholder="https://example1.com\nhttps://example2.com\nhttps://example3.com",
                    lines=5
                )
                
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What would you like to know from these websites?",
                    lines=2
                )
                
                answer_length = gr.Radio(
                    choices=["short", "medium", "detailed"],
                    value="medium",
                    label="Answer Length"
                )
                
                submit_btn = gr.Button("Ask Question", variant="primary", size="lg")
                
            with gr.Column(scale=3):
                answer_output = gr.Textbox(
                    label="AI Answer",
                    lines=20,
                    max_lines=30
                )
                
                status_output = gr.Textbox(
                    label="Processing Status",
                    lines=2
                )
        
        # Event handlers
        init_btn.click(
            initialize_system,
            inputs=[],
            outputs=[init_status]
        )
        
        submit_btn.click(
            process_urls_and_query,
            inputs=[urls_input, query_input, answer_length],
            outputs=[answer_output, status_output]
        )
        
        # Examples
        gr.Examples(
            examples=[
                [
                    "https://en.wikipedia.org/wiki/Artificial_intelligence\nhttps://en.wikipedia.org/wiki/Machine_learning",
                    "What is the difference between AI and machine learning?",
                    "medium"
                ],
                [
                    "https://docs.python.org/3/tutorial/\nhttps://realpython.com/python-basics/",
                    "How do I get started with Python programming?",
                    "detailed"
                ]
            ],
            inputs=[urls_input, query_input, answer_length]
        )
    
    return demo


if __name__ == "__main__":
    # For local testing
    demo = create_gradio_interface()
    demo.launch(share=True)
