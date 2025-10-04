import os
import re
from typing import List, Dict, Any, Optional, Annotated
import requests
from bs4 import BeautifulSoup
import gradio as gr

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# Enhanced State definition with proper typing
class RAGState(TypedDict):
    urls: List[str]
    raw_content: Dict[str, str]
    documents: List[Document]
    query: str
    answer_length: str
    retrieved_docs: List[Document]
    final_answer: str
    citations: List[Dict[str, str]]
    error_messages: List[str]
    processing_status: str
    retry_count: int

class MultiURLRAGSystem:
    def __init__(self, openrouter_api_key: str):
        self.openrouter_api_key = openrouter_api_key
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize LLM with OpenRouter
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            model="openai/gpt-3.5-turbo",
            temperature=0.1
        )
        
        # Store vector store
        self.vector_store = None
        
        # Create LangGraph workflow with memory
        self.memory = MemorySaver()
        self.graph = self._create_graph()
        
    def _create_graph(self):
        """Create enhanced LangGraph workflow with conditional routing and error handling"""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("fetch_content", self.fetch_content_node)
        workflow.add_node("validate_content", self.validate_content_node)
        workflow.add_node("process_documents", self.process_documents_node)
        workflow.add_node("create_embeddings", self.create_embeddings_node)
        workflow.add_node("retrieve_context", self.retrieve_context_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        workflow.add_node("handle_error", self.handle_error_node)
        
        # Set entry point
        workflow.set_entry_point("fetch_content")
        
        # Add conditional edges for error handling and routing
        workflow.add_conditional_edges(
            "fetch_content",
            self.should_continue_after_fetch,
            {
                "continue": "validate_content",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_content",
            self.should_retry_fetch,
            {
                "continue": "process_documents",
                "retry": "fetch_content",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("process_documents", "create_embeddings")
        
        workflow.add_conditional_edges(
            "create_embeddings",
            self.should_continue_to_retrieval,
            {
                "continue": "retrieve_context",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("handle_error", END)
        
        # Compile with memory checkpointer
        return workflow.compile(checkpointer=self.memory)
    
    # Conditional routing functions
    def should_continue_after_fetch(self, state: RAGState) -> str:
        """Decide if we should continue after fetching content"""
        if state["raw_content"]:
            return "continue"
        return "error"
    
    def should_retry_fetch(self, state: RAGState) -> str:
        """Decide if we should retry fetching or continue"""
        if not state["raw_content"] and state["retry_count"] < 2:
            return "retry"
        elif state["raw_content"]:
            return "continue"
        return "error"
    
    def should_continue_to_retrieval(self, state: RAGState) -> str:
        """Decide if embeddings were created successfully"""
        if self.vector_store is not None and state["documents"]:
            return "continue"
        return "error"
    
    # Node implementations
    def clean_html_content(self, html_content: str, base_url: str = "") -> str:
        """Clean and extract text from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
        except Exception as e:
            print(f"Error cleaning HTML: {e}")
            return ""
    
    def fetch_url_content(self, url: str) -> Dict[str, str]:
        """Fetch content from a single URL with retry logic"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            content = self.clean_html_content(response.text, url)
            
            if len(content) < 100:
                return {
                    "url": url,
                    "content": "",
                    "status": "error",
                    "error": "Content too short (possible paywall or blocked access)"
                }
            
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
        print(f"📥 Fetching content (Attempt {state.get('retry_count', 0) + 1})...")
        raw_content = state.get("raw_content", {})
        error_messages = state.get("error_messages", [])
        
        for url in state["urls"]:
            if url in raw_content:  # Skip already fetched URLs
                continue
                
            result = self.fetch_url_content(url)
            if result["status"] == "success" and result["content"]:
                raw_content[url] = result["content"]
                print(f"✅ Fetched: {url[:50]}...")
            else:
                error_msg = f"Failed: {url[:50]}... - {result.get('error', 'Unknown')}"
                error_messages.append(error_msg)
                print(f"❌ {error_msg}")
        
        return {
            **state,
            "raw_content": raw_content,
            "error_messages": error_messages,
            "processing_status": f"Fetched {len(raw_content)}/{len(state['urls'])} URLs",
            "retry_count": state.get("retry_count", 0) + 1
        }
    
    def validate_content_node(self, state: RAGState) -> RAGState:
        """Validate fetched content quality"""
        print("🔍 Validating content...")
        
        valid_content = {}
        error_messages = state.get("error_messages", [])
        
        for url, content in state["raw_content"].items():
            word_count = len(content.split())
            if word_count < 50:
                error_messages.append(f"Content from {url} is too short ({word_count} words)")
            else:
                valid_content[url] = content
        
        print(f"✅ Validated {len(valid_content)} URLs")
        
        return {
            **state,
            "raw_content": valid_content,
            "error_messages": error_messages,
            "processing_status": f"Validated {len(valid_content)} URLs"
        }
    
    def process_documents_node(self, state: RAGState) -> RAGState:
        """Process and chunk documents using LangChain"""
        print("📄 Processing documents...")
        documents = []
        
        for url, content in state["raw_content"].items():
            # Use LangChain's text splitter
            chunks = self.text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": url,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
        
        print(f"✅ Created {len(documents)} chunks from {len(state['raw_content'])} URLs")
        
        return {
            **state,
            "documents": documents,
            "processing_status": f"Created {len(documents)} document chunks"
        }
    
    def create_embeddings_node(self, state: RAGState) -> RAGState:
        """Create vector embeddings using LangChain FAISS"""
        print("🔮 Creating embeddings...")
        
        if not state["documents"]:
            return {
                **state,
                "error_messages": state.get("error_messages", []) + ["No documents to embed"],
                "processing_status": "Error: No documents"
            }
        
        try:
            # Use LangChain's FAISS vector store
            self.vector_store = FAISS.from_documents(
                state["documents"],
                self.embeddings
            )
            print(f"✅ Created vector store with {len(state['documents'])} embeddings")
            
            return {
                **state,
                "processing_status": f"Embeddings created for {len(state['documents'])} chunks"
            }
        except Exception as e:
            error_msg = f"Embedding error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "error_messages": state.get("error_messages", []) + [error_msg],
                "processing_status": "Error creating embeddings"
            }
    
    def retrieve_context_node(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents using LangChain retriever"""
        print(f"🔎 Retrieving context for: {state['query'][:50]}...")
        
        if not self.vector_store:
            return {
                **state,
                "retrieved_docs": [],
                "error_messages": state.get("error_messages", []) + ["No vector store available"]
            }
        
        try:
            # Adjust retrieval based on answer length
            k_map = {"short": 3, "medium": 6, "detailed": 10}
            k = k_map.get(state["answer_length"], 6)
            
            # Use LangChain's similarity search with score
            retrieved_docs = self.vector_store.similarity_search(
                state["query"],
                k=k
            )
            
            print(f"✅ Retrieved {len(retrieved_docs)} relevant chunks")
            
            return {
                **state,
                "retrieved_docs": retrieved_docs,
                "processing_status": f"Retrieved {len(retrieved_docs)} relevant chunks"
            }
        except Exception as e:
            error_msg = f"Retrieval error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "retrieved_docs": [],
                "error_messages": state.get("error_messages", []) + [error_msg]
            }
    
    def generate_answer_node(self, state: RAGState) -> RAGState:
        """Generate answer using LangChain's chain composition"""
        print("🤖 Generating answer...")
        
        if not state["retrieved_docs"]:
            return {
                **state,
                "final_answer": "I couldn't find relevant information to answer your question.",
                "citations": [],
                "processing_status": "No relevant content found"
            }
        
        try:
            # Extract citations
            citations = []
            seen_urls = set()
            
            for doc in state["retrieved_docs"]:
                url = doc.metadata["source"]
                if url not in seen_urls:
                    citations.append({
                        "url": url,
                        "title": f"Source {len(citations) + 1}"
                    })
                    seen_urls.add(url)
            
            # Length instructions
            length_map = {
                "short": "Provide a concise answer in 2-3 sentences.",
                "medium": "Provide a comprehensive answer in 2-3 paragraphs.",
                "detailed": "Provide a detailed, thorough answer with multiple paragraphs."
            }
            
            # Create LangChain prompt
            prompt = ChatPromptTemplate.from_template(
                """You are a helpful AI assistant that answers questions based on provided context.
Context from web sources:
{context}
Question: {question}
Instructions:
- {length_instruction}
- Base your answer strictly on the provided context
- Be specific and cite relevant details
- If the context lacks information, state this clearly
- Organize your response in a clear, readable format
Answer:"""
            )
            
            # Format context from documents
            context = "\n\n".join([
                f"[Source {i+1}]: {doc.page_content}"
                for i, doc in enumerate(state["retrieved_docs"])
            ])
            
            # Create and invoke chain using LangChain's LCEL
            chain = prompt | self.llm | StrOutputParser()
            
            final_answer = chain.invoke({
                "context": context,
                "question": state["query"],
                "length_instruction": length_map[state["answer_length"]]
            })
            
            print("✅ Answer generated successfully")
            
            return {
                **state,
                "final_answer": final_answer,
                "citations": citations,
                "processing_status": "Answer generated successfully"
            }
            
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "final_answer": f"Sorry, I encountered an error: {str(e)}",
                "citations": [],
                "error_messages": state.get("error_messages", []) + [error_msg],
                "processing_status": "Error generating answer"
            }
    
    def handle_error_node(self, state: RAGState) -> RAGState:
        """Handle errors gracefully"""
        print("⚠️ Handling errors...")
        
        error_summary = "\n".join(state.get("error_messages", []))
        
        return {
            **state,
            "final_answer": f"I encountered issues processing your request:\n\n{error_summary}\n\nPlease check the URLs and try again.",
            "citations": [],
            "processing_status": "Error occurred"
        }
    
    def process_query(self, urls: List[str], query: str, answer_length: str = "medium"):
        """Process a query using LangGraph workflow"""
        print(f"\n🚀 Starting RAG pipeline")
        print(f"📊 URLs: {len(urls)}")
        print(f"❓ Query: {query[:100]}...")
        
        # Create initial state
        initial_state = RAGState(
            urls=urls,
            raw_content={},
            documents=[],
            query=query,
            answer_length=answer_length,
            retrieved_docs=[],
            final_answer="",
            citations=[],
            error_messages=[],
            processing_status="Starting...",
            retry_count=0
        )
        
        # Run the graph with a unique thread ID for memory
        config = {"configurable": {"thread_id": "session_1"}}
        result = self.graph.invoke(initial_state, config)
        
        print(f"✅ Pipeline complete: {result['processing_status']}")
        
        return result


def create_gradio_interface():
    """Create simplified, goal-focused Gradio interface"""
    
    # Initialize system on startup
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment variables. Please add it to HF Spaces Secrets.")
    
    rag_system = MultiURLRAGSystem(api_key)
    print("✅ RAG System initialized successfully!")
    
    def process_question(urls_text, query, answer_length):
        if not urls_text.strip():
            return "❌ Please add at least one URL", "", ""
        
        if not query.strip():
            return "❌ Please enter your question", "", ""
        
        # Parse URLs
        urls = [url.strip() for url in urls_text.strip().split('\n') if url.strip()]
        
        try:
            # Process with LangGraph workflow
            result = rag_system.process_query(urls, query, answer_length)
            
            # Format answer
            answer = result["final_answer"]
            
            # Format citations
            citations = ""
            if result["citations"]:
                citations = "\n\n**📚 Sources:**\n"
                for i, citation in enumerate(result["citations"], 1):
                    citations += f"{i}. {citation['url']}\n"
            
            # Format status
            status = f"✅ {result['processing_status']}"
            if result.get("error_messages"):
                status += f"\n⚠️ {len(result['error_messages'])} warning(s)"
            
            return answer, citations, status
            
        except Exception as e:
            return f"❌ Error: {str(e)}", "", "Failed"
    
    # Create clean, simple interface
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🔍 Multi-URL Question Answering
            ### How to use:
            1. **Add URLs** - Enter website links (one per line) you want to analyze
            2. **Ask your question** - Type what you want to know from those websites  
            3. **Choose answer length** - Select short, medium, or detailed response
            4. **Get AI-powered answer** - Receive a synthesized answer with source citations
            
            💡 *The AI reads all websites, extracts relevant information, and answers your question based on the content.*
            """
        )
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input")
                
                urls_input = gr.Textbox(
                    label="Website URLs",
                    placeholder="https://example.com/article-1\nhttps://example.com/article-2",
                    lines=6,
                    info="Enter one URL per line"
                )
                
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What would you like to know?",
                    lines=3
                )
                
                answer_length = gr.Radio(
                    choices=["short", "medium", "detailed"],
                    value="medium",
                    label="Answer Length",
                    info="Choose how detailed you want the answer"
                )
                
                submit_btn = gr.Button("✨ Get Answer", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                gr.Markdown("### 💡 Answer")
                
                answer_output = gr.Textbox(
                    label="",
                    lines=15,
                    show_label=False,
                    placeholder="Your answer will appear here..."
                )
                
                citations_output = gr.Textbox(
                    label="",
                    lines=4,
                    show_label=False
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False
                )
        
        # Examples
        gr.Markdown("---")
        gr.Markdown("### 📌 Try these examples:")
        
        gr.Examples(
            examples=[
                [
                    "https://en.wikipedia.org/wiki/Climate_change\nhttps://en.wikipedia.org/wiki/Global_warming",
                    "What are the main causes of climate change?",
                    "medium"
                ],
                [
                    "https://docs.python.org/3/tutorial/introduction.html",
                    "How do I use variables in Python?",
                    "short"
                ]
            ],
            inputs=[urls_input, query_input, answer_length]
        )
        
        # Event handlers
        submit_btn.click(
            process_question,
            inputs=[urls_input, query_input, answer_length],
            outputs=[answer_output, citations_output, status_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)
