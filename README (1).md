# Multi-URL RAG System

A powerful RAG (Retrieval-Augmented Generation) system that allows users to ask questions across multiple websites using LangChain, LangGraph, OpenRouter, and Gradio.

## Demo
Access this link for a live demo, if it is not working, checkout the video below - 
[Demo link](https://huggingface.co/spaces/ajnx014/Multi-URL-RAG-LangGraph)

### Video 

https://github.com/user-attachments/assets/a2c89f10-fd70-4528-b6e0-c2083fc1c7d9



## Features

- 🌐 **Multi-URL Support**: Process multiple websites simultaneously
- 🧠 **Smart Chunking**: Intelligent text splitting and embedding
- 🔍 **Vector Search**: FAISS-powered similarity search
- 🤖 **AI Answers**: OpenRouter integration for high-quality responses
- 📏 **Flexible Length**: Short, medium, or detailed answers
- 📚 **Citations**: Automatic source attribution
- 🔄 **LangGraph Workflow**: Node-based processing pipeline
- 🎨 **Web Interface**: Clean Gradio UI

## Architecture

The system uses LangGraph to orchestrate a multi-node workflow:

```
📥 Fetch Content → 📄 Process Documents → 🔍 Create Vector Store → 🔎 Retrieve Documents → 🤖 Generate Answer
```

### Node Details

1. **Fetch Content**: Downloads and cleans HTML from URLs
2. **Process Documents**: Chunks text and creates Document objects
3. **Create Vector Store**: Generates embeddings and builds FAISS index
4. **Retrieve Documents**: Finds relevant chunks based on query
5. **Generate Answer**: Uses LLM to compose final response with citations

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get OpenRouter API Key

1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up and get your API key
3. You can use various models through OpenRouter (GPT-4, Claude, Llama, etc.)

### 3. Run Locally

```bash
python app.py
```

## Deployment on Hugging Face Spaces

### 1. Create a new Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose "Gradio" as the SDK
4. Set visibility to Public or Private

### 2. Upload Files

Upload these files to your Space:
- `app.py` (main application)
- `requirements.txt`
- `README.md`

### 3. Configure Space

Create an `app.py` file in your Space with the main code, ensuring the last lines are:

```python
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
```

### 4. Environment Variables (Optional)

You can set a default API key in Space secrets:
- Go to your Space settings
- Add a secret named `OPENROUTER_API_KEY`
- Users can still override this in the UI

## Usage

1. **Initialize**: Enter your OpenRouter API key and click "Initialize System"
2. **Add URLs**: Paste website URLs (one per line)
3. **Ask Question**: Enter your question
4. **Choose Length**: Select answer length (short/medium/detailed)
5. **Get Answer**: Click "Ask Question" to get AI-powered response with citations

## Example Use Cases

- **Research**: Compare information across multiple sources
- **Documentation**: Query multiple API docs or tutorials
- **News Analysis**: Analyze multiple news articles
- **Product Research**: Compare features across different websites

## Configuration Options

### Models

You can change the OpenRouter model in the `MultiURLRAGSystem` constructor:

```python
self.llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    model="anthropic/claude-3-haiku",  # or "openai/gpt-4", "meta-llama/llama-2-70b-chat", etc.
    temperature=0.1
)
```

### Embedding Model

Change the embedding model for different languages or performance:

```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # Better quality
    # model_name="sentence-transformers/all-MiniLM-L6-v2"  # Faster
)
```

### Chunking Strategy

Adjust text splitting parameters:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Larger chunks
    chunk_overlap=300,  # More overlap
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
```

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your OpenRouter API key is valid
2. **URL Fetch Failure**: Some websites block automated requests
3. **Memory Issues**: Large documents may require more RAM
4. **Slow Processing**: Consider using faster embedding models

### Performance Tips

- Use fewer URLs for faster processing
- Choose appropriate chunk sizes for your content
- Consider caching embeddings for frequently used URLs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the OpenRouter documentation for API issues
- Review LangChain/LangGraph docs for framework questions
