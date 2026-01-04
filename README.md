# Books RAG System

A Retrieval-Augmented Generation (RAG) system for answering questions based on PDF documents using LangGraph, LangChain, OpenAI, and Qdrant vector database.

## 🏗️ System Architecture

This RAG system consists of three main components:

1. **Qdrant Vector Database** - Stores document embeddings for semantic search
2. **Embedding Generator** - Processes PDFs and creates searchable embeddings
3. **RAG Agent** - Intelligent question-answering system with document retrieval

## 📋 Prerequisites

- Docker and Docker Compose installed
- Python 3.8 or higher
- OpenAI API key

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:elsayed85/quick_rag.git
cd quick_rag
```

### 2. Configure Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Start Qdrant Vector Database

Start the Qdrant vector database using Docker Compose:

```bash
docker-compose up -d
```

This will start Qdrant on:
- HTTP: `http://localhost:6333`
- gRPC: `http://localhost:6334`

Verify Qdrant is running:
```bash
curl http://localhost:6333/health
```

### 4. Install Python Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

### 5. Add Your PDF Documents

Place your PDF files in the `books/` directory:

```bash
cp your_book.pdf books/
```

### 6. Generate Embeddings

Process the PDFs and create vector embeddings:

```bash
python generate_embeddings.py
```

This script will:
- Load all PDF files from the `books/` directory
- Split documents into chunks (1000 characters with 200 overlap)
- Generate embeddings using OpenAI's `text-embedding-3-small` model
- Store embeddings in Qdrant vector database

### 7. Run the RAG System

You have two options to use the RAG system:

#### Option A: Interactive CLI

Start the interactive question-answering system:

```bash
python rag.py
```

#### Option B: REST API

Start the FastAPI server:

```bash
uvicorn api:app --reload
```

The API will be available at:
- **API Endpoint**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 💡 How It Works

### System Flow

```
┌─────────────────────────────────────────────────────────┐
│                     User Question                        │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   RAG Agent (LangGraph)                  │
│                                                          │
│  1. Question Analysis                                    │
│     ├─ Need retrieval? → Search Qdrant                  │
│     └─ Simple question? → Direct answer                 │
│                                                          │
│  2. Document Retrieval (if needed)                       │
│     └─ Semantic search in vector store                  │
│                                                          │
│  3. Relevance Grading                                    │
│     ├─ Relevant docs? → Generate answer                 │
│     └─ Not relevant? → Rewrite question & retry         │
│                                                          │
│  4. Answer Generation                                    │
│     └─ LLM synthesizes answer from context              │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Final Answer    │
              └──────────────────┘
```

### Component Details

#### 1. **Embedding Generation** (`generate_embeddings.py`)

- Loads PDF documents from `books/` directory
- Splits text into manageable chunks using `RecursiveCharacterTextSplitter`
- Creates vector embeddings using OpenAI embeddings
- Stores in Qdrant collection named `school_books`

#### 2. **RAG Agent** (`rag.py`)

The agent uses a sophisticated workflow powered by LangGraph:

- **Router Node**: Decides if vector search is needed
- **Retrieval Node**: Fetches relevant document chunks
- **Grading Node**: Evaluates document relevance
- **Rewriting Node**: Reformulates question if needed
- **Generation Node**: Creates final answer

#### 3. **Vector Database** (Qdrant)

- Provides fast semantic search capabilities
- Persists embeddings in Docker volume
- Accessible via REST API and gRPC

## 🎯 Usage Examples

### Interactive CLI Mode

```bash
python rag.py
```

Then ask questions like:
```
You: What is distributed systems?
You: Explain the CAP theorem
You: What are the benefits of microservices?
```

Type `quit`, `exit`, or `q` to stop.

### REST API Mode

Start the API server:

```bash
uvicorn api:app --reload
```

**Example API Request:**

```bash
curl -X POST "http://localhost:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is distributed systems?",
    "include_sources": true
  }'
```

**Example Response:**

```json
{
  "question": "What is distributed systems?",
  "answer": "A distributed system is a collection of independent computers that appears to its users as a single coherent system. These computers communicate and coordinate their actions by passing messages over a network...",
  "sources": [
    {
      "source_file": "intro_to_distributed_systems.pdf",
      "page": 5,
      "content_preview": "Distributed systems are computing environments where multiple..."
    }
  ]
}
```

**Using Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/ask",
    json={
        "question": "Explain the CAP theorem",
        "include_sources": True
    }
)

data = response.json()
print(f"Answer: {data['answer']}")
```

**Using JavaScript:**

```javascript
fetch('http://localhost:8000/api/ask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'What are microservices?',
    include_sources: true
  })
})
.then(response => response.json())
.then(data => console.log(data.answer));
```

## 🛠️ Project Structure

```
books_rag/
├── books/                          # PDF documents directory
│   └── intro_to_distributed_systems.pdf
├── api.py                          # FastAPI REST API server
├── docker-compose.yml              # Qdrant container configuration
├── generate_embeddings.py          # Embedding generation script
├── generate_pdf.py                 # PDF generation utility
├── rag.py                          # Interactive CLI RAG agent
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 📡 API Reference

### Endpoints

#### `GET /`
Get API information and available endpoints.

**Response:**
```json
{
  "name": "Books RAG API",
  "version": "1.0.0",
  "description": "API for answering questions using RAG over PDF documents",
  "endpoints": {
    "POST /api/ask": "Ask a question",
    "GET /health": "Health check"
  }
}
```

#### `GET /health`
Check the health of the API and Qdrant connection.

**Response:**
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "collection_exists": true,
  "documents_count": 42
}
```

#### `POST /api/ask`
Ask a question and receive an AI-generated answer.

**Request Body:**
```json
{
  "question": "What is a distributed system?",
  "include_sources": true
}
```

**Parameters:**
- `question` (string, required): The question to ask
- `include_sources` (boolean, optional): Whether to include source documents (default: false)

**Response:**
```json
{
  "question": "What is a distributed system?",
  "answer": "A distributed system is...",
  "sources": [
    {
      "source_file": "intro_to_distributed_systems.pdf",
      "page": 5,
      "content_preview": "Distributed systems are..."
    }
  ]
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Your OpenAI API key (required) |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |

### Embedding Settings

Modify in `generate_embeddings.py`:

```python
CHUNK_SIZE = 1000        # Characters per chunk
CHUNK_OVERLAP = 200      # Overlap between chunks
COLLECTION_NAME = "school_books"  # Qdrant collection name
```

## 📦 Dependencies

- **langgraph** - Workflow orchestration
- **langchain** - LLM framework
- **langchain-openai** - OpenAI integration
- **langchain-qdrant** - Qdrant vector store
- **qdrant-client** - Qdrant Python client
- **pypdf** - PDF processing
- **python-dotenv** - Environment management
- **fastapi** - REST API framework
- **uvicorn** - ASGI server

## 🔍 Troubleshooting

### Qdrant not responding

```bash
# Check if container is running
docker ps

# Restart Qdrant
docker-compose restart

# View logs
docker-compose logs -f qdrant
```

### No embeddings generated

- Ensure PDF files are in `books/` directory
- Check Qdrant is running: `curl http://localhost:6333/health`
- Verify OpenAI API key is set correctly

### Import errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 🧹 Cleanup

Stop and remove containers:

```bash
docker-compose down
```

Remove Qdrant data volume:

```bash
docker-compose down -v
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
