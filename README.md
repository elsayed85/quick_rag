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

Start the interactive question-answering system:

```bash
python rag.py
```

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

### Interactive Mode

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

## 🛠️ Project Structure

```
books_rag/
├── books/                          # PDF documents directory
│   └── intro_to_distributed_systems.pdf
├── docker-compose.yml              # Qdrant container configuration
├── generate_embeddings.py          # Embedding generation script
├── generate_pdf.py                 # PDF generation utility
├── rag.py                          # RAG agent implementation
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
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
