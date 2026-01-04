"""
Generate embeddings from PDF files and store them in Qdrant.

Usage:
    1. Place your PDF files in the ./books/ directory
    2. Make sure Qdrant is running: docker-compose up -d
    3. Run: python generate_embeddings.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# Configuration
BOOKS_DIR = Path("./books")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "school_books"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_pdfs(books_dir: Path) -> list:
    """Load all PDF files from the specified directory."""
    documents = []
    pdf_files = list(books_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {books_dir}")
        return documents

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            # Add source metadata
            for doc in docs:
                doc.metadata["source_file"] = pdf_path.name
            documents.extend(docs)
            print(f"  - Loaded {len(docs)} pages")
        except Exception as e:
            print(f"  - Error loading {pdf_path.name}: {e}")

    return documents


def split_documents(documents: list) -> list:
    """Split documents into smaller chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Create Qdrant collection if it doesn't exist."""
    if client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting and recreating...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    print(f"Created collection '{collection_name}'")


def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it in your .env file or export it")
        return

    # Create books directory if it doesn't exist
    BOOKS_DIR.mkdir(exist_ok=True)

    # Load PDFs
    print("\n=== Loading PDF files ===")
    documents = load_pdfs(BOOKS_DIR)

    if not documents:
        print("\nNo documents to process. Please add PDF files to the ./books/ directory")
        return

    # Split documents
    print("\n=== Splitting documents ===")
    chunks = split_documents(documents)

    # Initialize embeddings
    print("\n=== Initializing embeddings ===")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Get vector size by embedding a sample text
    sample_embedding = embeddings.embed_query("sample text")
    vector_size = len(sample_embedding)
    print(f"Embedding dimension: {vector_size}")

    # Connect to Qdrant
    print("\n=== Connecting to Qdrant ===")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Create collection
    create_qdrant_collection(client, COLLECTION_NAME, vector_size)

    # Create vector store and add documents
    print("\n=== Generating embeddings and storing in Qdrant ===")
    print("This may take a while depending on the number of documents...")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # Add documents in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"  - Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"\n=== Done! ===")
    print(f"Stored {len(chunks)} document chunks in Qdrant collection '{COLLECTION_NAME}'")
    print(f"You can now run the RAG system with: python rag.py")


if __name__ == "__main__":
    main()
