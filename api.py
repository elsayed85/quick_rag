"""
FastAPI REST API for the Books RAG System.

This API provides a simple endpoint to query the RAG system.

Usage:
    1. Make sure Qdrant is running and embeddings are generated
    2. Run: uvicorn api:app --reload
    3. Send POST requests to http://localhost:8000/api/ask
"""

import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "school_books"

# Initialize FastAPI app
app = FastAPI(
    title="Books RAG API",
    description="Retrieval-Augmented Generation API for answering questions based on PDF documents",
    version="1.0.0"
)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM and embeddings
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Initialize vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# ============================================
# Pydantic Models
# ============================================

class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(..., description="The question to ask", min_length=1)
    include_sources: Optional[bool] = Field(
        default=False, 
        description="Whether to include source documents in the response"
    )


class SourceDocument(BaseModel):
    """Model for source document metadata."""
    source_file: str
    page: int
    content_preview: str


class QuestionResponse(BaseModel):
    """Response model for the answer."""
    question: str
    answer: str
    sources: Optional[list[SourceDocument]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    qdrant_connected: bool
    collection_exists: bool
    documents_count: int


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


# ============================================
# Global State for Sources
# ============================================
retrieved_sources = []


# ============================================
# Retriever Tool
# ============================================

@tool
def retrieve_from_books(query: str) -> str:
    """Search and return information from school books (math, science, etc.)."""
    global retrieved_sources
    retrieved_sources = []
    
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant information found in the books."
    
    results = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        
        # Store source information
        retrieved_sources.append({
            "source_file": source,
            "page": int(page) if isinstance(page, (int, float)) else 0,
            "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        })
        
        results.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(results)


retriever_tool = retrieve_from_books


# ============================================
# Graph Components
# ============================================

GRADE_PROMPT = """You are a grader assessing relevance of retrieved documents to a student's question.

Here is the retrieved document:
{context}

Here is the student's question: {question}

If the document contains information that could help answer the student's question (keywords, concepts, formulas, explanations related to the topic), grade it as relevant.

Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""


def grade_documents(state: MessagesState):
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    grader_llm = llm.with_structured_output(GradeDocuments)
    response = grader_llm.invoke([{"role": "user", "content": prompt}])

    if response.binary_score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


def generate_query_or_respond(state: MessagesState):
    """Call the model to decide whether to retrieve from books or respond directly."""
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


REWRITE_PROMPT = """You are helping to improve a student's question for better search results.

The previous search didn't return relevant results. Please rewrite the question to be more specific or use different terminology that might be found in school textbooks.

Original question: {question}

Rewrite the question to improve search results (keep it focused on the educational topic):"""


def rewrite_question(state: MessagesState):
    """Rewrite the original question for better retrieval."""
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


GENERATE_PROMPT = """You are a helpful educational assistant for students. Use the following information from school books to answer the student's question.

Instructions:
- Provide a clear, educational explanation suitable for students
- If the context includes formulas or equations, explain them step by step
- If you're not sure about something, say so
- Keep the answer focused and concise

Question: {question}

Context from books:
{context}

Answer:"""


def generate_answer(state: MessagesState):
    """Generate an answer based on the retrieved context."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm.invoke([{"role": "user", "content": prompt}])
    
    return {"messages": [response]}


def build_graph():
    """Build and compile the RAG agent graph."""
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    # Add edges
    workflow.add_edge(START, "generate_query_or_respond")

    # Conditional edge: decide whether to retrieve or respond directly
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    # After retrieval, grade the documents
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )

    # After generating answer, end
    workflow.add_edge("generate_answer", END)

    # After rewriting, try to retrieve again
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    return workflow.compile()


# ============================================
# API Endpoints
# ============================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Books RAG API",
        "version": "1.0.0",
        "description": "API for answering questions using RAG over PDF documents",
        "endpoints": {
            "POST /api/ask": "Ask a question",
            "GET /health": "Health check"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the API and its dependencies."""
    try:
        # Check Qdrant connection
        collection_info = client.get_collection(COLLECTION_NAME)
        return HealthResponse(
            status="healthy",
            qdrant_connected=True,
            collection_exists=True,
            documents_count=collection_info.points_count
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            qdrant_connected=False,
            collection_exists=False,
            documents_count=0
        )


@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer from the RAG system.
    
    Args:
        request: QuestionRequest containing the question and optional flags
        
    Returns:
        QuestionResponse with the answer and optional source documents
        
    Raises:
        HTTPException: If there's an error processing the question
    """
    global retrieved_sources
    
    try:
        # Reset sources
        retrieved_sources = []
        
        # Build and run the graph
        graph = build_graph()
        
        # Invoke the graph and get the final state
        result = graph.invoke({"messages": [{"role": "user", "content": request.question}]})
        
        # Extract the final response from the last message
        final_response = result["messages"][-1].content
        
        if not final_response:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate a response. Please try rephrasing your question."
            )
        
        # Prepare response
        response = QuestionResponse(
            question=request.question,
            answer=final_response
        )
        
        # Include sources if requested
        if request.include_sources and retrieved_sources:
            response.sources = [
                SourceDocument(**source) for source in retrieved_sources
            ]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )


# ============================================
# Startup Event
# ============================================

@app.on_event("startup")
async def startup_event():
    """Run checks on startup."""
    print("🚀 Starting Books RAG API...")
    
    # Check Qdrant connection
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"✅ Connected to Qdrant")
        print(f"📚 Collection: {COLLECTION_NAME}")
        print(f"📄 Documents: {collection_info.points_count}")
        
        if collection_info.points_count == 0:
            print("⚠️  Warning: Collection is empty. Run generate_embeddings.py first.")
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        print("⚠️  Make sure Qdrant is running: docker-compose up -d")
    
    print("\n📡 API is ready!")
    print("📖 Docs: http://localhost:8000/docs")
    print("🔍 Health: http://localhost:8000/health")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

