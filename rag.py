"""
RAG Agent for answering student questions using LangGraph.

This agent:
1. Decides whether to retrieve from the vector store or respond directly
2. Retrieves relevant documents from school books
3. Grades the retrieved documents for relevance
4. Rewrites the question if documents aren't relevant
5. Generates a final answer based on the context

Usage:
    1. Make sure Qdrant is running and embeddings are generated
    2. Run: python rag.py
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

load_dotenv()

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "school_books"

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# ============================================
# Logging Utilities
# ============================================

def print_box(message: str, color=Fore.CYAN, width=70):
    """Print a message in a colored box."""
    top_border = f"{color}╔{'═' * (width - 2)}╗{Style.RESET_ALL}"
    bottom_border = f"{color}╚{'═' * (width - 2)}╝{Style.RESET_ALL}"
    
    print(top_border)
    lines = message.split('\n')
    for line in lines:
        padding = width - len(line) - 4
        print(f"{color}║{Style.RESET_ALL} {color}{line}{' ' * padding}{color} ║{Style.RESET_ALL}")
    print(bottom_border)


def print_section(title: str, color=Fore.CYAN, width=70):
    """Print a section header."""
    title_str = f" {title} "
    padding = (width - len(title_str) - 2) // 2
    line = f"{'─' * padding}{title_str}{'─' * (width - padding - len(title_str) - 2)}"
    print(f"{color}┌{line}┐{Style.RESET_ALL}")


def print_section_end(color=Fore.CYAN, width=70):
    """Print a section end."""
    print(f"{color}└{'─' * (width - 2)}┘{Style.RESET_ALL}")


def log_step(message: str, color=Fore.CYAN):
    """Log a step in the RAG process with color."""
    print(f"{color}│ ▸ {message}{Style.RESET_ALL}")


def log_info(message: str):
    """Log informational message."""
    print(f"{Fore.BLUE}│   ℹ {message}{Style.RESET_ALL}")


def log_success(message: str):
    """Log success message."""
    print(f"{Fore.GREEN}│   ✓ {message}{Style.RESET_ALL}")


def log_warning(message: str):
    """Log warning message."""
    print(f"{Fore.YELLOW}│   ⚠ {message}{Style.RESET_ALL}")


def log_decision(message: str):
    """Log decision point."""
    print(f"{Fore.MAGENTA}│   ➜ {message}{Style.RESET_ALL}")


def print_answer(answer: str, width=70):
    """Print the final answer in a colored box."""
    print(f"\n{Fore.GREEN}╔{'═' * (width - 2)}╗{Style.RESET_ALL}")
    print(f"{Fore.GREEN}║{Style.RESET_ALL} {Fore.WHITE + Style.BRIGHT}ASSISTANT RESPONSE{' ' * (width - 22)}{Fore.GREEN}║{Style.RESET_ALL}")
    print(f"{Fore.GREEN}╠{'═' * (width - 2)}╣{Style.RESET_ALL}")
    
    # Wrap text to fit in the box
    import textwrap
    wrapped_lines = []
    for line in answer.split('\n'):
        if line.strip():
            wrapped_lines.extend(textwrap.wrap(line, width - 4))
        else:
            wrapped_lines.append('')
    
    for line in wrapped_lines:
        padding = width - len(line) - 4
        print(f"{Fore.GREEN}║{Style.RESET_ALL} {Fore.CYAN}{line}{' ' * padding}{Fore.GREEN} ║{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}╚{'═' * (width - 2)}╝{Style.RESET_ALL}\n")


# ============================================
# Retriever Tool
# ============================================

@tool
def retrieve_from_books(query: str) -> str:
    """Search and return information from school books (math, science, etc.)."""
    print_section("🔍 VECTOR STORE RETRIEVAL", Fore.CYAN)
    log_step(f"Query: '{query}'", Fore.CYAN)
    docs = retriever.invoke(query)
    
    if not docs:
        log_warning("No documents found in vector store")
        print_section_end(Fore.CYAN)
        return "No relevant information found in the books."
    
    log_success(f"Retrieved {len(docs)} documents")
    print(f"{Fore.CYAN}│{Style.RESET_ALL}")
    
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        log_info(f"Doc {i}: {Fore.WHITE}{source}{Style.RESET_ALL} {Fore.YELLOW}(Page {page}){Style.RESET_ALL}")
        results.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    
    print_section_end(Fore.CYAN)
    return "\n\n---\n\n".join(results)


retriever_tool = retrieve_from_books


# ============================================
# Document Grading
# ============================================

class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


GRADE_PROMPT = """You are a grader assessing relevance of retrieved documents to a student's question.

Here is the retrieved document:
{context}

Here is the student's question: {question}

If the document contains information that could help answer the student's question (keywords, concepts, formulas, explanations related to the topic), grade it as relevant.

Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    print_section("📋 DOCUMENT RELEVANCE GRADING", Fore.YELLOW)
    question = state["messages"][0].content
    context = state["messages"][-1].content

    log_step("Evaluating document relevance...", Fore.YELLOW)
    prompt = GRADE_PROMPT.format(question=question, context=context)
    grader_llm = llm.with_structured_output(GradeDocuments)
    response = grader_llm.invoke([{"role": "user", "content": prompt}])

    print(f"{Fore.YELLOW}│{Style.RESET_ALL}")
    if response.binary_score == "yes":
        log_success(f"Relevance Score: {Fore.WHITE + Style.BRIGHT}{response.binary_score.upper()}{Style.RESET_ALL}")
        log_decision("Next Step: {Fore.WHITE}Generate Answer{Style.RESET_ALL}")
        print_section_end(Fore.YELLOW)
        return "generate_answer"
    else:
        log_warning(f"Relevance Score: {Fore.WHITE + Style.BRIGHT}{response.binary_score.upper()}{Style.RESET_ALL}")
        log_decision("Next Step: {Fore.WHITE}Rewrite Question{Style.RESET_ALL}")
        print_section_end(Fore.YELLOW)
        return "rewrite_question"


# ============================================
# Graph Nodes
# ============================================

def generate_query_or_respond(state: MessagesState):
    """
    Call the model to decide whether to retrieve from books or respond directly.
    For general greetings or off-topic questions, respond directly.
    For educational questions, use the retriever tool.
    """
    question = state["messages"][0].content if state["messages"] else "Unknown"
    question_preview = question[:60] + "..." if len(question) > 60 else question
    
    print_section("🤔 QUERY ANALYSIS", Fore.MAGENTA)
    log_step(f"Question: {Fore.WHITE}'{question_preview}'{Style.RESET_ALL}", Fore.MAGENTA)
    log_info("Analyzing query intent...")
    
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    
    print(f"{Fore.MAGENTA}│{Style.RESET_ALL}")
    # Check if tool was called
    if hasattr(response, 'tool_calls') and response.tool_calls:
        log_decision(f"Action: {Fore.WHITE + Style.BRIGHT}USE RETRIEVER TOOL{Style.RESET_ALL}")
    else:
        log_decision(f"Action: {Fore.WHITE + Style.BRIGHT}RESPOND DIRECTLY{Style.RESET_ALL}")
    
    print_section_end(Fore.MAGENTA)
    return {"messages": [response]}


REWRITE_PROMPT = """You are helping to improve a student's question for better search results.

The previous search didn't return relevant results. Please rewrite the question to be more specific or use different terminology that might be found in school textbooks.

Original question: {question}

Rewrite the question to improve search results (keep it focused on the educational topic):"""


def rewrite_question(state: MessagesState):
    """Rewrite the original question for better retrieval."""
    question = state["messages"][0].content
    question_preview = question[:60] + "..." if len(question) > 60 else question
    
    print_section("✏️  QUESTION REWRITING", Fore.YELLOW)
    log_step("Improving query for better results...", Fore.YELLOW)
    log_info(f"Original: {Fore.WHITE}'{question_preview}'{Style.RESET_ALL}")
    
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([{"role": "user", "content": prompt}])
    
    rewritten_preview = response.content[:60] + "..." if len(response.content) > 60 else response.content
    print(f"{Fore.YELLOW}│{Style.RESET_ALL}")
    log_success(f"Rewritten: {Fore.WHITE}'{rewritten_preview}'{Style.RESET_ALL}")
    log_decision(f"Action: {Fore.WHITE + Style.BRIGHT}RETRY RETRIEVAL{Style.RESET_ALL}")
    print_section_end(Fore.YELLOW)
    
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
    print_section("📝 ANSWER GENERATION", Fore.GREEN)
    question = state["messages"][0].content
    context = state["messages"][-1].content
    
    context_preview = context[:80].replace("\n", " ") + "..."
    log_step("Synthesizing answer from retrieved context...", Fore.GREEN)
    log_info(f"Context preview: {Fore.WHITE}{context_preview}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}│{Style.RESET_ALL}")

    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm.invoke([{"role": "user", "content": prompt}])
    
    log_success(f"Status: {Fore.WHITE + Style.BRIGHT}ANSWER READY{Style.RESET_ALL}")
    print_section_end(Fore.GREEN)
    return {"messages": [response]}


# ============================================
# Build the Graph
# ============================================

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
# Main Chat Loop
# ============================================

def chat():
    """Interactive chat loop for the RAG agent."""
    print_box(
        f"{Fore.WHITE + Style.BRIGHT}Student Question Answering System\n" +
        f"{Fore.CYAN}Ask questions about your school subjects\n" +
        f"{Fore.YELLOW}Type 'quit' or 'exit' to stop",
        Fore.GREEN,
        width=70
    )
    print()

    # Check if collection exists and has documents
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        points_count = collection_info.points_count
        if points_count == 0:
            print("Warning: The collection is empty. Please run generate_embeddings.py first.")
            print()
    except Exception:
        print("Warning: Could not connect to Qdrant or collection doesn't exist.")
        print("Make sure Qdrant is running (docker-compose up -d) and run generate_embeddings.py")
        print()
        return

    graph = build_graph()

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        print()
        print_box("🚀 RAG WORKFLOW STARTED", Fore.CYAN, width=70)
        print()

        # Run the graph
        final_response = None
        for chunk in graph.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="updates"
        ):
            for node, update in chunk.items():
                if node == "generate_answer" or (node == "generate_query_or_respond" and "tool_calls" not in str(update)):
                    final_response = update["messages"][-1].content

        print()
        print_box("✅ RAG WORKFLOW COMPLETED", Fore.GREEN, width=70)
        
        if final_response:
            print_answer(final_response, width=70)
        else:
            print(f"\n{Fore.RED}⚠ I couldn't generate a response. Please try rephrasing your question.{Style.RESET_ALL}\n")


def single_query(question: str) -> str:
    """Process a single query and return the response."""
    graph = build_graph()

    final_response = None
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates"
    ):
        for node, update in chunk.items():
            if node == "generate_answer" or (node == "generate_query_or_respond" and "tool_calls" not in str(update)):
                final_response = update["messages"][-1].content

    return final_response or "No response generated."


if __name__ == "__main__":
    chat()
