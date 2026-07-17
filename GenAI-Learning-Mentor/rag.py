from functools import lru_cache
import os

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from monitor import PerformanceMonitor

# =====================================================
# Configuration Constants
# =====================================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "qwen3.5:9b"
LLM_DISPLAY_NAME = "Qwen3.5:9B (Ollama)"

# Initialize performance monitor global instance
monitor = PerformanceMonitor()


# =====================================================
# Load PDF
# =====================================================
def load_pdf(file_path):
    monitor.start("pdf_loading")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    finally:
        monitor.stop("pdf_loading")
    return documents


# =====================================================
# Split Documents
# =====================================================
def split_documents(documents):
    monitor.start("chunking")
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
    finally:
        monitor.stop("chunking")
    return chunks


# =====================================================
# Embedding Model
# =====================================================
@lru_cache(maxsize=1)
def get_embeddings():
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cuda"}
    )
    return embeddings


# =====================================================
# Create Vector Database
# =====================================================
def create_vectorstore(chunks):
    monitor.start("embedding_and_faiss")
    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
    finally:
        monitor.stop("embedding_and_faiss")
    return vectorstore


# =====================================================
# Create Conversational RAG Chain
# =====================================================
def create_qa_chain(vectorstore):
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0.3)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    # -----------------------------
    # Dual Retrievers
    # -----------------------------
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )

    summary_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 30, "lambda_mult": 0.5},
    )

    # -----------------------------
    # Strict Custom RAG Prompt
    # -----------------------------
    prompt_template = """
You are an AI Learning Mentor.

Answer ONLY using the provided context.

Before answering, first determine whether the provided context actually contains the answer.

Rules:
1. Use ONLY the provided context.
2. Never use outside knowledge.
3. If the context does not directly answer the user's question, reply with EXACTLY:
"This information is not available in the uploaded document."
4. Do not summarize unrelated sections of the document.
5. Do not answer a different question from the one asked.
6. Explain concepts clearly when the answer exists in the context.
7. Keep mathematical notation and technical terms accurate.

Context:
{context}

Question:
{question}

Answer:
"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Build the default base chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=base_retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )

    # Build the dedicated summary chain sharing the same conversational memory tracking
    summary_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=summary_retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )

    # Dictionary containing both pre-configured chains
    chains_container = {
        "base_chain": qa_chain,
        "summary_chain": summary_chain
    }

    # Preserves Option A interface structure compatibility for app.py
    return chains_container, base_retriever


# =====================================================
# Ask Question
# =====================================================
def ask_question(qa_chain, retriever, question):
    monitor.start("total_query")
    monitor.start("generation")

    question_lower = question.lower()
    summary_keywords = [
        "summary",
        "summarize",
        "summarise",
        "overview",
        "summarize the paper",
        "summarise the paper",
        "summarize this paper",
        "paper summary",
        "document summary",
        "whole paper",
        "entire paper",
        "full paper",
        "complete paper",
        "whole document",
        "entire document",
        "complete summary",
    ]
    is_summary = any(
        keyword in question_lower for keyword in summary_keywords
    )

    try:
        # Dynamic routing between high-k or low-k chains
        if is_summary:
            active_chain = qa_chain["summary_chain"]
        else:
            active_chain = qa_chain["base_chain"]

        # Resolved conflict: Let ConversationBufferMemory autonomously manage chat_history
        response = active_chain.invoke({"question": question})
    finally:
        monitor.stop("generation")
        monitor.stop("total_query")

    sources = []
    for doc in response["source_documents"]:
        page = doc.metadata.get("page")
        page = (page + 1) if page is not None else "Unknown"
        sources.append({"page": page, "content": doc.page_content})

    metrics = monitor.collect_metrics(
        question=question,
        retrieved_chunks=len(response["source_documents"]),
        context_length=sum(
            len(doc.page_content) for doc in response["source_documents"]
        ),
        answer=response["answer"],
        embedding_model=EMBEDDING_MODEL,
        llm=LLM_DISPLAY_NAME,
    )

    return {
        "answer": response["answer"],
        "sources": sources,
        "metrics": metrics,
    }


# =====================================================
# Complete RAG Pipeline
# =====================================================
def create_rag_pipeline(pdf_path):
    monitor.reset()
    monitor.start("pipeline")

    print("1. Loading PDF")
    documents = load_pdf(pdf_path)

    print("2. Splitting")
    chunks = split_documents(documents)

    print("3. Creating Vector Store")
    vectorstore = create_vectorstore(chunks)

    print("4. Creating QA Chain")
    chains_container, retriever = create_qa_chain(vectorstore)

    print("5. Done")
    monitor.stop("pipeline")

    info = {
        "filename": os.path.basename(pdf_path),
        "pages": len(documents),
        "chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "llm": LLM_DISPLAY_NAME,
        "metrics": monitor.collect_pipeline_metrics(),
    }

    return chains_container, retriever, info