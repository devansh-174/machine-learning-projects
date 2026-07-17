import os
import streamlit as st

from rag import ask_question, create_rag_pipeline

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="GenAI Learning Mentor", layout="wide")

st.title("📚 GenAI Learning Mentor")
st.write("Upload your study material and ask questions from it.")

# -------------------------
# Session State
# -------------------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------
# Cache RAG Pipeline
# -------------------------
@st.cache_resource
def build_rag(pdf_path):
    return create_rag_pipeline(pdf_path)


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("📄 Document Information")

    if st.session_state.doc_info:
        info = st.session_state.doc_info
        st.write("**File:**", info.get("filename", "Unknown"))
        st.write("**Pages:**", info.get("pages", 0))
        st.write("**Chunks:**", info.get("chunks", 0))
        st.write("**Embedding:**", info.get("embedding_model", "Unknown"))
        st.write("**LLM:**", info.get("llm", "Unknown"))

        # Display Pipeline Setup Times inside the sidebar
        if "metrics" in info:
            st.divider()
            st.subheader("⏱️ Setup Times")
            pipeline_metrics = info["metrics"]
            st.write(
                f"**PDF Loading:** {pipeline_metrics.get('pdf_loading', 0.0):.3f}s"
            )
            st.write(
                f"**Chunking:** {pipeline_metrics.get('chunking', 0.0):.3f}s"
            )
            st.write(
                f"**Embedding/FAISS:** {pipeline_metrics.get('embedding_and_faiss', 0.0):.3f}s"
            )
            st.write(
                f"**Total Pipeline Init:** {pipeline_metrics.get('pipeline', 0.0):.3f}s"
            )
    else:
        st.info("Upload a PDF to see details")


# -------------------------
# PDF Upload
# -------------------------
uploaded_file = st.file_uploader("Upload your PDF notes", type=["pdf"])

if uploaded_file:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            os.makedirs("temp", exist_ok=True)
            pdf_path = os.path.join("temp", uploaded_file.name)

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Unpack the 3-element tuple cleanly
            (
                st.session_state.qa_chain,
                st.session_state.retriever,
                st.session_state.doc_info,
            ) = build_rag(pdf_path)
            
            st.session_state.chat_history = []

        st.success("PDF processed successfully!")


# -------------------------
# Conversation History
# -------------------------
if st.session_state.chat_history:
    st.subheader("💬 Conversation")
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**AI:** {chat['answer']}")
        st.divider()


# -------------------------
# Question Input
# -------------------------
question = st.text_input("Ask your question")

if st.button("Get Answer"):
    if st.session_state.qa_chain is None:
        st.warning("Please upload and process a PDF first.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            # Pass explicit chain and retriever parameters down to the execution interface
            response = ask_question(
                st.session_state.qa_chain,
                st.session_state.retriever,
                question,
            )

        answer = response["answer"]
        metrics = response["metrics"]

        # Save Conversation
        st.session_state.chat_history.append(
            {"question": question, "answer": answer}
        )

        # -------------------------
        # Answer
        # -------------------------
        st.subheader("Answer")
        st.write(answer)

        # -------------------------
        # Sources
        # -------------------------
        st.subheader("Sources")
        for source in response["sources"]:
            st.markdown(f"### Page {source['page']}")
            st.text_area(
                label=f"Source - Page {source['page']}",
                value=source["content"],
                height=180,
                disabled=True,
                label_visibility="collapsed",
            )
            st.divider()

        # -------------------------
        # Performance Metrics
        # -------------------------
        st.subheader("📊 Performance Metrics")
        col1, col2 = st.columns(2)

        with col1:
            total_time = metrics.get('total_query', 0.0)
            st.metric("Total Query Time", f"{total_time:.2f} sec")
            
            gen_time = metrics.get('generation_time', 0.0)
            st.metric("Generation Time", f"{gen_time:.2f} sec")
            
            st.metric("Retrieved Chunks", metrics.get("retrieved_chunks", 0))
            st.metric("Context Length (Chars)", metrics.get("context_length", 0))

        with col2:
            st.metric("CPU Usage", f"{metrics.get('cpu_usage', 0.0):.1f}%")
            st.metric("RAM Usage", f"{metrics.get('ram_usage', 0.0):.1f}%")
            st.metric("GPU Utilization", f"{metrics.get('gpu_utilization', 0)}%")
            st.metric("GPU Memory Used", f"{metrics.get('gpu_memory_used', 0.0):.2f} GB")

        # -------------------------
        # Detailed Metrics
        # -------------------------
        with st.expander("📈 Detailed Performance Report"):
            st.json(metrics)