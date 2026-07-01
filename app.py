"""
Medical RAG Assistant
Retrieval-Augmented Generation for medical Q&A
Original coursework: Llama 3.1 8B + ChromaDB + LangChain
Demo version: Claude Haiku as the generation backend
"""


import os
import streamlit as st
import anthropic

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide"
)

# ── Try to import RAG dependencies ───────────────────────────────────────────

try:
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ── Vector store (session-scoped) ────────────────────────────────────────────

def build_vectorstore(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Load PDF, chunk, embed, and store in ChromaDB."""
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)


def retrieve_context(vectorstore, query: str, k: int = 4) -> str:
    """Retrieve top-k relevant chunks for a query."""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join([d.page_content for d in docs])


def generate_answer(query: str, context: str, temperature: float,
                    top_p: float, max_tokens: int) -> str:
    """Call Claude Haiku with the retrieved context."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "_ANTHROPIC_API_KEY not set — cannot generate answer._"

    client = anthropic.Anthropic(api_key=api_key)

    system = """You are a medical information assistant. Answer the user's question
using ONLY the provided context from the medical reference manual.
Be precise, structured, and clinically accurate. If the context does not
contain enough information to answer fully, say so explicitly.
Do not add information not present in the context."""

    user_msg = f"""Context from medical reference:
{context}

Question: {query}

Provide a clear, structured medical answer based solely on the context above."""

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user_msg}]
    )
    return msg.content[0].text.strip()


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🏥 Medical RAG Assistant")
st.caption(
    "Retrieval-Augmented Generation for medical Q&A · "
    "Coursework pipeline (Llama 3.1 8B + ChromaDB) adapted for deployment with Claude Haiku"
)

# Sidebar
st.sidebar.header("Configuration")

# PDF upload
uploaded_pdf = st.sidebar.file_uploader(
    "Upload a medical reference PDF",
    type="pdf",
    help="Upload the Merck Manual or any medical reference PDF. "
         "The file is not stored — processing happens in memory."
)

st.sidebar.markdown("---")
st.sidebar.subheader("RAG Parameters")

chunk_size = st.sidebar.slider(
    "Chunk size (tokens)", 500, 2000, 1000, 100,
    help="Size of text segments. Larger = more context per chunk, fewer chunks."
)
chunk_overlap = st.sidebar.slider(
    "Chunk overlap", 0, 400, 150, 50,
    help="Overlap between chunks to preserve context at boundaries."
)
top_k = st.sidebar.slider(
    "Retrieved chunks (k)", 2, 8, 4,
    help="Number of relevant chunks retrieved per query."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Generation Parameters")
temperature = st.sidebar.slider(
    "Temperature", 0.0, 1.0, 0.0, 0.05,
    help="0 = deterministic, 1 = most creative. "
         "Coursework tested 5 combinations: 0, 0.5, 0.7, 0.9, and 0 + repeat_penalty."
)
top_p = st.sidebar.slider("Top-p", 0.5, 1.0, 1.0, 0.05)
max_tokens = st.sidebar.slider("Max tokens", 256, 1024, 512, 64)

# Main area
if not RAG_AVAILABLE:
    st.warning(
        "RAG dependencies not installed. Run: "
        "`pip install langchain langchain-community langchain-chroma "
        "sentence-transformers chromadb pymupdf`"
    )

tab1, tab2 = st.tabs(["Ask a Question", "About This Project"])

with tab1:
    if uploaded_pdf is None:
        st.info(
            "**Upload a medical reference PDF** in the sidebar to get started.\n\n"
            "The Merck Manual was used in the original coursework. "
            "Any medical reference PDF will work."
        )

        st.markdown("---")
        st.subheader("Example questions from the coursework evaluation")
        examples = [
            "What is the protocol for managing sepsis in a critical care unit?",
            "What are the common symptoms and treatments for pulmonary embolism?",
            "What are the first-line options for managing rheumatoid arthritis?",
            "What are the diagnostic steps for suspected endocrine disorders?",
            "Can you provide the trade names of medications used for treating hypertension?",
        ]
        for q in examples:
            st.markdown(f"- *{q}*")

    else:
        # Build vectorstore
        if "vectorstore" not in st.session_state or \
           st.session_state.get("pdf_name") != uploaded_pdf.name or \
           st.session_state.get("chunk_size") != chunk_size or \
           st.session_state.get("chunk_overlap") != chunk_overlap:

            import tempfile, os as _os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name

            with st.spinner(f"Building vector index from {uploaded_pdf.name}..."):
                try:
                    vs, n_chunks = build_vectorstore(tmp_path, chunk_size, chunk_overlap)
                    st.session_state["vectorstore"] = vs
                    st.session_state["n_chunks"] = n_chunks
                    st.session_state["pdf_name"] = uploaded_pdf.name
                    st.session_state["chunk_size"] = chunk_size
                    st.session_state["chunk_overlap"] = chunk_overlap
                    _os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                    st.stop()

        vs = st.session_state["vectorstore"]
        n_chunks = st.session_state["n_chunks"]
        st.success(f"✅ Index ready — {n_chunks:,} chunks from **{uploaded_pdf.name}**")

        query = st.text_input(
            "Ask a medical question",
            placeholder="What is the protocol for managing sepsis in a critical care unit?"
        )

        if st.button("Ask", type="primary") and query:
            col1, col2 = st.columns([1, 1])

            with st.spinner("Retrieving relevant passages..."):
                context = retrieve_context(vs, query, k=top_k)

            with col1:
                st.subheader("📄 Retrieved Context")
                st.caption(f"Top {top_k} chunks · chunk_size={chunk_size} · overlap={chunk_overlap}")
                with st.expander("View retrieved passages", expanded=False):
                    for i, chunk in enumerate(context.split("\n\n---\n\n"), 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                        st.markdown("---")

            with col2:
                st.subheader("🤖 Generated Answer")
                st.caption(
                    f"Claude Haiku · temperature={temperature} · "
                    f"top_p={top_p} · max_tokens={max_tokens}"
                )
                with st.spinner("Generating answer..."):
                    answer = generate_answer(query, context, temperature, top_p, max_tokens)
                st.markdown(answer)

                st.info(
                    "**Parameter note:** The original coursework tested 5 parameter "
                    "combinations using Llama 3.1 8B. Try adjusting temperature in the "
                    "sidebar to see how it affects response style — the same effect "
                    "documented in the notebook applies here."
                )

with tab2:
    st.header("About This Project")
    st.markdown("""
### Coursework Context

This is the deployed version of my NLP/RAG coursework project, originally built in
Google Colab using **Meta Llama 3.1 8B Instruct** (Q5_K_M quantized GGUF) running
on a Tesla T4 GPU, with **ChromaDB** as the vector store and **LangChain** as the
orchestration framework.

The deployed demo replaces Llama with **Claude Haiku** as the generation backend
for accessibility (no GPU required), while keeping the full RAG pipeline intact.

### Technical Stack

| Component | Coursework | This Demo |
|-----------|-----------|-----------|
| Document loading | PyMuPDF | PyMuPDF |
| Chunking | RecursiveCharacterTextSplitter | RecursiveCharacterTextSplitter |
| Embeddings | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| Vector store | ChromaDB | ChromaDB |
| Generation | Llama 3.1 8B (quantized) | Claude Haiku |
| Infrastructure | Google Colab + Tesla T4 | Streamlit Cloud |

### What Was Evaluated

The coursework evaluated **5 parameter combinations** across 5 clinical questions:

| Combination | Settings | Finding |
|-------------|----------|---------|
| 1 (baseline) | temperature=0 | Deterministic, structured, shortest |
| 2 | temperature=0.7 | Longer, more detailed, added context |
| 3 | temperature=0.5, top_p=0.85, top_k=40 | Shortest, most focused |
| 4 | temperature=0, repeat_penalty=1.3 | Similar to baseline, less repetition |
| 5 | temperature=0.9, seed=99 | Most varied, added novel sections |

Key finding: **temperature=0 with parameter tuning (combinations 3 and 4)**
produced the most clinically reliable responses. Higher temperatures surfaced
more detail but risked adding information not directly in the retrieved context.

### Connection to research_assistant

This coursework project was the direct predecessor of the
[research_assistant](https://github.com/xmashaxxx/research_assistant) agentic
pipeline. Key design decisions carried forward:

- **RAG over LLM alone** — grounding outputs in retrieved documents reduces hallucination
- **Chunking strategy matters** — chunk size and overlap significantly affect retrieval quality
- **Separation of retrieval and generation** — allows independent optimization of each stage
- **Schema-driven extraction** — the tool-use pattern in research_assistant evolved from
  lessons learned about structured output reliability here
    """)
