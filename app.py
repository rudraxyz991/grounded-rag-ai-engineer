import os
import time
import logging
import streamlit as st
from pypdf import PdfReader
from groq import Groq, APIError, RateLimitError
import chromadb

# Import configuration and backend helpers
from config import (
    GROQ_MODEL_NAME,
    TOP_K,
    MAX_HISTORY_TURNS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from vector_store import get_embedding_model, CHROMA_DB_PATH, COLLECTION_NAME
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Grounded RAG | Engineering Portfolio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS: MODERN SLATE & TECHNICAL INDUSTRIAL THEME ---
st.markdown("""
    <style>
    /* Remove default Streamlit top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Clean, professional typography for headers */
    h1, h2, h3 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    /* Style the Sidebar to look like a clean control console */
    [data-testid="stSidebar"] {
        background-color: #1A1D20;
        border-right: 1px solid #2D3136;
    }
    [data-testid="stSidebar"] * {
        color: #E0E4E8 !important;
    }

    /* Custom Citation Expander Cards */
    .streamlit-expanderHeader {
        background-color: #1E2227 !important;
        border-radius: 6px !important;
        border: 1px solid #2D3136 !important;
        font-weight: 600 !important;
    }
    
    /* Telemetry Caption Pills */
    .telemetry-pill {
        display: inline-block;
        background-color: #23272D;
        border: 1px solid #363B42;
        border-radius: 4px;
        padding: 4px 10px;
        font-family: 'monospace';
        font-size: 0.80rem;
        color: #9AA4B2;
        margin-top: 6px;
        margin-bottom: 12px;
    }

    /* Clean Chat Input Box */
    [data-testid="stChatInput"] {
        border-radius: 8px !important;
        border: 1px solid #3E454D !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- CACHED RESOURCE INITIALIZATION ---
@st.cache_resource
def load_backend_resources():
    """Caches persistent client and models so they survive Streamlit UI re-renders."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    embed_model = get_embedding_model()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return collection, embed_model, groq_client

collection, embed_model, groq_client = load_backend_resources()

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "token_audit" not in st.session_state:
    st.session_state.token_audit = {"prompt": 0, "completion": 0, "total": 0}
if "active_doc" not in st.session_state:
    st.session_state.active_doc = "Default Sample PDF"

# --- HELPER: DYNAMIC PDF PROCESSING ---
def process_uploaded_pdf(uploaded_file):
    """Chunks, embeds, and indexes custom PDFs on the fly into persistent ChromaDB."""
    with st.spinner(f"Ingesting '{uploaded_file.name}' — chunking & generating semantic embeddings..."):
        reader = PdfReader(uploaded_file)
        raw_pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                raw_pages.append({"text": text, "page": i})

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        documents, metadatas, ids = [], [], []
        chunk_counter = 0

        for page_data in raw_pages:
            chunks = text_splitter.split_text(page_data["text"])
            for chunk in chunks:
                chunk_counter += 1
                doc_id = f"{uploaded_file.name}_p{page_data['page']}_c{chunk_counter}"
                documents.append(chunk)
                metadatas.append({
                    "source_doc": uploaded_file.name,
                    "page_number": page_data["page"]
                })
                ids.append(doc_id)

        embeddings = embed_model.encode(documents).tolist()
        collection.upsert(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        st.session_state.active_doc = uploaded_file.name
        st.toast(f"Successfully indexed {chunk_counter} chunks from {uploaded_file.name}", icon="✅")

# --- RAG RETRIEVAL & GENERATION LOGIC ---
def retrieve_and_generate(user_query: str):
    """Executes search and LLM inference with latency profiling and error handling."""
    t0_start = time.time()

    # Contextual Query Augmentation for follow-up turns
    search_query = user_query
    if len(st.session_state.messages) >= 2:
        last_user_msg = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "")
        search_query = f"{last_user_msg} {user_query}"

    # Vector Search
    t0_search = time.time()
    query_vector = embed_model.encode([search_query]).tolist()
    raw_results = collection.query(
        query_embeddings=query_vector,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    search_latency = time.time() - t0_search

    chunks = []
    for i in range(len(raw_results["documents"][0])):
        chunks.append({
            "text": raw_results["documents"][0][i],
            "metadata": raw_results["metadatas"][0][i],
            "distance": raw_results["distances"][0][i],
        })

    # Prompt Assembly
    context_blocks = [
        f"--- Chunk {i+1} [Source: {c['metadata']['source_doc']} | Page: {c['metadata']['page_number']}] ---\n{c['text']}"
        for i, c in enumerate(chunks)
    ]
    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an expert technical assistant. Answer the user's question explicitly and ONLY using the "
        "provided context blocks below. Do NOT use outside knowledge. If the answer is not in the context, "
        "state clearly that it is not present in the document.\n\n"
        "CITATION RULE: Include inline citations referencing the source and page number, e.g., [Source: doc.pdf | Page: 4]."
    )

    llm_messages = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages[-(MAX_HISTORY_TURNS * 2):]:
        llm_messages.append({"role": msg["role"], "content": msg["content"]})
    
    llm_messages.append({
        "role": "user",
        "content": f"CONTEXT BLOCKS:\n{context_str}\n\nUSER QUESTION: {user_query}"
    })

    # Groq Generation with Graceful Failure Handling
    try:
        t0_llm = time.time()
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=llm_messages,
            temperature=0.2,
        )
        llm_latency = time.time() - t0_llm
        total_latency = time.time() - t0_start

        answer = response.choices[0].message.content
        usage = {
            "prompt": response.usage.prompt_tokens,
            "completion": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
        }

        st.session_state.token_audit["prompt"] += usage["prompt"]
        st.session_state.token_audit["completion"] += usage["completion"]
        st.session_state.token_audit["total"] += usage["total"]

        metrics = {
            "search_latency": search_latency,
            "llm_latency": llm_latency,
            "total_latency": total_latency,
            "usage": usage,
        }
        return answer, chunks, metrics, None

    except (APIError, RateLimitError) as api_err:
        logger.error(f"Groq API Failure: {str(api_err)}")
        error_msg = (
            "⚠️ **LLM Inference Unavailable:** The Groq API encountered a rate limit or network issue. "
            "Vector retrieval succeeded, but the response could not be generated. Please retry in a few seconds."
        )
        return error_msg, chunks, None, error_msg
    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        return "⚠️ An unexpected system error occurred. Please verify your environment variables.", [], None, str(e)


# ==============================================================================
# --- SIDEBAR: CONTROLS & TELEMETRY CONSOLE ---
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Engine Console")
    st.markdown("---")
    
    uploaded_pdf = st.file_uploader(
        "Ingest Custom PDF Document",
        type=["pdf"],
        help="Upload any PDF to dynamically chunk, embed, and index into ChromaDB."
    )
    if uploaded_pdf is not None:
        if uploaded_pdf.name != st.session_state.active_doc:
            process_uploaded_pdf(uploaded_pdf)

    st.markdown("### 🪙 Production Cost Auditing")
    audit = st.session_state.token_audit
    st.metric(label="Total Session Tokens", value=f"{audit['total']:,}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Prompt (In)", value=f"{audit['prompt']:,}")
    with col2:
        st.metric(label="Output (Out)", value=f"{audit['completion']:,}")

    st.markdown("---")
    st.markdown(f"**Active Index:** `{st.session_state.active_doc}`")
    st.markdown(f"**LLM Backend:** `{GROQ_MODEL_NAME}`")
    
    if st.button("🗑️ Clear Active Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.token_audit = {"prompt": 0, "completion": 0, "total": 0}
        st.rerun()

# ==============================================================================
# --- MAIN CHAT INTERFACE ---
# ==============================================================================
st.title("Grounded RAG Architecture | Technical Preview")
st.markdown("An enterprise RAG system featuring inline citation grounding, L2 distance sorting, and real-time inference latency auditing.")
st.markdown("---")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "extra" in msg:
            extra = msg["extra"]
            if extra.get("chunks"):
                with st.expander("🔍 Verified Source Citations & L2 Distance Metrics"):
                    for i, c in enumerate(extra["chunks"], 1):
                        meta = c["metadata"]
                        st.markdown(f"**Source #{i}: `{meta['source_doc']}` (Page {meta['page_number']}) — L2 Dist: `{c['distance']:.4f}`**")
                        st.info(c["text"])
            
            if extra.get("metrics"):
                m = extra["metrics"]
                u = m["usage"]
                st.markdown(
                    f'<div class="telemetry-pill">'
                    f'LATENCY: {m["total_latency"]:.2f}s (Search: {m["search_latency"]*1000:.0f}ms | Groq: {m["llm_latency"]:.2f}s) • '
                    f'TOKENS: {u["total"]} (In: {u["prompt"]} / Out: {u["completion"]})'
                    f'</div>',
                    unsafe_allow_html=True
                )

# Chat Input Box
if user_input := st.chat_input("Ask a technical question about the indexed document..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Executing semantic similarity search & generating grounded response..."):
            answer, chunks, metrics, err = retrieve_and_generate(user_input)
            
            st.markdown(answer)
            
            if chunks:
                with st.expander("🔍 Verified Source Citations & L2 Distance Metrics"):
                    for i, c in enumerate(chunks, 1):
                        meta = c["metadata"]
                        st.markdown(f"**Source #{i}: `{meta['source_doc']}` (Page {meta['page_number']}) — L2 Dist: `{c['distance']:.4f}`**")
                        st.info(c["text"])
            
            if metrics:
                u = metrics["usage"]
                st.markdown(
                    f'<div class="telemetry-pill">'
                    f'LATENCY: {metrics["total_latency"]:.2f}s (Search: {metrics["search_latency"]*1000:.0f}ms | Groq: {metrics["llm_latency"]:.2f}s) • '
                    f'TOKENS: {u["total"]} (In: {u["prompt"]} / Out: {u["completion"]})'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "extra": {"chunks": chunks, "metrics": metrics, "error": err}
        })