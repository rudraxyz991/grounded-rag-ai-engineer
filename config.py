import os
from dotenv import load_dotenv
from groq import Groq

# 1. Load local environment variables from .env if present (local development)
load_dotenv()

# 2. Try reading GROQ_API_KEY from standard environment / .env
api_key = os.getenv("GROQ_API_KEY")

# 3. If not found, fallback to Streamlit Cloud Secrets (cloud deployment)
if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        pass

# 4. Safety check: fail loudly only if BOTH sources fail
if not api_key:
    raise ValueError(
        "GROQ_API_KEY is missing. Please check your .env file or Streamlit Secrets setup."
    )

# --- API & MODEL CONFIGURATION ---
MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT = 30.0  # API call timeout in seconds

# Initialize the Groq Client
client = Groq(
    api_key=api_key,
    timeout=REQUEST_TIMEOUT
)

# --- DAY 2: LANGCHAIN CHUNKING PARAMETERS ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# --- DAY 3: EMBEDDINGS & CHROMADB PARAMETERS ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"
EMBED_BATCH_SIZE = 32  # Batching parameter for AI Engineer evaluation

# --- DAY 4: RETRIEVAL & RESILIENCY PARAMETERS ---
TOP_K = 3  # Easy flag to sweep during retrieval evaluation
MAX_RETRIES = 3  # Exponential backoff retry limit
INITIAL_RETRY_DELAY = 1.0  # Base delay in seconds for backoff

# --- DAY 5: CONVERSATIONAL MEMORY & PRODUCTION CONTROLS ---
MAX_HISTORY_TURNS = 3  # Number of past conversation turns (user + assistant pairs) to retain
ENABLE_QUERY_CACHE = True  # Flag to toggle in-session semantic/exact caching