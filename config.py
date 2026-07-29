from dotenv import dotenv_values
from groq import Groq

# 1. Load environment variables securely into a dictionary (does not alter os.environ)
env_vars = dotenv_values(".env")
api_key = env_vars.get("GROQ_API_KEY")

# Safety check to fail loudly if the key is missing
if not api_key:
    raise ValueError("GROQ_API_KEY is missing from the .env file. Please check your setup.")


MODEL_NAME = "llama-3.3-70b-versatile" 
REQUEST_TIMEOUT = 30.0  # API call timeout in seconds

# 3. Initialize the Groq Client
client = Groq(
    api_key=api_key,
    timeout=REQUEST_TIMEOUT
)
# Day 2: LangChain Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Day 3: Embeddings & ChromaDB Parameters
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_collection"
EMBED_BATCH_SIZE = 32  # Batching parameter for AI Engineer evaluation

# Day 4: Retrieval, LLM & Resiliency Parameters
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"  # High-performance Llama 3.3 model on Groq
TOP_K = 3  # Easy flag to sweep during retrieval evaluation
MAX_RETRIES = 3  # Exponential backoff retry limit
INITIAL_RETRY_DELAY = 1.0  # Base delay in seconds for backoff

# Day 5: Conversational Memory & Production Controls
MAX_HISTORY_TURNS = 3  # Number of past conversation turns (user + assistant pairs) to retain
ENABLE_QUERY_CACHE = True  # Flag to toggle in-session semantic/exact caching

import os
from dotenv import load_dotenv

load_dotenv()

# 1. Try reading from standard environment variables (.env / os.environ)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 2. If not found, fallback to Streamlit Secrets (for Cloud deployment)
if not GROQ_API_KEY:
    try:
        import streamlit as st
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        pass

# 3. Raise error only if both sources fail
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please check your .env file or Streamlit Secrets setup.")

import os
from dotenv import load_dotenv

load_dotenv()

# 1. Try reading from local .env / environment
api_key = os.getenv("GROQ_API_KEY")

# 2. If None, fallback to Streamlit Cloud Secrets
if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        pass

# 3. Safety check to fail loudly if still missing
if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Please check your .env file or Streamlit Secrets setup.")