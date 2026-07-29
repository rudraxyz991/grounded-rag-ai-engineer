import os
import time
import logging
from typing import List, Dict, Any
from groq import Groq, APIError, RateLimitError
from config import (
    GROQ_MODEL_NAME,
    TOP_K,
    MAX_RETRIES,
    INITIAL_RETRY_DELAY,
)
from vector_store import get_embedding_model, CHROMA_DB_PATH, COLLECTION_NAME
import chromadb

# Configure professional logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_groq_client() -> Groq:
    """Initializes the Groq client. Requires GROQ_API_KEY in environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. Please add it to your .env or environment."
        )
    return Groq(api_key=api_key)


def retrieve_context_with_latency(
    collection, embed_model, query: str, top_k: int = TOP_K
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Executes similarity search and logs separated latency metrics
    for Query Embedding vs. Vector DB Search.
    """
    latency_metrics = {}

    # 1. Track Query Embedding Time
    t0_embed = time.time()
    query_embedding = embed_model.encode([query]).tolist()
    latency_metrics["embed_time"] = time.time() - t0_embed

    # 2. Track Vector Store Retrieval Time
    t0_search = time.time()
    raw_results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    latency_metrics["search_time"] = time.time() - t0_search

    # Format into clean dictionaries with metadata lineage
    retrieved_chunks = []
    for i in range(len(raw_results["documents"][0])):
        retrieved_chunks.append(
            {
                "text": raw_results["documents"][0][i],
                "metadata": raw_results["metadatas"][0][i],
                "distance": raw_results["distances"][0][i],
            }
        )

    logger.info(
        f"Retrieval Complete | Embed Time: {latency_metrics['embed_time']*1000:.2f}ms | Search Time: {latency_metrics['search_time']*1000:.2f}ms"
    )
    return retrieved_chunks, latency_metrics


def build_grounded_prompt(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Constructs a strict prompt with source attribution instructions
    and injected context chunks.
    """
    # Build formatted context block with exact document & page lineage
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        header = f"[Source: {meta['source_doc']} | Page: {meta['page_number']}]"
        context_blocks.append(f"--- Chunk {i} {header} ---\n{chunk['text']}")

    context_str = "\n\n".join(context_blocks)

    system_instruction = (
        "You are an expert technical assistant. Answer the user's question explicitly and ONLY "
        "using the provided context blocks below. Do NOT use outside knowledge or speculate. "
        "If the answer cannot be found in the context, state clearly that the information is not present.\n\n"
        "CITATION RULE: Whenever you assert a fact from the context, include an inline citation "
        "referencing the exact source and page, for example: [Source: sample.pdf | Page: 31]."
    )

    user_message = f"CONTEXT BLOCKS:\n{context_str}\n\nUSER QUESTION: {query}"

    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message},
    ]


def call_llm_with_backoff(client: Groq, messages: List[Dict[str, str]]) -> tuple[str, float]:
    """
    AI Engineer Layer: Executes Groq API call with exponential backoff
    for rate limits/failures and returns LLM latency.
    """
    delay = INITIAL_RETRY_DELAY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0_llm = time.time()
            response = client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=messages,
                temperature=0.2,  # Low temperature for factual consistency
            )
            llm_time = time.time() - t0_llm
            return response.choices[0].message.content, llm_time

        except (APIError, RateLimitError) as e:
            logger.warning(
                f"Groq API Error on attempt {attempt}/{MAX_RETRIES}: {str(e)}"
            )
            if attempt == MAX_RETRIES:
                logger.error("Max retries reached. Raising exception.")
                raise e
            logger.info(f"Backing off for {delay:.2f} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff factor of 2


def run_rag_query(collection, embed_model, groq_client, user_query: str):
    """Orchestrates end-to-end RAG execution, prints cited response, and logs all latency metrics."""
    logger.info(f"Processing End-to-End Query: '{user_query}'")
    total_start = time.time()

    # Step 1: Retrieve Context & Measure Retrieval Latencies
    chunks, latencies = retrieve_context_with_latency(
        collection, embed_model, user_query, top_k=TOP_K
    )

    # Step 2: Build Grounded Prompt
    prompt_messages = build_grounded_prompt(user_query, chunks)

    # Step 3: Call LLM with Exponential Backoff
    answer, llm_time = call_llm_with_backoff(groq_client, prompt_messages)
    latencies["llm_time"] = llm_time
    latencies["total_time"] = time.time() - total_start

    # Step 4: Display Grounded Answer & Citation Footprint
    print("\n================== GROUNDED RAG ANSWER ==================")
    print(f"Question: {user_query}\n")
    print(answer)
    print("---------------------------------------------------------")
    print("Retrieved Sources Displayed to LLM:")
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        print(
            f"  * Chunk #{i}: {meta['source_doc']} (Page {meta['page_number']}) -> Distance: {c['distance']:.4f}"
        )
    print("=========================================================\n")

    # Step 5: Professional Latency Summary
    print("--- AI ENGINEER METRIC: LATENCY PROFILE ---")
    print(f"  * Query Embedding Time : {latencies['embed_time']*1000:.2f} ms")
    print(f"  * Vector DB Search Time: {latencies['search_time']*1000:.2f} ms")
    print(f"  * Groq LLM Generation  : {latencies['llm_time']:.4f} seconds")
    print(f"  * TOTAL End-to-End     : {latencies['total_time']:.4f} seconds")
    print("-------------------------------------------\n")


if __name__ == "__main__":
    logger.info("Initializing persistent ChromaDB client and local embedding model...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    rag_collection = client.get_collection(name=COLLECTION_NAME)
    embedding_model = get_embedding_model()
    groq_api_client = get_groq_client()

    # ---> REPLACE THE QUESTION HERE <---
    test_question = "What is mach band effect?"
    
    run_rag_query(rag_collection, embedding_model, groq_api_client, test_question)