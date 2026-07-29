import os
import time
import logging
from typing import List, Dict, Any, Tuple
from groq import Groq, APIError, RateLimitError
import chromadb
from config import (
    GROQ_MODEL_NAME,
    TOP_K,
    MAX_RETRIES,
    INITIAL_RETRY_DELAY,
    MAX_HISTORY_TURNS,
    ENABLE_QUERY_CACHE,
)
from vector_store import get_embedding_model, CHROMA_DB_PATH, COLLECTION_NAME

# Configure professional logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGChatSession:
    """
    Stateful RAG Session Manager that tracks conversation history,
    token consumption, and in-session query caching.
    """

    def __init__(self, collection, embed_model, groq_client: Groq):
        self.collection = collection
        self.embed_model = embed_model
        self.client = groq_client

        # Stateful session memory
        self.history: List[Dict[str, str]] = []  # Stores {"role": ..., "content": ...}
        self.query_cache: Dict[str, Dict[str, Any]] = {}  # Normalized query -> cached response

        # Production Cost Tracking (Running Token Totals)
        self.token_audit = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _normalize_query(self, query: str) -> str:
        """Lowercases and strips whitespace for deterministic cache matching."""
        return query.strip().lower()

    def _augment_query_for_search(self, current_query: str) -> str:
        """
        AI Engineer Layer: Solves pronoun ambiguity in vector search by
        prepending the last user question if conversation history exists.
        """
        if not self.history:
            return current_query

        # Find the most recent user turn in history
        for msg in reversed(self.history):
            if msg["role"] == "user":
                logger.info("Augmenting search query with previous conversation context.")
                return f"{msg['content']} {current_query}"
        return current_query

    def retrieve_context(
        self, search_query: str, top_k: int = TOP_K
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """Executes embedding and ChromaDB similarity search with latency profiling."""
        t0_embed = time.time()
        query_embedding = self.embed_model.encode([search_query]).tolist()
        embed_time = time.time() - t0_embed

        t0_search = time.time()
        raw_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        search_time = time.time() - t0_search

        retrieved_chunks = []
        for i in range(len(raw_results["documents"][0])):
            retrieved_chunks.append(
                {
                    "text": raw_results["documents"][0][i],
                    "metadata": raw_results["metadatas"][0][i],
                    "distance": raw_results["distances"][0][i],
                }
            )
        return retrieved_chunks, embed_time, search_time

    def build_prompt_with_history(
        self, current_query: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Constructs prompt messages injecting:
        1. System instructions + Grounding rules
        2. Formatted conversation history (last N turns)
        3. Retrieved document context blocks
        4. Current user query
        """
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            header = f"[Source: {meta['source_doc']} | Page: {meta['page_number']}]"
            context_blocks.append(f"--- Chunk {i} {header} ---\n{chunk['text']}")

        context_str = "\n\n".join(context_blocks)

        system_instruction = (
            "You are an expert technical assistant participating in a multi-turn conversation. "
            "Answer the user's latest question explicitly and ONLY using the provided context blocks below. "
            "Use the conversation history only to understand follow-up questions or pronouns. "
            "Do NOT use outside knowledge. If the answer is not in the context, state clearly that it is not present.\n\n"
            "CITATION RULE: Whenever you assert a fact from the context, include an inline citation "
            "referencing the exact source and page, for example: [Source: sample.pdf | Page: 31]."
        )

        messages = [{"role": "system", "content": system_instruction}]

        # Inject conversation history (capped at last MAX_HISTORY_TURNS pairs)
        max_messages = MAX_HISTORY_TURNS * 2
        for turn in self.history[-max_messages:]:
            messages.append(turn)

        # Append latest user turn with freshly retrieved context
        user_content = f"CONTEXT BLOCKS:\n{context_str}\n\nLATEST USER QUESTION: {current_query}"
        messages.append({"role": "user", "content": user_content})

        return messages

    def call_llm_with_backoff(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, float, Dict[str, int]]:
        """Executes Groq API call with retry backoff and captures usage token counts."""
        delay = INITIAL_RETRY_DELAY

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0_llm = time.time()
                response = self.client.chat.completions.create(
                    model=GROQ_MODEL_NAME,
                    messages=messages,
                    temperature=0.2,
                )
                llm_time = time.time() - t0_llm

                # Extract token accounting from API response payload
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                return response.choices[0].message.content, llm_time, usage

            except (APIError, RateLimitError) as e:
                logger.warning(
                    f"Groq API Error on attempt {attempt}/{MAX_RETRIES}: {str(e)}"
                )
                if attempt == MAX_RETRIES:
                    raise e
                time.sleep(delay)
                delay *= 2

    def ask(self, user_query: str):
        """Processes a single conversational turn through the RAG pipeline."""
        total_start = time.time()
        norm_query = self._normalize_query(user_query)

        # --- OPTIMIZATION: Check In-Session Query Cache ---
        if ENABLE_QUERY_CACHE and norm_query in self.query_cache:
            logger.info("CACHE HIT! Serving response from in-session cache.")
            cached = self.query_cache[norm_query]
            self._display_response(
                user_query,
                cached["answer"],
                cached["chunks"],
                {"embed_time": 0.0, "search_time": 0.0, "llm_time": 0.0, "total_time": 0.0},
                cached["usage"],
                cache_hit=True,
            )
            return

        # Step 1: Contextual Query Augmentation for Search
        search_query = self._augment_query_for_search(user_query)

        # Step 2: Retrieve Chunks
        chunks, embed_time, search_time = self.retrieve_context(search_query)

        # Step 3: Build Multi-Turn Prompt
        prompt_messages = self.build_prompt_with_history(user_query, chunks)

        # Step 4: Generate Grounded Answer
        answer, llm_time, usage = self.call_llm_with_backoff(prompt_messages)
        total_time = time.time() - total_start

        # Step 5: Update Running Token Audit
        self.token_audit["prompt_tokens"] += usage["prompt_tokens"]
        self.token_audit["completion_tokens"] += usage["completion_tokens"]
        self.token_audit["total_tokens"] += usage["total_tokens"]

        # Step 6: Update Conversation History (store original user query, not raw context blocks)
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": answer})

        # Step 7: Update Query Cache
        if ENABLE_QUERY_CACHE:
            self.query_cache[norm_query] = {
                "answer": answer,
                "chunks": chunks,
                "usage": usage,
            }

        # Step 8: Display Formatted Output
        metrics = {
            "embed_time": embed_time,
            "search_time": search_time,
            "llm_time": llm_time,
            "total_time": total_time,
        }
        self._display_response(user_query, answer, chunks, metrics, usage, cache_hit=False)

    def _display_response(
        self,
        query: str,
        answer: str,
        chunks: List[Dict[str, Any]],
        latencies: Dict[str, float],
        usage: Dict[str, int],
        cache_hit: bool,
    ):
        status = "[CACHED RESPONSE]" if cache_hit else "[LIVE API INFERENCE]"
        print(f"\n================== GROUNDED RAG ANSWER {status} ==================")
        print(f"Question: {query}\n")
        print(answer)
        print("----------------------------------------------------------------------")
        print("Retrieved Sources Displayed to LLM:")
        for i, c in enumerate(chunks, 1):
            meta = c["metadata"]
            print(
                f"  * Chunk #{i}: {meta['source_doc']} (Page {meta['page_number']}) -> Distance: {c['distance']:.4f}"
            )
        print("======================================================================\n")

        print("--- AI ENGINEER METRICS: LATENCY & TOKEN AUDIT ---")
        print(f"  * Embedding Latency    : {latencies['embed_time']*1000:.2f} ms")
        print(f"  * Chroma Search Latency: {latencies['search_time']*1000:.2f} ms")
        print(f"  * Groq Generation Time : {latencies['llm_time']:.4f} seconds")
        print(f"  * TOTAL Turn Time      : {latencies['total_time']:.4f} seconds")
        print("  ----------------------------------------------")
        print(f"  * Turn Prompt Tokens   : {usage['prompt_tokens']}")
        print(f"  * Turn Output Tokens   : {usage['completion_tokens']}")
        print(f"  * SESSION RUNNING TOTAL: {self.token_audit['total_tokens']} tokens "
              f"({self.token_audit['prompt_tokens']} in / {self.token_audit['completion_tokens']} out)")
        print("--------------------------------------------------\n")


if __name__ == "__main__":
    logger.info("Initializing persistent ChromaDB client and embedding model...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    rag_collection = client.get_collection(name=COLLECTION_NAME)
    embedding_model = get_embedding_model()
    groq_api_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Initialize stateful chat session
    chat_session = RAGChatSession(rag_collection, embedding_model, groq_api_client)

    print("\n" + "="*70)
    print(" DAY 5: MULTI-TURN RAG CHAT INTERFACE (Type 'exit' to quit)")
    print("="*70)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print(f"\nExiting session. Final Token Count: {chat_session.token_audit['total_tokens']} tokens.")
                break
            
            chat_session.ask(user_input)
            
        except KeyboardInterrupt:
            print("\nSession interrupted by user.")
            break