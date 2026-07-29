import time
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBED_BATCH_SIZE,
)
from extract import extract_text_by_page
from chunker import chunk_document_langchain

# Configure professional logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """Loads and returns the SentenceTransformer model."""
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def generate_embeddings_batched(model, texts: list, batch_size: int = EMBED_BATCH_SIZE):
    """AI Engineer Layer: Generates embeddings in batches and logs throughput (chunks/sec)."""
    start_time = time.time()
    logger.info(
        f"Generating embeddings for {len(texts)} chunks (Batch size: {batch_size})..."
    )

    # Encode supports batched processing natively
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
    )

    duration = time.time() - start_time
    throughput = len(texts) / duration if duration > 0 else 0

    logger.info(f"Embedding completed in {duration:.4f} seconds.")
    logger.info(
        f"AI ENGINEER METRIC -> Throughput: {throughput:.2f} chunks/sec | Total Chunks: {len(texts)}"
    )

    return embeddings, throughput


def build_and_populate_vector_store(chunks: list, model):
    """Creates a persistent ChromaDB collection and inserts chunks with metadata + embeddings."""
    logger.info(
        f"Initializing persistent ChromaDB client at path: '{CHROMA_DB_PATH}'"
    )
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Use get_or_create_collection so running the script multiple times won't crash
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Extract lists required by ChromaDB
    ids = [chunk["chunk_id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]

    # Clean metadata: ChromaDB strictly requires string, int, float, or bool values
    metadatas = [
        {
            "source_doc": str(chunk["source_doc"]),
            "page_number": int(chunk["page_number"]),
            "char_count": int(chunk["char_count"]),
            "token_count": int(chunk["token_count"]),
        }
        for chunk in chunks
    ]

    # Generate embeddings with throughput logging
    embeddings, _ = generate_embeddings_batched(
        model, texts, batch_size=EMBED_BATCH_SIZE
    )

    logger.info("Upserting records into ChromaDB collection...")
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    logger.info(
        f"Success! Indexed {len(ids)} chunks into collection '{COLLECTION_NAME}'."
    )
    return collection


def search_similar_chunks(collection, model, query: str, top_k: int = 3):
    """Executes a similarity search against the ChromaDB collection."""
    logger.info(f"Searching top-{top_k} results for query: '{query}'")
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


if __name__ == "__main__":
    sample_doc = "sample.pdf"

    # 1. Run Pipeline from Day 1 & Day 2
    logger.info("--- Phase 1: Ingestion & Chunking ---")
    pages = extract_text_by_page(sample_doc)
    chunks = chunk_document_langchain(pages, CHUNK_SIZE, CHUNK_OVERLAP)

    # 2. Initialize Model & Vector Store
    logger.info("--- Phase 2: Embedding & Vector Storage ---")
    embed_model = get_embedding_model()
    chroma_collection = build_and_populate_vector_store(chunks, embed_model)

    # 3. Core Task: Manual Semantic Search Testing
    logger.info("--- Phase 3: Testing Semantic Search ---")
    test_query = "What is the main topic or key conclusion of this document?"

    search_results = search_similar_chunks(
        chroma_collection, embed_model, test_query, top_k=2
    )

    print("\n================== SEMANTIC SEARCH RESULTS ==================")
    print(f"Query: '{test_query}'\n")

    for i in range(len(search_results["documents"][0])):
        doc_text = search_results["documents"][0][i]
        metadata = search_results["metadatas"][0][i]
        distance = search_results["distances"][0][i]

        print(f"--- Result #{i+1} (Distance Score: {distance:.4f}) ---")
        print(
            f"Source: {metadata['source_doc']} | Page: {metadata['page_number']} | Tokens: {metadata['token_count']}"
        )
        print(f"Snippet: {doc_text[:250]}...\n")
    print("=============================================================\n")