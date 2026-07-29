import tiktoken
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
from extract import extract_text_by_page

# Set up professional logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize tokenizer for the AI Engineer Layer token counting
tokenizer = tiktoken.get_encoding("cl100k_base") 

def count_tokens(text: str) -> int:
    """Returns the token count of a given string."""
    return len(tokenizer.encode(text))

def chunk_document_langchain(pages_data: list, chunk_size: int, chunk_overlap: int) -> list:
    """Splits text using LangChain while preserving rigorous metadata."""
    
    # Initialize LangChain's smart splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = []
    chunk_idx = 0
    
    for page in pages_data:
        text = page["text"]
        source = page["source_doc"]
        page_num = page["page_number"]
        
        # Let LangChain handle the complex semantic splitting
        split_texts = text_splitter.split_text(text)
        
        for chunk_text in split_texts:
            tokens = count_tokens(chunk_text)
            
            # Core Task: Store chunk metadata exactly as before
            chunk_metadata = {
                "chunk_id": f"{source}_p{page_num}_c{chunk_idx}",
                "text": chunk_text,
                "source_doc": source,
                "page_number": page_num,
                "char_count": len(chunk_text),
                "token_count": tokens
            }
            chunks.append(chunk_metadata)
            chunk_idx += 1
            
    return chunks

if __name__ == "__main__":
    sample_doc = "sample.pdf" 
    
    logger.info("Step 1: Extracting text by page...")
    pages = extract_text_by_page(sample_doc)
    
    logger.info(f"Step 2: LangChain Chunking (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
    document_chunks = chunk_document_langchain(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    
    logger.info(f"Success! Created a clean list of {len(document_chunks)} chunks using LangChain.")
    
    # Core Task: Print and inspect chunks manually
    if document_chunks:
        print("\n--- Inspecting First Chunk Metadata ---")
        sample = document_chunks[0]
        for key, value in sample.items():
            if key == "text":
                print(f"{key}: {value[:100]}... (truncated for preview)")
            else:
                print(f"{key}: {value}")
        print("---------------------------------------\n")