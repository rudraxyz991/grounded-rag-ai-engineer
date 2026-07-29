import time
import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)

def extract_text_by_page(file_path: str) -> list:
    """Extracts text and preserves page numbers for metadata."""
    start_time = time.time()
    logger.info(f"Starting page-level extraction for: {file_path}")
    
    pages_data = []
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                clean_text = " ".join(text.split())
                pages_data.append({
                    "page_number": i + 1, 
                    "text": clean_text, 
                    "source_doc": file_path
                })
                
        duration = time.time() - start_time
        logger.info(f"Extracted {len(pages_data)} pages in {duration:.4f} seconds.")
        return pages_data
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return []