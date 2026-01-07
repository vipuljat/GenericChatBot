"""Functional embedding service with OpenAI and Gemini fallback."""

import re
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import google.generativeai as genai
import config
from utils.logging import log


# ============================================================================
# COST CALCULATION
# ============================================================================

EMBEDDING_COST_PER_1K_TOKENS = 0.0001  # text-embedding-ada-002
AVG_CHARS_PER_TOKEN = 4               # ~4 chars per token (English)

def estimate_embedding_cost(texts: List[str]) -> Dict[str, float]:
    """
    Estimate token usage and cost for embedding texts.
    """
    total_chars = sum(len(t) for t in texts)
    estimated_tokens = total_chars / AVG_CHARS_PER_TOKEN
    estimated_cost = (estimated_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS

    return {
        "total_chars": total_chars,
        "estimated_tokens": int(estimated_tokens),
        "estimated_cost_usd": estimated_cost,
    }


# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

_openai_client: Optional[OpenAI] = None
_use_gemini: bool = False  # Changed to False - we'll set this properly during init
_openai_model: str = None
_gemini_model: str = None
_initialized: bool = False  # Track if we've actually initialized


def _init_openai() -> bool:
    """Initialize OpenAI client. Returns True if successful."""
    global _openai_client, _openai_model
    try:
        if not config.OPENAI_API_KEY:
            return False
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        _openai_model = config.OPENAI_EMBEDDING_MODEL
        log.info(f"✓ Embedding: OpenAI ({_openai_model})")
        return True
    except Exception as e:
        log.warning(f"OpenAI init failed: {e}")
        return False


def _init_gemini() -> bool:
    """Initialize Gemini client. Returns True if successful."""
    global _gemini_model
    try:
        if not hasattr(config, 'GEMINI_API_KEY') or not config.GEMINI_API_KEY:
            return False
        genai.configure(api_key=config.GEMINI_API_KEY)
        _gemini_model = getattr(config, 'GEMINI_EMBEDDING_MODEL', 'models/text-embedding-004')
        log.info(f"✓ Embedding: Gemini ({_gemini_model})")
        return True
    except Exception as e:
        log.error(f"Gemini init failed: {e}")
        return False


def _ensure_initialized():
    """Ensure embedding client is initialized with fallback."""
    global _openai_client, _use_gemini, _initialized
    
    # If already initialized, skip
    if _initialized:
        return
    
    # Try OpenAI first
    if _init_openai():
        _use_gemini = False
        _initialized = True
        return
    
    # Fallback to Gemini
    log.warning("OpenAI unavailable, falling back to Gemini...")
    if _init_gemini():
        _use_gemini = True
        _initialized = True
        return
    
    raise RuntimeError("Neither OpenAI nor Gemini embedding services available")


# ============================================================================
# CORE EMBEDDING FUNCTIONS
# ============================================================================

def generate_embedding(text: str) -> List[float]:
    """
    Generate single embedding vector.
    Automatically logs cost estimation.
    """
    global _use_gemini

    _ensure_initialized()
    
    # Log cost
    cost = estimate_embedding_cost([text])
    log.info(
        f"Embedding: {cost['estimated_tokens']} tokens, "
        f"${cost['estimated_cost_usd']:.6f} (single text)"
    )
    
    if _use_gemini:
        result = genai.embed_content(
            model=_gemini_model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    else:
        try:
            response = _openai_client.embeddings.create(
                model=_openai_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            log.error(f"OpenAI embedding failed: {e}")
            # Auto-fallback to Gemini
            log.warning("Falling back to Gemini for embedding...")
            _use_gemini = True
            if not _init_gemini():
                raise RuntimeError("Gemini fallback failed after OpenAI error")
            return generate_embedding(text)


def generate_embeddings_batch(
    texts: List[str],
    batch_size: int = 100
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.
    Automatically logs cost for each batch.
    """
    global _use_gemini
    
    _ensure_initialized()
    
    # Log total estimated cost
    total_cost = estimate_embedding_cost(texts)
    log.info(
        f"📊 Batch embedding: {len(texts)} texts, "
        f"{total_cost['estimated_tokens']} tokens, "
        f"${total_cost['estimated_cost_usd']:.6f}"
    )
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Log batch cost
        batch_cost = estimate_embedding_cost(batch)
        log.info(
            f"Batch {i//batch_size + 1}: {len(batch)} texts, "
            f"{batch_cost['estimated_tokens']} tokens, "
            f"${batch_cost['estimated_cost_usd']:.6f}"
        )
        
        if _use_gemini:
            # Gemini: process sequentially
            for text in batch:
                result = genai.embed_content(
                    model=_gemini_model,
                    content=text,
                    task_type="retrieval_document"
                )
                all_embeddings.append(result['embedding'])
        else:
            try:
                response = _openai_client.embeddings.create(
                    model=_openai_model,
                    input=batch
                )
                embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(embeddings)

            except Exception as e:
                log.error(f"OpenAI batch embedding failed: {e}")
                log.warning("Falling back to Gemini for batch embedding...")

                _use_gemini = True
                if not _init_gemini():
                    raise RuntimeError("Gemini fallback failed after OpenAI batch error")

                # Retry this batch using Gemini
                for text in batch:
                    result = genai.embed_content(
                        model=_gemini_model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    all_embeddings.append(result["embedding"])

        
        log.info(f"✓ Batch {i//batch_size + 1} complete")
    
    log.info(f"✓ Generated {len(all_embeddings)} embeddings total")
    return all_embeddings


# ============================================================================
# TEXT CHUNKING FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text more aggressively."""
    # Remove multiple whitespaces, newlines, and special characters
    text = re.sub(r'\s+', ' ', text)  # Replace all whitespace with single space
    text = re.sub(r'[\n\r\t\f\v]+', ' ', text)  # Extra newline handling
    text = text.replace('\x00', '')  # Null bytes
    text = re.sub(r'[-_]{2,}', ' ', text)  # Multiple dashes/underscores
    return text.strip()


def split_by_sentences(text: str) -> List[str]:
    """Improved sentence splitting using regex for better semantic chunks."""
    # Split on sentence boundaries: . ! ? followed by space or capital letter
    sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if s.strip()]


def split_by_chars(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Character-level split with overlap (last resort)."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else start + 1  # Ensure progress
    return chunks


def split_by_separators(
    text: str,
    separators: List[str],
    chunk_size: int,
    overlap: int
) -> List[str]:
    """Recursively split text using separator hierarchy with improved overlap handling."""
    if not separators or not separators[0]:
        return split_by_chars(text, chunk_size, overlap)
    
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    sep = separators[0]
    remaining_seps = separators[1:]
    splits = re.split(f'({re.escape(sep)})', text)  # Preserve separators
    
    chunks = []
    current_chunk = ""
    current_len = 0
    
    for split in splits:
        if not split:
            continue
        
        split_len = len(split)
        
        # If single split is too large, recursively split it
        if split_len > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap from end of previous chunk
                overlap_text = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
                current_chunk = overlap_text
                current_len = len(overlap_text)
            else:
                current_len = 0
            
            sub_chunks = split_by_separators(split, remaining_seps, chunk_size, overlap)
            chunks.extend(sub_chunks)
            continue
        
        # Check if adding this split exceeds chunk size
        if current_len + split_len > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap from end
                overlap_text = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
                current_chunk = overlap_text
                current_len = len(overlap_text)
            else:
                current_len = 0
        
        current_chunk += split
        current_len += split_len
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [c for c in chunks if c]


def chunk_text(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 600,  # Reduced for more granular chunks
    overlap: int = 150  # Increased overlap for better context continuity
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks with metadata.
    Improved for better semantic splitting.
    """
    if not text or not text.strip():
        return []
    
    text = clean_text(text)
    
    # First, split into sentences for semantic chunks
    sentences = split_by_sentences(text)
    
    # Separator hierarchy: now starting from paragraphs, then sentences (already split), words, chars
    separators = ['\n\n', '\n', ' ', '']
    
    # Join sentences back and split using hierarchy
    text = ' '.join(sentences)  # Rejoin for hierarchical splitting
    chunks = split_by_separators(text, separators, chunk_size, overlap)
    
    # Build chunk dictionaries
    result = []
    for i, chunk in enumerate(chunks):
        if len(chunk) < 50:  # Skip very small chunks
            continue
        chunk_dict = {
            "text": chunk,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "char_count": len(chunk),
            **(metadata or {})
        }
        result.append(chunk_dict)
        # Log sample of chunk for debugging
        log.info(f"Generated Chunk {i+1}/{len(chunks)}: {chunk[:200]}...")
    
    log.info(f"Chunked text into {len(result)} chunks (size={chunk_size}, overlap={overlap})")
    return result


# ============================================================================
# COMPLETE DOCUMENT PROCESSING PIPELINE
# ============================================================================

def process_document_for_embedding(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 600,  # Smaller chunks for better retrieval
    overlap: int = 150  # Better overlap
) -> List[Dict[str, Any]]:
    """
    Complete pipeline: chunk text and generate embeddings.
    ONE-STOP functional solution.
    
    Automatically logs all costs.
    """
    # Step 1: Chunk
    chunks = chunk_text(text, metadata, chunk_size, overlap)
    if not chunks:
        log.warning("No chunks generated from text")
        return []
    
    log.info(f"Processing {len(chunks)} chunks for embedding...")
    
    # Step 2: Generate embeddings with cost logging
    texts = [c['text'] for c in chunks]
    embeddings = generate_embeddings_batch(texts)
    
    # Step 3: Attach embeddings
    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding
    
    total_chars = sum(c['char_count'] for c in chunks)
    log.info(f"✓ Processed document: {len(chunks)} chunks, {total_chars} chars total")
    
    return chunks