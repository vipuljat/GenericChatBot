"""
Refactored Service Architecture - FUNCTIONAL APPROACH
======================================================

All services using pure functions instead of classes.
Cost tracking integrated throughout.
"""

# ============================================================================
# 1. services/embedding_service.py - FUNCTIONAL EMBEDDING SERVICE
# ============================================================================

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
_use_gemini: bool = False
_openai_model: str = None
_gemini_model: str = None


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
    global _openai_client, _use_gemini
    
    if _openai_client is not None or _use_gemini:
        return
    
    # Try OpenAI first
    if _init_openai():
        _use_gemini = False
        return
    
    # Fallback to Gemini
    log.warning("OpenAI unavailable, falling back to Gemini...")
    if _init_gemini():
        _use_gemini = True
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
    global _use_gemini   # ✅ REQUIRED

    _ensure_initialized()
    
    # Log cost
    cost = estimate_embedding_cost([text])
    log.info(
        f"Embedding: {cost['estimated_tokens']} tokens, "
        f"${cost['estimated_cost_usd']:6f} (single text)"
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
            _init_gemini()
            return generate_embedding(text)


def generate_embeddings_batch(
    texts: List[str],
    batch_size: int = 100
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.
    Automatically logs cost for each batch.
    """
    _ensure_initialized()
    
    # Log total estimated cost
    total_cost = estimate_embedding_cost(texts)
    log.info(
        f"📊 Batch embedding: {len(texts)} texts, "
        f"{total_cost['estimated_tokens']} tokens, "
        f"${total_cost['estimated_cost_usd']}"
    )
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Log batch cost
        batch_cost = estimate_embedding_cost(batch)
        log.info(
            f"Batch {i//batch_size + 1}: {len(batch)} texts, "
            f"{batch_cost['estimated_tokens']} tokens, "
            f"${batch_cost['estimated_cost_usd']}"
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
            # OpenAI: true batch processing
            response = _openai_client.embeddings.create(
                model=_openai_model,
                input=batch
            )
            embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(embeddings)
        
        log.info(f"✓ Batch {i//batch_size + 1} complete")
    
    log.info(f"✓ Generated {len(all_embeddings)} embeddings total")
    return all_embeddings


# ============================================================================
# TEXT CHUNKING FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\x00', '')
    return text.strip()


def split_by_chars(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Character-level split with overlap (last resort)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def split_by_separators(
    text: str,
    separators: List[str],
    chunk_size: int,
    overlap: int
) -> List[str]:
    """Recursively split text using separator hierarchy."""
    if not separators or not separators[0]:
        return split_by_chars(text, chunk_size, overlap)
    
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    sep = separators[0]
    remaining_seps = separators[1:]
    splits = text.split(sep)
    
    chunks = []
    current_parts = []
    current_len = 0
    
    for split in splits:
        split_len = len(split) + len(sep)
        
        # If single split is too large, recursively split it
        if split_len > chunk_size:
            if current_parts:
                chunks.append(sep.join(current_parts))
                current_parts = []
                current_len = 0
            
            sub_chunks = split_by_separators(split, remaining_seps, chunk_size, overlap)
            chunks.extend(sub_chunks)
            continue
        
        # Check if adding this split exceeds chunk size
        if current_len + split_len > chunk_size and current_parts:
            chunks.append(sep.join(current_parts))
            
            # Add overlap
            if current_len > overlap:
                overlap_parts = current_parts[-(len(current_parts)//2):]
                current_parts = overlap_parts
                current_len = sum(len(s) + len(sep) for s in overlap_parts)
            else:
                current_parts = []
                current_len = 0
        
        current_parts.append(split)
        current_len += split_len
    
    if current_parts:
        chunks.append(sep.join(current_parts))
    
    return chunks


def chunk_text(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 1200,
    overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks with metadata.
    Pure functional approach.
    """
    if not text or not text.strip():
        return []
    
    text = clean_text(text)
    
    # Separator hierarchy: paragraphs → sentences → words → chars
    separators = ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ', ' ', '']
    chunks = split_by_separators(text, separators, chunk_size, overlap)
    
    # Build chunk dictionaries
    result = [
        {
            "text": chunk,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "char_count": len(chunk),
            **(metadata or {})
        }
        for i, chunk in enumerate(chunks)
    ]
    
    log.info(f"Chunked text into {len(result)} chunks")
    return result


# ============================================================================
# COMPLETE DOCUMENT PROCESSING PIPELINE
# ============================================================================

def process_document_for_embedding(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 1200,
    overlap: int = 200
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


