"""Functional embedding service with OpenAI and Gemini fallback."""

import re
from typing import List, Dict, Any, Optional, Tuple
from google.api_core.exceptions import (
    DeadlineExceeded,
    GoogleAPICallError,
    InvalidArgument,
    PermissionDenied,
    RetryError,
    ServiceUnavailable,
)
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
import google.generativeai as genai
import config
from utils.logging import log


# ============================================================================
# COST CALCULATION
# ============================================================================

EMBEDDING_COST_PER_1K_TOKENS = 0.0001  # text-embedding-ada-002
AVG_CHARS_PER_TOKEN = 4               # ~4 chars per token (English)
MIN_CHUNK_SIZE_TOKENS = 50
MAX_CHUNK_SIZE_TOKENS = 2000
SUPPORTED_CHUNKING_TYPES = {"smart", "legacy", "fixed"}

OPENAI_EMBEDDING_EXCEPTIONS = (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAIError,
    RateLimitError,
)

GEMINI_EMBEDDING_EXCEPTIONS = (
    DeadlineExceeded,
    GoogleAPICallError,
    InvalidArgument,
    PermissionDenied,
    RetryError,
    ServiceUnavailable,
)

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
_openai_initialized: bool = False
_gemini_initialized: bool = False


def _init_openai() -> bool:
    """Initialize OpenAI client. Returns True if successful."""
    global _openai_client, _openai_model, _openai_initialized
    if _openai_initialized and _openai_client is not None and _openai_model:
        return True
    try:
        if not config.OPENAI_API_KEY:
            return False
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        _openai_model = config.OPENAI_EMBEDDING_MODEL
        _openai_initialized = True
        log.info(f"✓ Embedding: OpenAI ({_openai_model})")
        return True
    except (AuthenticationError, BadRequestError, OpenAIError, TypeError, ValueError) as e:
        log.warning(f"OpenAI init failed: {e}")
        return False


def _init_gemini() -> bool:
    """Initialize Gemini client. Returns True if successful."""
    global _gemini_model, _gemini_initialized
    if _gemini_initialized and _gemini_model:
        return True
    try:
        if not hasattr(config, 'GEMINI_API_KEY') or not config.GEMINI_API_KEY:
            return False
        genai.configure(api_key=config.GEMINI_API_KEY)
        _gemini_model = getattr(config, 'GEMINI_EMBEDDING_MODEL', 'models/gemini-embedding-001')
        _gemini_initialized = True
        log.info(f"✓ Embedding: Gemini ({_gemini_model})")
        return True
    except (TypeError, ValueError, RuntimeError) as e:
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


def _validate_embedding_vector(embedding: Any, provider: str) -> List[float]:
    """Validate a provider response contains a usable embedding vector."""
    if not isinstance(embedding, list) or not embedding:
        raise ValueError(f"{provider} response missing embedding vector")

    if not all(isinstance(value, (int, float)) for value in embedding):
        raise ValueError(f"{provider} embedding vector contains non-numeric values")

    return embedding


def _extract_gemini_embedding(result: Any) -> List[float]:
    """Extract and validate a Gemini embedding response."""
    embedding = result.get("embedding") if isinstance(result, dict) else getattr(result, "embedding", None)
    return _validate_embedding_vector(embedding, "Gemini")


def _extract_openai_embedding(response: Any) -> List[float]:
    """Extract and validate a single OpenAI embedding response."""
    data = getattr(response, "data", None)
    if not isinstance(data, list) or not data:
        raise ValueError("OpenAI response missing data")

    return _validate_embedding_vector(getattr(data[0], "embedding", None), "OpenAI")


def _extract_openai_batch_embeddings(response: Any, expected_count: int) -> List[List[float]]:
    """Extract and validate batch embeddings from OpenAI."""
    data = getattr(response, "data", None)
    if not isinstance(data, list) or len(data) != expected_count:
        raise ValueError(
            f"OpenAI batch response count mismatch: expected {expected_count}, got {len(data) if isinstance(data, list) else 'invalid'}"
        )

    embeddings: List[List[float]] = []
    for index, item in enumerate(data):
        embeddings.append(
            _validate_embedding_vector(getattr(item, "embedding", None), f"OpenAI batch item {index}")
        )

    return embeddings


def _generate_gemini_embedding(text: str, task_type: str = "retrieval_document") -> List[float]:
    """Generate and validate a single Gemini embedding."""
    result = genai.embed_content(
        model=_gemini_model,
        content=text,
        task_type=task_type
    )
    return _extract_gemini_embedding(result)


def _generate_gemini_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate and validate a Gemini batch sequentially (document indexing)."""
    embeddings: List[List[float]] = []
    for text in texts:
        embeddings.append(_generate_gemini_embedding(text, task_type="retrieval_document"))
    return embeddings


def _switch_to_gemini(reason: str) -> None:
    """Enable Gemini fallback without redundant reinitialization."""
    global _use_gemini
    log.warning(reason)
    _use_gemini = True
    if not _init_gemini():
        raise RuntimeError("Gemini fallback failed after OpenAI error")


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
    
    # Queries use retrieval_query task type (different from document indexing)
    if _use_gemini:
        try:
            return _generate_gemini_embedding(text, task_type="retrieval_query")
        except GEMINI_EMBEDDING_EXCEPTIONS as e:
            log.error(f"Gemini embedding failed: {e}")
            raise RuntimeError("Gemini embedding request failed") from e
        except ValueError as e:
            log.error(f"Invalid Gemini embedding response: {e}")
            raise RuntimeError("Gemini embedding response invalid") from e

    try:
        response = _openai_client.embeddings.create(
            model=_openai_model,
            input=text
        )
        return _extract_openai_embedding(response)
    except OPENAI_EMBEDDING_EXCEPTIONS as e:
        log.error(f"OpenAI embedding failed: {e}")
        _switch_to_gemini("Falling back to Gemini for embedding...")
        try:
            return _generate_gemini_embedding(text, task_type="retrieval_query")
        except GEMINI_EMBEDDING_EXCEPTIONS as gemini_error:
            log.error(f"Gemini embedding failed after fallback: {gemini_error}")
            raise RuntimeError("Gemini fallback failed after OpenAI error") from gemini_error
        except ValueError as gemini_error:
            log.error(f"Invalid Gemini embedding response after fallback: {gemini_error}")
            raise RuntimeError("Gemini fallback response invalid") from gemini_error
    except (AttributeError, IndexError, TypeError, ValueError) as e:
        log.error(f"Invalid OpenAI embedding response: {e}")
        _switch_to_gemini("OpenAI returned an invalid embedding response. Falling back to Gemini...")
        try:
            return _generate_gemini_embedding(text, task_type="retrieval_query")
        except GEMINI_EMBEDDING_EXCEPTIONS as gemini_error:
            log.error(f"Gemini embedding failed after invalid OpenAI response: {gemini_error}")
            raise RuntimeError("Gemini fallback failed after invalid OpenAI response") from gemini_error
        except ValueError as gemini_error:
            log.error(f"Invalid Gemini embedding response after invalid OpenAI response: {gemini_error}")
            raise RuntimeError("Gemini fallback response invalid") from gemini_error


def generate_embeddings_batch(
    texts: List[str],
    batch_size: int = 100
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.
    Automatically logs cost for each batch.
    """
    global _use_gemini      
    
    _ensure_initialized()     # Ensure client is initialized
    
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
            try:
                all_embeddings.extend(_generate_gemini_embeddings_batch(batch))
            except GEMINI_EMBEDDING_EXCEPTIONS as e:
                log.error(f"Gemini batch embedding failed: {e}")
                raise RuntimeError("Gemini batch embedding request failed") from e
            except ValueError as e:
                log.error(f"Invalid Gemini batch embedding response: {e}")
                raise RuntimeError("Gemini batch embedding response invalid") from e
        else:
            try:
                response = _openai_client.embeddings.create(
                    model=_openai_model,
                    input=batch
                )
                embeddings = _extract_openai_batch_embeddings(response, len(batch))
                all_embeddings.extend(embeddings)

            except OPENAI_EMBEDDING_EXCEPTIONS as e:
                log.error(f"OpenAI batch embedding failed: {e}")
                _switch_to_gemini("Falling back to Gemini for batch embedding...")
                try:
                    all_embeddings.extend(_generate_gemini_embeddings_batch(batch))
                except GEMINI_EMBEDDING_EXCEPTIONS as gemini_error:
                    log.error(f"Gemini batch embedding failed after fallback: {gemini_error}")
                    raise RuntimeError("Gemini batch fallback failed after OpenAI error") from gemini_error
                except ValueError as gemini_error:
                    log.error(f"Invalid Gemini batch response after fallback: {gemini_error}")
                    raise RuntimeError("Gemini batch fallback response invalid") from gemini_error
            except (AttributeError, IndexError, TypeError, ValueError) as e:
                log.error(f"Invalid OpenAI batch embedding response: {e}")
                _switch_to_gemini("OpenAI returned an invalid batch response. Falling back to Gemini...")
                try:
                    all_embeddings.extend(_generate_gemini_embeddings_batch(batch))
                except GEMINI_EMBEDDING_EXCEPTIONS as gemini_error:
                    log.error(f"Gemini batch embedding failed after invalid OpenAI response: {gemini_error}")
                    raise RuntimeError("Gemini batch fallback failed after invalid OpenAI response") from gemini_error
                except ValueError as gemini_error:
                    log.error(f"Invalid Gemini batch response after invalid OpenAI response: {gemini_error}")
                    raise RuntimeError("Gemini batch fallback response invalid") from gemini_error

        
        log.info(f"✓ Batch {i//batch_size + 1} complete")
    
    log.info(f"✓ Generated {len(all_embeddings)} embeddings total")
    return all_embeddings


# ============================================================================
# TEXT CHUNKING FUNCTIONS
# ============================================================================

# Answer markers — require an explicit delimiter (:, -, .) after the prefix so
# that words like "Administration" or "And" are never mistaken for answer starts.
_QA_ANSWER_MARKER = re.compile(
    r'^(?:'
    r'A\s*[.:]\s+'           # "A: " or "A. "
    r'|Ans\s*[-:.]\s*'       # "Ans: " / "Ans - " / "Ans. "
    r'|Answer\s*\d*\s*[:.]\s*'  # "Answer: " / "Answer 1: "
    r')',
    re.IGNORECASE | re.MULTILINE
)

# Question markers — explicit "Q:" / "Q." / "Question:" only (no plain numbers,
# because "1." appears everywhere in bullet lists inside answers).
_QA_EXPLICIT_QUESTION = re.compile(
    r'^(?:Q\s*[.:]\s*|Question\s*\d*\s*[.:]\s*)',
    re.IGNORECASE | re.MULTILINE
)

# Numbered paragraph question: "1. " / "1) " at the START of a paragraph
# (preceded by a blank line or beginning of text).  Keeps numbered sub-steps
# inside answers from matching — they appear mid-paragraph, not after \n\n.
_QA_NUMBERED_PARA_QUESTION = re.compile(
    r'(?:\A|\n\n)\s*\d+\s*[.)]\s+\S',
)


def _para_is_question(para: str) -> bool:
    """True when a single paragraph looks like a Q&A question."""
    return bool(
        _QA_EXPLICIT_QUESTION.match(para)
        or _QA_NUMBERED_PARA_QUESTION.match(para)
    )


def _para_is_answer(para: str) -> bool:
    """True when a single paragraph looks like a Q&A answer."""
    return bool(_QA_ANSWER_MARKER.match(para))


def _chunk_mixed_content(
    text: str,
    chunk_size_chars: int,
    overlap_chars: int,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """
    Paragraph-level hybrid chunker for documents with mixed content
    (Q&A sections, tables, policy text, department info, etc.).

    Algorithm
    ---------
    1. Split text into paragraphs (on blank lines).
    2. Walk paragraphs and detect Q+A pairs: a paragraph that looks like a
       question immediately followed by a paragraph that looks like an answer
       → glue them into one atomic unit.
    3. Everything else (tables, headings, policy prose, etc.) becomes its own
       unit and is handled by smart/sentence chunking.
    4. Batch units into final chunks respecting chunk_size_chars.
    5. Oversized single units are sub-chunked so nothing is ever dropped.
    """
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    # --- Step 2 & 3: build atomic units, tracking which are Q+A pairs ---
    units: List[str] = []
    qa_flags: List[bool] = []   # True = this unit is a Q+A pair (must not be split)
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        # Peek ahead: is next paragraph the answer to this question?
        if _para_is_question(para) and i + 1 < len(paragraphs) and _para_is_answer(paragraphs[i + 1]):
            units.append(para + '\n\n' + paragraphs[i + 1])
            qa_flags.append(True)
            i += 2
            continue
        units.append(para)
        qa_flags.append(False)
        i += 1

    # --- Step 4 & 5: batch units into chunks ---
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for unit, is_qa in zip(units, qa_flags):
        unit_len = len(unit) + 2  # +2 for '\n\n' separator between units

        if unit_len > chunk_size_chars:
            # Q+A pairs: NEVER split — keep together even if oversized.
            # The RAG context window (15 000 chars) can handle it.
            if is_qa:
                if current:
                    chunks.append('\n\n'.join(current))
                    current = []
                    current_len = 0
                chunks.append(unit)
                continue

            # Non-Q+A oversized unit (long table, policy section): sub-chunk it
            # so content is not silently dropped.
            if current:
                chunks.append('\n\n'.join(current))
                current = []
                current_len = 0
            sub = split_smart_chunks(unit, chunk_size_tokens, overlap_tokens)
            chunks.extend(sub if sub else [unit])
            continue

        if current and current_len + unit_len > chunk_size_chars:
            chunks.append('\n\n'.join(current))
            # Overlap: carry the last unit forward for context continuity
            last = current[-1]
            if len(last) <= overlap_chars:
                current = [last]
                current_len = len(last)
            else:
                current = []
                current_len = 0

        current.append(unit)
        current_len += unit_len

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


def clean_text(text: str) -> str:
    """Clean text while preserving paragraph and sentence boundaries."""
    text = text.replace('\x00', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t\f\v]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[-_]{2,}', ' ', text)

    # ── PDF structure restoration ────────────────────────────────────────────
    # PDF extractors (even pdfplumber) can merge numbered section headings inline.
    # Insert paragraph break before "N.M SectionTitle" or "N. SectionTitle" patterns
    # that appear mid-text (preceded by any non-whitespace char including ".").
    # This is generic — works for any policy/handbook/legal document.
    text = re.sub(r'(?<=\S) (\d{1,2}\.\d+\s+[A-Z][a-zA-Z]{3,})', r'\n\n\1', text)

    # ── Q&A paragraph recovery ────────────────────────────────────────────────
    # PDF extractors often use single \n between Q&A pairs (not double \n), causing
    # the entire page to appear as one giant paragraph. Promote answer and question
    # markers to paragraph breaks so _chunk_mixed_content can pair them correctly.
    # Handles: "Ans:", "Ans.", "Answer:", "A: " and "Q1.", "Q 1.", "QUE 1."
    text = re.sub(r'(?<!\n)\n((?:Ans|Answer)\s*[.:]\s)', r'\n\n\1', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!\n)\n(A\s*:\s)', r'\n\n\1', text)
    text = re.sub(r'(?<!\n)\n(Q(?:ue)?\s*\d+\s*[.:])', r'\n\n\1', text, flags=re.IGNORECASE)

    # ── Merged table-row recovery ─────────────────────────────────────────────
    # When PDF extractors merge tabular rows onto one line, rows that contain a
    # date-like token (1-Jan-26, 15/03/2026, 2026-01-01) end up as one long string.
    # Insert a newline before each date so each row starts on its own line.
    # Pattern covers:  DD-Mon-YY(YY)  |  DD/MM/YYYY  |  YYYY-MM-DD
    text = re.sub(
        r'(?<=[A-Za-z])\s+(\d{1,2}[-/][A-Z][a-z]{2}[-/]\d{2,4})',
        r'\n\1',
        text
    )
    text = re.sub(
        r'(?<=[A-Za-z])\s+(\d{4}[-/]\d{2}[-/]\d{2})',
        r'\n\1',
        text
    )

    return text.strip()


def split_by_sentences(text: str) -> List[str]:
    """Split text on likely sentence boundaries while keeping punctuation."""
    sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+')
    sentences = sentence_end.split(text)
    return [s.strip() for s in sentences if s.strip()]


def split_by_chars(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Character-level split with overlap (last resort)."""
    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))
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
    overlap: int,
    depth: int = 0,
    max_depth: Optional[int] = None
) -> List[str]:
    """Recursively split text using separator hierarchy with improved overlap handling."""
    if max_depth is None:
        max_depth = max(1, len(separators) + 2)

    if depth >= max_depth:
        log.warning("Chunk split depth limit reached; falling back to character splitting.")
        return split_by_chars(text, chunk_size, overlap)

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
            
            sub_chunks = split_by_separators(
                split,
                remaining_seps,
                chunk_size,
                overlap,
                depth=depth + 1,
                max_depth=max_depth
            )
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


def normalize_chunk_settings(chunk_size: int, overlap: int) -> Tuple[int, int]:
    """Ensure chunk settings are valid and overlap never exceeds chunk size."""
    chunk_size = int(chunk_size)
    overlap = max(0, int(overlap))

    if chunk_size < MIN_CHUNK_SIZE_TOKENS:
        log.warning(
            f"Chunk size {chunk_size} is below the supported minimum of {MIN_CHUNK_SIZE_TOKENS}; using minimum."
        )
        chunk_size = MIN_CHUNK_SIZE_TOKENS
    elif chunk_size > MAX_CHUNK_SIZE_TOKENS:
        log.warning(
            f"Chunk size {chunk_size} is above the supported maximum of {MAX_CHUNK_SIZE_TOKENS}; using maximum."
        )
        chunk_size = MAX_CHUNK_SIZE_TOKENS

    if overlap >= chunk_size:
        log.warning(
            f"Chunk overlap {overlap} cannot be greater than or equal to chunk size {chunk_size}; reducing overlap."
        )
        overlap = max(0, chunk_size // 4)

    max_recommended_overlap = max(0, chunk_size // 2)
    if overlap > max_recommended_overlap:
        log.warning(
            f"Chunk overlap {overlap} is too large for chunk size {chunk_size}; capping at {max_recommended_overlap}."
        )
        overlap = max_recommended_overlap

    return chunk_size, overlap


def tokens_to_chars(token_count: int) -> int:
    """Approximate token-based limits using the shared chars/token estimate."""
    return max(1, int(token_count) * AVG_CHARS_PER_TOKEN)


def split_into_semantic_units(text: str, chunk_size_chars: int) -> List[str]:
    """Create sentence-aware units and recursively split very long sentences."""
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    units: List[str] = []
    fallback_separators = ["\n", "; ", ": ", ", ", " ", ""]

    for paragraph_index, paragraph in enumerate(paragraphs):
        sentences = split_by_sentences(paragraph) or [paragraph]
        paragraph_units: List[str] = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > chunk_size_chars:
                paragraph_units.extend(
                    split_by_separators(sentence, fallback_separators, chunk_size_chars, 0)
                )
            else:
                paragraph_units.append(sentence)

        for unit_index, unit in enumerate(paragraph_units):
            unit = unit.strip()
            if not unit:
                continue

            is_last_unit = unit_index == len(paragraph_units) - 1
            has_more_paragraphs = paragraph_index < len(paragraphs) - 1

            if is_last_unit and has_more_paragraphs:
                units.append(f"{unit}\n\n")
            elif is_last_unit:
                units.append(unit)
            else:
                units.append(f"{unit} ")

    return units


def merge_units_with_overlap(
    units: List[str],
    chunk_size_chars: int,
    overlap_chars: int
) -> List[str]:
    """Merge sentence-aware units into overlapping chunks."""
    if not units:
        return []

    chunks: List[str] = []
    current_units: List[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit)

        if current_units and current_len + unit_len > chunk_size_chars:
            chunk_text = ''.join(current_units).strip()
            if chunk_text:
                chunks.append(chunk_text)

            while current_units and current_len > overlap_chars:
                removed = current_units.pop(0)
                current_len -= len(removed)

            while current_units and current_len + unit_len > chunk_size_chars:
                removed = current_units.pop(0)
                current_len -= len(removed)

        current_units.append(unit)
        current_len += unit_len

    if current_units:
        chunk_text = ''.join(current_units).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


def split_smart_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Sentence-aware recursive chunking with token-based limits."""
    chunk_size, overlap = normalize_chunk_settings(chunk_size, overlap)
    chunk_size_chars = tokens_to_chars(chunk_size)
    overlap_chars = tokens_to_chars(overlap)

    units = split_into_semantic_units(text, chunk_size_chars)
    return merge_units_with_overlap(units, chunk_size_chars, overlap_chars)


def normalize_chunking_type(chunking_type: Optional[str]) -> str:
    """Normalize chunking type and warn when the requested value is unsupported."""
    normalized = (chunking_type or getattr(config, 'CHUNKING_TYPE', 'smart')).strip().lower()
    if normalized not in SUPPORTED_CHUNKING_TYPES:
        log.warning(f"Unsupported chunking type '{normalized}'. Falling back to 'smart'.")
        return "smart"
    return normalized


def chunk_text(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    chunking_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks with metadata.
    Uses smart sentence-aware chunking by default.
    """
    if not text or not text.strip():
        return []

    chunk_size = int(chunk_size or getattr(config, 'CHUNK_SIZE', 400))
    overlap = int(overlap if overlap is not None else getattr(config, 'CHUNK_OVERLAP', 100))
    chunking_type = normalize_chunking_type(chunking_type)

    text = clean_text(text)

    chunk_size_chars = tokens_to_chars(chunk_size)
    overlap_chars = tokens_to_chars(overlap)

    if chunking_type in {'legacy', 'fixed'}:
        flattened_text = re.sub(r'\s+', ' ', text).strip()
        chunks = split_by_separators(
            flattened_text,
            ['\n\n', '\n', ' ', ''],
            chunk_size_chars,
            overlap_chars
        )
    else:
        # Paragraph-level hybrid: handles Q&A pairs, tables, policy text, etc.
        chunks = _chunk_mixed_content(
            text,
            chunk_size_chars,
            overlap_chars,
            chunk_size,
            overlap,
        )

    # Build chunk dictionaries
    result = []
    for i, chunk in enumerate(chunks):
        if not chunk or not chunk.strip():  # Skip empty chunks only
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
        log.debug(f"Generated Chunk {i+1}/{len(chunks)}: {chunk[:200]}...") 

    log.info(
        f"Chunked text into {len(result)} chunks "
        f"(type={chunking_type}, size={chunk_size}, overlap={overlap})"
    )
    return result


# ============================================================================
# COMPLETE DOCUMENT PROCESSING PIPELINE
# ============================================================================

def process_document_for_embedding(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    chunking_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Complete pipeline: chunk text and generate embeddings.
    ONE-STOP functional solution.
    
    Automatically logs all costs.
    """
    # Step 1: Chunk
    chunks = chunk_text(text, metadata, chunk_size, overlap, chunking_type)
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
