# """Embedding service with text chunking support."""

# import re
# from typing import List, Dict, Any, Optional
# from openai import OpenAI

# import config
# from utils.logging import log
# from utils.utilities import _estimate_embedding_cost


# # Module-level client (singleton)
# _client: Optional[OpenAI] = None
# _model: Optional[str] = None


# def _initialize_client():
#     """Initialize OpenAI client and model if not already initialized."""
#     global _client, _model
#     if _client is None:
#         if not config.OPENAI_API_KEY:
#             raise ValueError("OPENAI_API_KEY not configured")
#         _client = OpenAI(api_key=config.OPENAI_API_KEY)
#         _model = config.OPENAI_EMBEDDING_MODEL
#         log.info(f"Embedding service initialized with model: {_model}")


# class TextChunker:
#     DEFAULT_CHUNK_SIZE = 1200   # ← Change this
#     DEFAULT_OVERLAP = 200       # ← And this
#     def __init__(
#         self, 
#         chunk_size: int = DEFAULT_CHUNK_SIZE,
#         chunk_overlap: int = DEFAULT_OVERLAP
#     ):
#         """
#         Initialize text chunker.
        
#         Args:
#             chunk_size: Maximum characters per chunk (default: 6000)
#             chunk_overlap: Overlap between chunks (default: 500)
#         """
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
    
#     def chunk_text(
#         self, 
#         text: str, 
#         metadata: Optional[Dict[str, Any]] = None
#     ) -> List[Dict[str, Any]]:
#         """
#         Split text into overlapping chunks with metadata.
        
#         Args:
#             text: Text to chunk
#             metadata: Optional metadata to attach to each chunk
            
#         Returns:
#             List of dicts with 'text' and metadata for each chunk
#         """
#         if not text or not text.strip():
#             return []
        
#         # Clean the text
#         text = self._clean_text(text)
        
#         # Split into chunks
#         chunks = self._split_text_recursive(text)
        
#         # Prepare chunks with metadata
#         chunk_list = []
#         for i, chunk in enumerate(chunks):
#             chunk_data = {
#                 "text": chunk,
#                 "chunk_index": i,
#                 "total_chunks": len(chunks),
#                 "char_count": len(chunk),
#                 **(metadata or {})
#             }
#             chunk_list.append(chunk_data)
        
#         log.info(f"Split text into {len(chunks)} chunks")
#         return chunk_list
    
#     def _clean_text(self, text: str) -> str:
#         """Clean and normalize text."""
#         # Remove excessive whitespace
#         text = re.sub(r'\s+', ' ', text)
#         # Remove null bytes
#         text = text.replace('\x00', '')
#         return text.strip()
    
#     def _split_text_recursive(self, text: str) -> List[str]:
#         """
#         Recursively split text using multiple separators.
        
#         Priority: paragraphs → sentences → words → characters
#         """
#         if len(text) <= self.chunk_size:
#             return [text] if text.strip() else []
        
#         chunks = []
        
#         # Try splitting by paragraphs first
#         separators = [
#             '\n\n',   # Paragraphs
#             '\n',     # Lines
#             '. ',     # Sentences
#             '! ',     # Sentences
#             '? ',     # Sentences
#             '; ',     # Clauses
#             ', ',     # Phrases
#             ' ',      # Words
#             ''        # Characters (last resort)
#         ]
        
#         return self._split_by_separators(text, separators)
    
#     def _split_by_separators(
#         self, 
#         text: str, 
#         separators: List[str]
#     ) -> List[str]:
#         """Split text using a hierarchy of separators."""
#         if not separators:
#             # Fallback: split by character
#             return self._split_by_chars(text)
        
#         separator = separators[0]
#         remaining_separators = separators[1:]
        
#         if not separator:
#             # Empty separator means character-level split
#             return self._split_by_chars(text)
        
#         splits = text.split(separator)
#         chunks = []
#         current_chunk = []
#         current_length = 0
        
#         for split in splits:
#             split_length = len(split) + len(separator)
            
#             # If single split is too large, recursively split it
#             if split_length > self.chunk_size:
#                 if current_chunk:
#                     chunks.append(separator.join(current_chunk))
#                     current_chunk = []
#                     current_length = 0
                
#                 # Recursively split the large piece
#                 sub_chunks = self._split_by_separators(
#                     split, 
#                     remaining_separators
#                 )
#                 chunks.extend(sub_chunks)
#                 continue
            
#             # Check if adding this split exceeds chunk size
#             if current_length + split_length > self.chunk_size and current_chunk:
#                 # Save current chunk
#                 chunks.append(separator.join(current_chunk))
                
#                 # Start new chunk with overlap
#                 overlap_text = separator.join(current_chunk)
#                 if len(overlap_text) > self.chunk_overlap:
#                     overlap_splits = current_chunk[-(len(current_chunk)//2):]
#                     current_chunk = overlap_splits
#                     current_length = sum(len(s) for s in overlap_splits) + \
#                                    len(separator) * len(overlap_splits)
#                 else:
#                     current_chunk = []
#                     current_length = 0
            
#             current_chunk.append(split)
#             current_length += split_length
        
#         # Add remaining chunk
#         if current_chunk:
#             chunks.append(separator.join(current_chunk))
        
#         return chunks
    
#     def _split_by_chars(self, text: str) -> List[str]:
#         """Split text by characters with overlap (last resort)."""
#         chunks = []
#         start = 0
        
#         while start < len(text):
#             end = start + self.chunk_size
#             chunk = text[start:end]
#             chunks.append(chunk)
#             start += self.chunk_size - self.chunk_overlap
        
#         return chunks


# def chunk_text(
#     text: str, 
#     metadata: Optional[Dict[str, Any]] = None,
#     chunk_size: int = TextChunker.DEFAULT_CHUNK_SIZE,
#     chunk_overlap: int = TextChunker.DEFAULT_OVERLAP
# ) -> List[Dict[str, Any]]:
#     """
#     Convenience function to chunk text.
    
#     Args:
#         text: Text to chunk
#         metadata: Optional metadata for each chunk
#         chunk_size: Maximum characters per chunk
#         chunk_overlap: Overlap between chunks
        
#     Returns:
#         List of chunk dictionaries
#     """
#     chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return chunker.chunk_text(text, metadata)


# def generate_embedding(text: str) -> List[float]:
#     """
#     Generate embedding for a single text.
    
#     WARNING: Text must be under token limit (~8000 tokens).
#     Use chunk_text() first for long documents.
#     """
#     try:
#         _initialize_client()
        
#         # Safety check
#         if len(text) > 30000:  # ~7500 tokens
#             log.warning(
#                 f"Text length {len(text)} chars may exceed token limit. "
#                 "Consider using chunk_text() first."
#             )
        
#         response = _client.embeddings.create(model=_model, input=text)
#         return response.data[0].embedding
#     except Exception as e:
#         log.error(f"Error generating embedding: {str(e)}")
#         raise


# def generate_embeddings_batch(
#     texts: List[str],
#     max_batch_size: int = 100
# ) -> List[List[float]]:
#     try:
#         _initialize_client()

#         # 🔍 Estimate & log cost BEFORE API call
#         cost_info = _estimate_embedding_cost(texts)
#         log.info(
#             f"Embedding estimate → "
#             f"{cost_info['estimated_tokens']} tokens, "
#             f"${cost_info['estimated_cost_usd']} USD "
#             f"({cost_info['total_chars']} chars)"
#         )

#         all_embeddings = []

#         for i in range(0, len(texts), max_batch_size):
#             batch = texts[i:i + max_batch_size]

#             response = _client.embeddings.create(
#                 model=_model,
#                 input=batch
#             )

#             embeddings = [data.embedding for data in response.data]
#             all_embeddings.extend(embeddings)

#             log.info(
#                 f"Generated embeddings for batch "
#                 f"{i // max_batch_size + 1} "
#                 f"({len(batch)} chunks)"
#             )

#         log.info(f"Generated {len(all_embeddings)} total embeddings")
#         return all_embeddings

#     except Exception as e:
#         log.error(f"Error generating batch embeddings: {str(e)}")
#         raise


# def generate_embeddings_from_chunks(
#     chunks: List[Dict[str, Any]]
# ) -> List[Dict[str, Any]]:
#     """
#     Generate embeddings for chunked text.
    
#     Args:
#         chunks: List of chunk dicts with 'text' key
        
#     Returns:
#         Same chunks with added 'embedding' key
#     """
#     try:
#         texts = [chunk['text'] for chunk in chunks]
#         embeddings = generate_embeddings_batch(texts)
        
#         # Add embeddings to chunks
#         for chunk, embedding in zip(chunks, embeddings):
#             chunk['embedding'] = embedding
        
#         log.info(f"Generated embeddings for {len(chunks)} chunks")
#         return chunks
#     except Exception as e:
#         log.error(f"Error generating embeddings from chunks: {str(e)}")
#         raise


# def process_document_for_embedding(
#     text: str,
#     metadata: Optional[Dict[str, Any]] = None,
#     chunk_size: int = TextChunker.DEFAULT_CHUNK_SIZE,
#     chunk_overlap: int = TextChunker.DEFAULT_OVERLAP
# ) -> List[Dict[str, Any]]:
#     """
#     Complete pipeline: chunk text and generate embeddings.
    
#     Args:
#         text: Document text to process
#         metadata: Metadata to attach to each chunk
#         chunk_size: Maximum characters per chunk
#         chunk_overlap: Overlap between chunks
        
#     Returns:
#         List of chunks with embeddings and metadata
#     """
#     # Chunk the text
#     chunks = chunk_text(
#         text=text,
#         metadata=metadata,
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
    
#     if not chunks:
#         log.warning("No chunks generated from text")
#         return []
    
#     # Generate embeddings
#     chunks_with_embeddings = generate_embeddings_from_chunks(chunks)
    
#     log.info(
#         f"Processed document: {len(chunks)} chunks, "
#         f"{sum(c['char_count'] for c in chunks)} total chars"
#     )
    
#     return chunks_with_embeddings



"""Embedding service with text chunking support and Gemini fallback."""

import re
from typing import List, Dict, Any, Optional
from openai import OpenAI

import config
from utils.logging import log
from utils.utilities import _estimate_embedding_cost


# Module-level clients (singleton)
_openai_client: Optional[OpenAI] = None
_gemini_client: Optional[Any] = None
_openai_model: Optional[str] = None
_gemini_model: Optional[str] = None
_use_gemini: bool = False


def _initialize_openai():
    """Initialize OpenAI client."""
    global _openai_client, _openai_model
    if _openai_client is None:
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        _openai_model = config.OPENAI_EMBEDDING_MODEL
        log.info(f"OpenAI embedding service initialized with model: {_openai_model}")


def _initialize_gemini():
    """Initialize Gemini client as fallback."""
    global _gemini_client, _gemini_model
    if _gemini_client is None:
        try:
            import google.generativeai as genai
            
            if not hasattr(config, 'GEMINI_API_KEY') or not config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not configured")
            
            genai.configure(api_key=config.GEMINI_API_KEY)
            _gemini_client = genai
            _gemini_model = getattr(config, 'GEMINI_EMBEDDING_MODEL', 'models/text-embedding-004')
            log.info(f"Gemini embedding service initialized with model: {_gemini_model}")
        except Exception as e:
            log.error(f"Failed to initialize Gemini: {str(e)}")
            raise


def _initialize_client():
    """Initialize embedding client with fallback logic."""
    global _use_gemini
    
    try:
        _initialize_openai()
        _use_gemini = False
    except Exception as e:
        log.warning(f"OpenAI initialization failed: {str(e)}. Falling back to Gemini...")
        try:
            _initialize_gemini()
            _use_gemini = True
        except Exception as gemini_error:
            log.error(f"Gemini initialization also failed: {str(gemini_error)}")
            raise RuntimeError("Both OpenAI and Gemini embedding services failed to initialize")


class TextChunker:
    DEFAULT_CHUNK_SIZE = 1200
    DEFAULT_OVERLAP = 200
    
    def __init__(
        self, 
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_OVERLAP
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Maximum characters per chunk (default: 1200)
            chunk_overlap: Overlap between chunks (default: 200)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of dicts with 'text' and metadata for each chunk
        """
        if not text or not text.strip():
            return []
        
        # Clean the text
        text = self._clean_text(text)
        
        # Split into chunks
        chunks = self._split_text_recursive(text)
        
        # Prepare chunks with metadata
        chunk_list = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "char_count": len(chunk),
                **(metadata or {})
            }
            chunk_list.append(chunk_data)
        
        log.info(f"Split text into {len(chunks)} chunks")
        return chunk_list
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove null bytes
        text = text.replace('\x00', '')
        return text.strip()
    
    def _split_text_recursive(self, text: str) -> List[str]:
        """
        Recursively split text using multiple separators.
        
        Priority: paragraphs → sentences → words → characters
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        
        # Try splitting by paragraphs first
        separators = [
            '\n\n',   # Paragraphs
            '\n',     # Lines
            '. ',     # Sentences
            '! ',     # Sentences
            '? ',     # Sentences
            '; ',     # Clauses
            ', ',     # Phrases
            ' ',      # Words
            ''        # Characters (last resort)
        ]
        
        return self._split_by_separators(text, separators)
    
    def _split_by_separators(
        self, 
        text: str, 
        separators: List[str]
    ) -> List[str]:
        """Split text using a hierarchy of separators."""
        if not separators:
            # Fallback: split by character
            return self._split_by_chars(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if not separator:
            # Empty separator means character-level split
            return self._split_by_chars(text)
        
        splits = text.split(separator)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split) + len(separator)
            
            # If single split is too large, recursively split it
            if split_length > self.chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Recursively split the large piece
                sub_chunks = self._split_by_separators(
                    split, 
                    remaining_separators
                )
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding this split exceeds chunk size
            if current_length + split_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(separator.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = separator.join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    overlap_splits = current_chunk[-(len(current_chunk)//2):]
                    current_chunk = overlap_splits
                    current_length = sum(len(s) for s in overlap_splits) + \
                                   len(separator) * len(overlap_splits)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(split)
            current_length += split_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks
    
    def _split_by_chars(self, text: str) -> List[str]:
        """Split text by characters with overlap (last resort)."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        
        return chunks


def chunk_text(
    text: str, 
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = TextChunker.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = TextChunker.DEFAULT_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk text.
    
    Args:
        text: Text to chunk
        metadata: Optional metadata for each chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_text(text, metadata)


def _generate_openai_embedding(text: str) -> List[float]:
    """Generate embedding using OpenAI."""
    response = _openai_client.embeddings.create(model=_openai_model, input=text)
    return response.data[0].embedding


def _generate_gemini_embedding(text: str) -> List[float]:
    """Generate embedding using Gemini."""
    result = _gemini_client.embed_content(
        model=_gemini_model,
        content=text,
        task_type="retrieval_document"
    )
    return result['embedding']


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text with fallback.
    
    WARNING: Text must be under token limit (~8000 tokens).
    Use chunk_text() first for long documents.
    """
    try:
        _initialize_client()
        
        # Safety check
        if len(text) > 30000:  # ~7500 tokens
            log.warning(
                f"Text length {len(text)} chars may exceed token limit. "
                "Consider using chunk_text() first."
            )
        
        if _use_gemini:
            return _generate_gemini_embedding(text)
        else:
            try:
                return _generate_openai_embedding(text)
            except Exception as openai_error:
                log.warning(f"OpenAI embedding failed: {str(openai_error)}. Trying Gemini...")
                try:
                    _initialize_gemini()
                    return _generate_gemini_embedding(text)
                except Exception as gemini_error:
                    log.error(f"Gemini fallback also failed: {str(gemini_error)}")
                    raise openai_error
                    
    except Exception as e:
        log.error(f"Error generating embedding: {str(e)}")
        raise


def _generate_openai_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings batch using OpenAI."""
    response = _openai_client.embeddings.create(
        model=_openai_model,
        input=texts
    )
    return [data.embedding for data in response.data]


def _generate_gemini_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings batch using Gemini."""
    embeddings = []
    for text in texts:
        result = _gemini_client.embed_content(
            model=_gemini_model,
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result['embedding'])
    return embeddings


def generate_embeddings_batch(
    texts: List[str],
    max_batch_size: int = 100
) -> List[List[float]]:
    """Generate embeddings for multiple texts with fallback."""
    try:
        _initialize_client()

        # Estimate & log cost BEFORE API call (OpenAI only)
        if not _use_gemini:
            cost_info = _estimate_embedding_cost(texts)
            log.info(
                f"Embedding estimate → "
                f"{cost_info['estimated_tokens']} tokens, "
                f"${cost_info['estimated_cost_usd']} USD "
                f"({cost_info['total_chars']} chars)"
            )

        all_embeddings = []

        for i in range(0, len(texts), max_batch_size):
            batch = texts[i:i + max_batch_size]

            try:
                if _use_gemini:
                    embeddings = _generate_gemini_embeddings_batch(batch)
                else:
                    try:
                        embeddings = _generate_openai_embeddings_batch(batch)
                    except Exception as openai_error:
                        log.warning(f"OpenAI batch failed: {str(openai_error)}. Trying Gemini...")
                        _initialize_gemini()
                        embeddings = _generate_gemini_embeddings_batch(batch)
                
                all_embeddings.extend(embeddings)

                log.info(
                    f"Generated embeddings for batch "
                    f"{i // max_batch_size + 1} "
                    f"({len(batch)} chunks) using {'Gemini' if _use_gemini else 'OpenAI'}"
                )
                
            except Exception as batch_error:
                log.error(f"Batch {i // max_batch_size + 1} failed: {str(batch_error)}")
                raise

        log.info(f"Generated {len(all_embeddings)} total embeddings")
        return all_embeddings

    except Exception as e:
        log.error(f"Error generating batch embeddings: {str(e)}")
        raise


def generate_embeddings_from_chunks(
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for chunked text.
    
    Args:
        chunks: List of chunk dicts with 'text' key
        
    Returns:
        Same chunks with added 'embedding' key
    """
    try:
        texts = [chunk['text'] for chunk in chunks]
        embeddings = generate_embeddings_batch(texts)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        log.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks
    except Exception as e:
        log.error(f"Error generating embeddings from chunks: {str(e)}")
        raise


def process_document_for_embedding(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = TextChunker.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = TextChunker.DEFAULT_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Complete pipeline: chunk text and generate embeddings.
    
    Args:
        text: Document text to process
        metadata: Metadata to attach to each chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunks with embeddings and metadata
    """
    # Chunk the text
    chunks = chunk_text(
        text=text,
        metadata=metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if not chunks:
        log.warning("No chunks generated from text")
        return []
    
    # Generate embeddings
    chunks_with_embeddings = generate_embeddings_from_chunks(chunks)
    
    log.info(
        f"Processed document: {len(chunks)} chunks, "
        f"{sum(c['char_count'] for c in chunks)} total chars"
    )
    
    return chunks_with_embeddings