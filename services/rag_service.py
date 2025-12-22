"""Functional RAG service with cost tracking."""

import time
from typing import List, Dict, Optional, Any, Tuple
import google.generativeai as genai
import config
from utils.logging import log

# Import our functional services
from utils.embedding import generate_embedding
from services.vectore_store_service import search_similar
from utils.utilities import estimate_llm_cost

genai.configure(api_key=config.GEMINI_API_KEY)


# ============================================================================
# CONTEXT RETRIEVAL (NO LLM)
# ============================================================================

def retrieve_context(
    chatbot_name: str,
    query: str,
    top_k: int = 10,
    min_score: float = 0.55,
    max_context_chars: int = 4000,
    document_type: Optional[str] = None
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve relevant context from vector store.
    Returns: (formatted_context, source_chunks)
    
    NO LLM CALLS - JUST RETRIEVAL.
    Logs embedding cost.
    """
    # Generate query embedding ONCE with cost logging
    log.info(f"🔍 Retrieving context for: {query[:60]}...")
    query_vector = generate_embedding(query)  # Cost logged inside
    
    # Search vector store
    filters = {'document_type': document_type} if document_type else None
    chunks = search_similar(
        collection_name=chatbot_name,
        query_vector=query_vector,
        limit=top_k,
        min_score=min_score,
        filters=filters
    )
    
    if not chunks:
        log.info("No relevant context found")
        return "", []
    
    # Format context
    context_parts = []
    total_chars = 0
    valid_chunks = []
    
    for chunk in chunks:
        text = chunk.get('text', '').strip()
        if not text or len(text) < 20:
            continue
        
        source = chunk.get('source_file', 'Unknown')
        doc_type = chunk.get('document_type', 'unknown')
        
        chunk_text = f"[Source: {source} ({doc_type})]\n{text}\n"
        
        if total_chars + len(chunk_text) > max_context_chars:
            break
        
        context_parts.append(chunk_text)
        valid_chunks.append(chunk)
        total_chars += len(chunk_text)
    
    context = "\n---\n".join(context_parts)
    log.info(f"✓ Context: {total_chars} chars from {len(valid_chunks)} chunks")
    
    return context, valid_chunks


# ============================================================================
# LLM GENERATION WITH RETRY
# ============================================================================

def map_role_to_gemini(role: str) -> str:
    """Map role names to Gemini format."""
    role_lower = role.lower()
    if role_lower == 'user':
        return 'user'
    if role_lower in ['model', 'assistant', 'bot', 'ai', 'system']:
        return 'model'
    return 'user'


# def generate_with_retry(
#     prompt: str,
#     conversation_history: List[Dict[str, str]],
#     model_name: str,
#     max_retries: int = 3
# ):
#     """Generate response with retry logic for rate limits."""
#     backoff = 2
#     last_error = None
    
#     for attempt in range(1, max_retries + 1):
#         try:
#             model = genai.GenerativeModel(model_name)
            
#             if conversation_history:
#                 history = [
#                     {
#                         "role": map_role_to_gemini(msg["role"]),
#                         "parts": [msg["content"]]
#                     }
#                     for msg in conversation_history
#                 ]
#                 chat = model.start_chat(history=history)
#                 return chat.send_message(prompt)
            
#             return model.generate_content(prompt)
        
#         except Exception as e:
#             last_error = e
#             msg = str(e)
            
#             if ("429" in msg or "quota" in msg.lower() or "rate" in msg.lower()) and attempt < max_retries:
#                 log.warning(f"⚠️  Rate limit (attempt {attempt}), retrying in {backoff}s")
#                 time.sleep(backoff)
#                 backoff *= 2
#                 continue
            
#             break
    
#     raise last_error or RuntimeError("Generation failed after retries")


def generate_with_retry(
    prompt: str,
    conversation_history: List[Dict[str, str]],
    model_name: str,
    max_retries: int = 3
):
    """
    Generate response with retry logic and automatic cost tracking.
    
    Args:
        prompt: The prompt to send to the model
        conversation_history: Previous conversation messages
        model_name: Model identifier (e.g., gemini-1.5-pro)
        max_retries: Maximum retry attempts for rate limits
        
    Returns:
        Model response object with .text attribute
    """
    backoff = 2
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            model = genai.GenerativeModel(model_name)
            
            if conversation_history:
                history = [
                    {
                        "role": map_role_to_gemini(msg["role"]),
                        "parts": [msg["content"]]
                    }
                    for msg in conversation_history
                ]
                chat = model.start_chat(history=history)
                response = chat.send_message(prompt)
            else:
                response = model.generate_content(prompt)
            
            # Calculate and log cost
            cost = estimate_llm_cost(prompt, response.text, model_name)
    
            
            return response
        
        except Exception as e:
            last_error = e
            msg = str(e)
            
            if ("429" in msg or "quota" in msg.lower() or "rate" in msg.lower()) and attempt < max_retries:
                log.warning(f"⚠️  Rate limit (attempt {attempt}), retrying in {backoff}s")
                time.sleep(backoff)
                backoff *= 2
                continue
            
            break
    
    raise last_error or RuntimeError("Generation failed after retries")


def generate_rag_response(
    query: str,
    chatbot_name: str,
    chatbot_instructions: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    top_k: int = 10,
    min_score: float = 0.55,
    max_retries: int = 3,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete RAG pipeline - functional approach.
    
    Steps:
    1. Retrieve context (logs embedding cost)
    2. Generate response (ONE LLM call)
    3. Return with sources
    
    All costs automatically logged.
    """
    try:
        log.info(f"🤖 RAG query for '{chatbot_name}': {query[:60]}...")
        
        # Step 1: Retrieve context (embedding cost logged inside)
        context, source_chunks = retrieve_context(
            chatbot_name=chatbot_name,
            query=query,
            top_k=top_k,
            min_score=min_score
        )
        
        # Step 2: Build prompt
        base_instructions = chatbot_instructions or (
            "You are a helpful, knowledgeable assistant. "
            "Answer naturally and conversationally."
        )
        
        has_context = bool(context and context.strip())
        
        if has_context:
            prompt = f"""{base_instructions}

Relevant Context:
{context}

User Question: {query}

Instructions:
- Answer naturally without mentioning "the documents" or "the context"
- Use the context to provide accurate information
- If context doesn't fully answer, say so clearly
- Be concise and helpful

Response:"""
        else:
            prompt = f"""{base_instructions}

User Question: {query}

Respond naturally and helpfully:"""
        
        # Step 3: Generate response (ONE LLM CALL)
        resolved_model = model_name or config.GEMINI_MODEL
        log.info(f"💬 Generating with {resolved_model}...")
        
        response_obj = generate_with_retry(
            prompt=prompt,
            conversation_history=conversation_history or [],
            model_name=resolved_model,
            max_retries=max_retries
        )
        
        llm_cost = estimate_llm_cost(
            prompt=prompt,
            response=response_obj.text,
            model_name=resolved_model
        )

        
                # Step 4: Format sources
        sources = [
            {
                "source_file": chunk['source_file'],
                "document_type": chunk['document_type'],
                "relevance_score": round(chunk['score'], 3),
                "chunk_index": chunk['chunk_index']
            }
            for chunk in source_chunks
        ] if has_context else []
        
        log.info(f"✓ Response generated ({len(sources)} sources)")
        
        return {
            "response": response_obj.text.strip(),
            "sources": sources,
            "context_used": has_context,
            "num_chunks_used": len(source_chunks)
        }
    
    except Exception as e:
        log.error(f"❌ RAG failed: {e}", exc_info=True)
        return {
            "response": "I'm having trouble processing your request. Please try again.",
            "sources": [],
            "context_used": False,
            "error": str(e)
        }