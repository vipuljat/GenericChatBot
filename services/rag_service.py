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
    print("prompt", prompt)
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
    min_score: float = 0.5,
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

CRITICAL INSTRUCTIONS - Read Carefully:

1. QUERY TYPE DETECTION:
   - GREETING (hi, hello, hey, how are you, what's up): Respond warmly, mention you can help with company info
   - SMALL TALK (thank you, ok, I see, got it): Acknowledge briefly and ask if they need more help
   - KNOWLEDGE QUESTION: Use context strictly as described below
   - CLARIFICATION REQUEST (what do you mean, can you explain): Refer to previous context or ask what specifically they want clarified
   - OUT OF SCOPE (weather, sports, personal advice): Politely redirect to company-related topics

2. FOR KNOWLEDGE QUESTIONS ONLY:
   a) If context DIRECTLY answers the question:
      - Provide clear, accurate answer
      - Use natural language (don't say "according to the documents")
      - Be specific with numbers, dates, policies if present
   
   b) If context is PARTIALLY relevant but incomplete:
      - Answer what you CAN from context
      - Clearly state what information is missing
      - Example: "Based on company policy, X is required. However, I don't have information about Y in the documents. Please contact HR for complete details."
   
   c) If context is NOT relevant to the question:
      - Say: "I don't have information about that in the company documents."
      - Suggest 2-3 related topics you CAN help with from the context
      - Example: "I don't have information about remote work policies. I can help with: leave policies, working hours, or expense reimbursement."

3. HANDLING SPECIFIC EDGE CASES:

   a) COMPARISON QUESTIONS ("What's the difference between X and Y?"):
      - Only compare if BOTH are in context
      - If only one is present, explain that one and note the other isn't covered

   b) YES/NO QUESTIONS ("Can I do X?", "Is Y allowed?"):
      - Give definitive answer if context is clear
      - If ambiguous: "Based on the policy, it appears [likely/not], but I recommend confirming with HR"
      - If not covered: "I don't have specific information about this. Please check with HR."

   c) HYPOTHETICAL/SCENARIO QUESTIONS ("What if I...", "What happens when..."):
      - Answer ONLY if scenario is explicitly covered in context
      - Otherwise: "This specific scenario isn't covered in the documents. Please consult HR for guidance."

   d) RECENT CHANGES ("What's the new policy?", "Has this changed?"):
      - Provide the information from context
      - Add: "This is based on available documentation. For the most recent updates, please verify with HR."

   e) NUMERICAL/DATE QUESTIONS ("How many days?", "What's the deadline?"):
      - Provide EXACT numbers/dates from context
      - If approximate or unclear, say so explicitly
      - Never guess or estimate numbers

   f) MULTI-PART QUESTIONS ("Can I do X and also how about Y?"):
      - Address each part separately
      - If some parts aren't covered, be explicit about which ones

   g) FOLLOW-UP QUESTIONS (referencing previous conversation):
      - Use conversation history if available
      - If unclear what they're referring to: "Could you please clarify what you'd like to know more about?"

   h) CONTRADICTORY INFORMATION in context:
      - Acknowledge there are different pieces of information
      - Present both and suggest confirming with HR

   i) PERSONAL SITUATIONS ("I am in X situation, what should I do?"):
      - Provide general policy information from context
      - Always add: "For your specific situation, please consult with HR for personalized guidance."

   j) SENSITIVE TOPICS (harassment, discrimination, legal issues):
      - Provide factual policy information if in context
      - ALWAYS add: "For serious matters like this, please contact HR immediately or use the official reporting channels."

4. TONE AND LANGUAGE RULES:
   - Be professional but friendly
   - Never say "the documents say" or "according to the context"
   - Use confidence when information is clear, express uncertainty when it's not
   - Don't over-apologize (one "I don't have that information" is enough)
   - Keep responses concise but complete

5. STRICT PROHIBITIONS:
   - NEVER make up information not in context
   - NEVER give medical, legal, or financial advice beyond what's in policy docs
   - NEVER share personal information about other employees
   - NEVER make promises on behalf of the company
   - NEVER interpret ambiguous policies - direct to HR instead

6. QUALITY CHECKS:
   - If you're about to cite a number/date/policy, verify it's actually in the context
   - If you're unsure, express uncertainty rather than guessing
   - If the answer would require combining information in a complex way not directly stated, acknowledge limitations

7. ALWAYS FOLLOW UP:
   - Always ask clarifying questions if unclear
   - Always ask for confirmation if ambiguous

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