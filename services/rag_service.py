"""
RAG (Retrieval-Augmented Generation) Service
Handles query processing, embedding search, and response generation
"""
from typing import List, Dict, Optional
import logging
from services.embedding_service import get_embedding_service
# from services.qdrant_service import get_qdrant_service
import google.generativeai as genai
import config

logger = logging.getLogger(__name__)

class RAGService:
    """Service for retrieval-augmented generation"""
    
    def __init__(self):
        self.embedding_service = get_embedding_service()
        # self.qdrant_service = get_qdrant_service()
        genai.configure(api_key=config.GEMINI_API_KEY)
    
    def retrieve_relevant_context(
        self,
        query: str,
        chatbot_id: int,
        top_k: int = 5
    ) -> tuple[List[Dict], str]:
        """
        Retrieve relevant document chunks for a query
        
        Args:
            query: User's question
            chatbot_id: ID of the chatbot to search within
            top_k: Number of most relevant chunks to retrieve
            
        Returns:
            Tuple of (search_results, formatted_context)
        """
        try:
            # Generate embedding for the query
            logger.info(f"Generating embedding for query: {query[:50]}...")
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Search for similar chunks in Qdrant
            logger.info(f"Searching for relevant chunks in chatbot {chatbot_id}")
            search_results = self.qdrant_service.search_similar(
                query_embedding=query_embedding,
                chatbot_id=chatbot_id,
                limit=top_k
            )
            
            if not search_results:
                logger.warning(f"No relevant documents found for chatbot {chatbot_id}")
                return [], ""
            
            # Format context from search results
            context_parts = []
            for i, result in enumerate(search_results, 1):
                metadata = result['metadata']
                text = metadata.get('text', '')
                doc_name = metadata.get('doc_name', 'Unknown')
                score = result['score']
                
                context_parts.append(
                    f"[Document {i}: {doc_name} (Relevance: {score:.2f})]\n{text}\n"
                )
            
            formatted_context = "\n---\n".join(context_parts)
            logger.info(f"Retrieved {len(search_results)} relevant chunks")
            
            return search_results, formatted_context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return [], ""
    
    def generate_response(
        self,
        query: str,
        chatbot_id: int,
        chatbot_instructions: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict:
        """
        Generate a response using RAG
        
        Args:
            query: User's question
            chatbot_id: ID of the chatbot
            chatbot_instructions: Custom instructions for the chatbot
            conversation_history: Previous conversation messages
            model_name: Gemini model to use
            
        Returns:
            Dict with response and metadata
        """
        try:
            # Retrieve relevant context
            search_results, context = self.retrieve_relevant_context(
                query=query,
                chatbot_id=chatbot_id,
                top_k=5
            )
            
            if not context:
                return {
                    "response": "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing or ask something else.",
                    "sources": [],
                    "context_used": False
                }
            
            # Build the prompt
            system_prompt = chatbot_instructions or "You are a helpful assistant. Answer questions based on the provided context."
            
            prompt = f"""You are an AI assistant helping users with questions based on provided documents.

Context from relevant documents:
{context}

Instructions: {system_prompt}

User Question: {query}

Please provide a clear, accurate answer based ONLY on the information in the context above. If the context doesn't contain enough information to answer the question, say so clearly. Cite which document(s) you used."""
            
            # Resolve model name (prefer explicit param, then config)
            resolved_model = model_name or config.GEMINI_MODEL
            if resolved_model.endswith("-exp"):
                logger.warning(
                    "Experimental model requested; falling back to configured stable model"
                )
                resolved_model = config.GEMINI_MODEL
            logger.info(f"Generating response with {resolved_model}")

            response_obj = self._generate_with_retry(
                prompt=prompt,
                conversation_history=conversation_history,
                initial_model=resolved_model,
                max_retries=max_retries
            )
            
            # Extract source documents
            sources = [
                {
                    "doc_name": result['metadata'].get('doc_name', 'Unknown'),
                    "relevance_score": result['score'],
                    "chunk_index": result['metadata'].get('chunk_index', 0)
                }
                for result in search_results
            ]
            
            logger.info("Response generated successfully")
            return {
                "response": response_obj.text,
                "sources": sources,
                "context_used": True,
                "num_chunks_used": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": f"An error occurred while generating the response: {str(e)}",
                "sources": [],
                "context_used": False,
                "error": str(e)
            }


    # Helper methods inside class
    def _attempt_model(self, model_name_local: str, prompt: str, conversation_history: Optional[List[Dict[str, str]]]):
        m = genai.GenerativeModel(model_name_local)
        if conversation_history:
            chat = m.start_chat(history=[
                {"role": msg["role"], "parts": [msg["content"]]}
                for msg in conversation_history
            ])
            return chat.send_message(prompt)
        return m.generate_content(prompt)

    def _generate_with_retry(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]],
        initial_model: str,
        max_retries: int
    ):
        resolved_model = initial_model
        backoff = 2
        last_error = None
        import time
        for attempt in range(1, max_retries + 1):
            try:
                return self._attempt_model(resolved_model, prompt, conversation_history)
            except Exception as e:
                msg = str(e)
                last_error = e
                rate_limited = ("429" in msg) or ("quota" in msg.lower()) or ("rate" in msg.lower())
                if rate_limited and attempt < max_retries:
                    logger.warning(
                        f"Rate limit/quota issue (attempt {attempt}); sleeping {backoff}s then retrying."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    if attempt == 1 and resolved_model != "gemini-1.5-flash":
                        logger.info("Switching fallback model to gemini-1.5-flash due to rate limit.")
                        resolved_model = "gemini-1.5-flash"
                    continue
                logger.error(f"Gemini generation failed: {msg}")
                break
        raise last_error or RuntimeError("Failed after retries")

# Singleton instance
_rag_service = None

def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
