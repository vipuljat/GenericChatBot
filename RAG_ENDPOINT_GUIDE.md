# RAG Chat Endpoint - Quick Guide

## Overview
The chatbot now supports **Retrieval-Augmented Generation (RAG)** - it can answer questions by searching through uploaded documents and generating accurate responses based on the content.

## How It Works

1. **Upload documents** when creating a chatbot → Documents are automatically chunked and embedded
2. **Ask questions** via the chat endpoint → System retrieves relevant chunks and generates contextual responses
3. **Get cited sources** → Response includes which documents were used

## API Endpoints

### 1. Create Chatbot with Documents
```http
POST /chatbot/chatbots
Content-Type: multipart/form-data

Form Data:
- chatbot_name: "HR Assistant"
- description: "HR policy chatbot"
- instructions: "You are an HR assistant..."
- is_quiz_mode: false
- is_active: true
- recommended_model: "gemini-2.0-flash-exp"
- file: <upload file>
```

**Response:**
```json
{
  "message": "Chatbot created successfully",
  "chatbot_id": 1
}
```

### 2. Chat with Chatbot (RAG)
```http
POST /chatbot/chatbots/{chatbot_id}/chat
Content-Type: application/json

{
  "query": "How many leaves am I entitled to?",
  "conversation_history": [  // Optional
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ]
}
```

**Response:**
```json
{
  "response": "Based on the HR policy document, you are entitled to:\n1. Annual Leave: 25 days per year\n2. Sick Leave: 15 days per year\n3. Casual Leave: 10 days per year...",
  "sources": [
    {
      "doc_name": "hr_policy_document.txt",
      "relevance_score": 0.89,
      "chunk_index": 0
    }
  ],
  "context_used": true,
  "num_chunks_used": 3
}
```

### 3. Get Chatbot Details
```http
GET /chatbot/chatbots/{chatbot_id}
```

## Testing the RAG System

### Quick Test (Simple Query)
```bash
# 1. Start the server
uvicorn main:app --reload

# 2. Run quick test
python test_chat_endpoint.py
```

### Full Test (HR Chatbot Example)
```bash
# This will:
# - Create HR policy document
# - Upload and create HR chatbot
# - Test 7 different HR queries
# - Show responses with sources

python test_hr_chatbot.py
```

## Example Use Cases

### HR Chatbot
**Documents:** Employee handbook, leave policy, benefits guide  
**Queries:**
- "How many leaves am I entitled to?"
- "What is the remote work policy?"
- "What benefits do I get?"

### Product Support Chatbot
**Documents:** Product manuals, troubleshooting guides, FAQs  
**Queries:**
- "How do I reset my password?"
- "What's the warranty period?"
- "How to configure email settings?"

### Code Documentation Chatbot
**Documents:** API docs, developer guides, tutorials  
**Queries:**
- "How to authenticate API requests?"
- "What endpoints are available?"
- "Show me an example of..."

## How RAG Works Behind the Scenes

1. **Document Upload:**
   - Text extracted from PDF/DOCX/TXT
   - Split into chunks (1000 words with 200 overlap)
   - Each chunk converted to embedding using OpenAI
   - Embeddings stored in Qdrant with metadata

2. **Query Processing:**
   - User query converted to embedding
   - Qdrant searches for top 5 most similar chunks
   - Relevant chunks retrieved with relevance scores

3. **Response Generation:**
   - Retrieved chunks combined into context
   - Context + chatbot instructions + query sent to Gemini
   - AI generates response based ONLY on provided context
   - Sources cited in response

## Configuration

### Environment Variables (.env)
```bash
# Required
OPENAI_API_KEY=sk-your-key-here
GEMINI_API_KEY=your-gemini-key

# Optional (defaults shown)
QDRANT_STORAGE_PATH=./qdrant_storage
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Architecture

```
User Query
    ↓
Embedding Service (OpenAI)
    ↓
Qdrant Vector Search
    ↓
Top 5 Relevant Chunks Retrieved
    ↓
RAG Service (Context + Instructions)
    ↓
Gemini AI (Response Generation)
    ↓
Response + Sources
```

## File Storage

- **Uploaded documents:** `uploaded_chatbots/`
- **Vector embeddings:** `qdrant_storage/` (local file-based, no Docker needed)
- **Database records:** PostgreSQL (`chatbots_details`, `chatbot_documents` tables)

## Frontend Integration

To integrate with your frontend:

```javascript
// Create chatbot with document
const formData = new FormData();
formData.append('chatbot_name', 'HR Assistant');
formData.append('description', 'HR chatbot');
formData.append('instructions', 'You are an HR assistant...');
formData.append('file', documentFile);
formData.append('is_quiz_mode', 'false');
formData.append('is_active', 'true');

const response = await fetch('http://localhost:8000/chatbot/chatbots', {
  method: 'POST',
  body: formData
});
const { chatbot_id } = await response.json();

// Chat with chatbot
const chatResponse = await fetch(`http://localhost:8000/chatbot/chatbots/${chatbot_id}/chat`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'How many leaves am I entitled to?',
    conversation_history: [] // Optional
  })
});
const result = await chatResponse.json();
console.log(result.response); // AI response
console.log(result.sources);  // Source documents
```

## Troubleshooting

**Issue: No relevant documents found**
- Check if document was uploaded successfully
- Verify embeddings were generated (check logs)
- Try rephrasing the query

**Issue: Response is generic/not using context**
- Check chatbot instructions
- Verify document content is relevant to query
- Increase `top_k` parameter in RAG service

**Issue: Embeddings not stored**
- Verify OPENAI_API_KEY is set
- Check `qdrant_storage/` folder exists
- Review server logs for errors

## Cost Optimization

**OpenAI Embeddings (text-embedding-3-small):**
- Cost: $0.02 per 1M tokens
- ~750 words = 1000 tokens
- Example: 100-page document ≈ $0.50

**Gemini AI (gemini-2.0-flash-exp):**
- Free tier available
- Check Google AI Studio for pricing

## Next Steps

1. ✅ Upload documents via `/chatbots` endpoint
2. ✅ Test queries via `/chatbots/{id}/chat` endpoint
3. 🔄 Integrate with frontend
4. 🔄 Add conversation history support
5. 🔄 Implement chat session management
6. 🔄 Add feedback mechanism for response quality

## Support

For issues or questions:
- Check server logs: `uvicorn main:app --reload --log-level debug`
- Run tests: `python test_hr_chatbot.py`
- Review `EMBEDDINGS_README.md` for technical details
