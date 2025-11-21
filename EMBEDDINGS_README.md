# Document Embeddings Pipeline

## Overview

This system automatically processes uploaded documents, generates embeddings using OpenAI, and stores them in Qdrant vector database for efficient semantic search.

## Architecture

```
Document Upload
     ↓
Extract Text (PDF/DOCX/TXT)
     ↓
Chunk Text (with overlap)
     ↓
Generate Embeddings (OpenAI)
     ↓
Store in Qdrant (with metadata)
     ↓
Link to Chatbot ID
```

## Components

### 1. Document Processing Service (`services/document_service.py`)
- **Purpose**: Extract and chunk document text
- **Supported Formats**: PDF, DOCX, TXT
- **Chunking Strategy**: Overlapping chunks for context preservation
- **Configuration**:
  - `CHUNK_SIZE`: Number of words per chunk (default: 1000)
  - `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

### 2. Embedding Service (`services/embedding_service.py`)
- **Purpose**: Generate vector embeddings using OpenAI
- **Model**: `text-embedding-3-small` (1536 dimensions)
- **Features**:
  - Batch processing for efficiency
  - Automatic retry on failures
  - Cost-effective model selection

### 3. Qdrant Service (`services/qdrant_service.py`)
- **Purpose**: Store and search embeddings
- **Features**:
  - Automatic collection creation
  - Metadata filtering by chatbot_id
  - Efficient similarity search
  - Batch upload support

### 4. Chatbot Service (`services/chatbot_service.py`)
- **Purpose**: Orchestrate the entire pipeline
- **Process**:
  1. Create chatbot in PostgreSQL
  2. Save uploaded documents
  3. Process documents and generate embeddings
  4. Store embeddings in Qdrant
  5. Link everything via chatbot_id

## Database Schema

### PostgreSQL Tables

#### chatbots_details
```sql
id                    INTEGER PRIMARY KEY
chatbot_name          VARCHAR
description           VARCHAR
instructions          VARCHAR
conversation_starters VARCHAR[]
is_quiz_mode          BOOLEAN
is_active             BOOLEAN
recommended_model     VARCHAR
```

#### chatbot_documents
```sql
id                 INTEGER PRIMARY KEY
chatbot_id         INTEGER FOREIGN KEY
doc_name           VARCHAR
file_path          VARCHAR
is_quiz_document   BOOLEAN
```

### Qdrant Collection

#### chatbot_documents (collection)
```python
{
    "id": "chatbot_id_doc_id_chunk_index",
    "vector": [1536 dimensions],
    "payload": {
        "chatbot_id": int,
        "doc_id": int,
        "doc_name": str,
        "file_path": str,
        "chunk_index": int,
        "chunk_size": int,
        "text": str
    }
}
```

## Setup Instructions

### 1. Qdrant Setup

**No Installation Required!** 

Qdrant now supports **local file-based storage** - the application will automatically:
- Create a `qdrant_storage` folder in your project
- Store all vectors locally (no server needed)
- Works out of the box with zero configuration

**Optional: Run Qdrant as a server**

If you prefer server mode:

**Option A: Docker**
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Standalone Binary**
Download from: https://github.com/qdrant/qdrant/releases
- Windows: `qdrant.exe`
- Linux/Mac: `./qdrant`

### 2. Configure Environment Variables

Add to `.env`:
```env
# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Qdrant Configuration (Local file storage - no server needed!)
QDRANT_URL=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=chatbot_documents
QDRANT_STORAGE_PATH=./qdrant_storage  # Local file-based storage

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
UPLOAD_DIR=uploaded_chatbots
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Application
```bash
uvicorn main:app --reload
```

## Usage

### Creating a Chatbot with Documents

**API Endpoint**: `POST /chatbot/chatbots`

**Request** (multipart/form-data):
```python
{
    "chatbot_name": "My AI Assistant",
    "description": "Helpful assistant",
    "instructions": "Answer questions based on documents",
    "conversation_starters": ["Hello", "Help"],
    "is_quiz_mode": false,
    "is_active": true,
    "recommended_model": "gpt-4",
    "file": <uploaded_file>,
    "quiz_file": <optional_quiz_file>
}
```

**Response**:
```json
{
    "message": "Chatbot created successfully",
    "chatbot_id": 123
}
```

**What Happens**:
1. Chatbot created in PostgreSQL
2. Document saved to disk
3. Text extracted from document
4. Text chunked into overlapping segments
5. Embeddings generated for each chunk
6. Embeddings stored in Qdrant with metadata
7. Document record created in PostgreSQL

### Searching for Similar Content

```python
from services.qdrant_service import get_qdrant_service
from services.embedding_service import get_embedding_service

# Generate query embedding
embedding_service = get_embedding_service()
query_embedding = embedding_service.generate_embedding("What is AI?")

# Search in Qdrant
qdrant_service = get_qdrant_service()
results = qdrant_service.search_similar(
    query_embedding=query_embedding,
    chatbot_id=123,
    limit=5
)

# Results contain relevant text chunks with metadata
for result in results:
    print(f"Score: {result['score']}")
    print(f"Text: {result['metadata']['text']}")
    print(f"Doc: {result['metadata']['doc_name']}")
```

## Data Flow

### Document Upload Flow
```
User uploads document
    ↓
FastAPI receives file
    ↓
chatbot_service.create_chatbot_with_documents()
    ↓
├─ Create chatbot in PostgreSQL
├─ Save file to disk (uploaded_chatbots/)
├─ Create document record in PostgreSQL
├─ document_service.process_document()
│   ├─ Extract text (PDF/DOCX/TXT)
│   └─ Chunk text (1000 words, 200 overlap)
├─ embedding_service.generate_embeddings_batch()
│   └─ Call OpenAI API
└─ qdrant_service.store_embeddings()
    └─ Store in Qdrant with chatbot_id filter
```

### Query Flow
```
User query: "What is machine learning?"
    ↓
Generate embedding for query
    ↓
Search Qdrant (filter by chatbot_id)
    ↓
Get top K similar chunks
    ↓
Use chunks as context for LLM
    ↓
Generate response
```

## Configuration

### Chunking Parameters

**CHUNK_SIZE** (default: 1000 words)
- Larger chunks: More context, fewer embeddings, higher costs
- Smaller chunks: More precise, more embeddings, lower costs

**CHUNK_OVERLAP** (default: 200 words)
- Ensures context continuity across chunk boundaries
- Helps capture information spanning chunk edges

### Embedding Model

**text-embedding-3-small**
- Dimensions: 1536
- Cost: ~$0.02 per 1M tokens
- Performance: Excellent for most use cases

**Alternative**: text-embedding-3-large
- Dimensions: 3072
- Cost: Higher
- Performance: Better for complex queries

## Monitoring & Debugging

### Check Qdrant Status
```bash
curl http://localhost:6333/collections
```

### View Collection Info
```bash
curl http://localhost:6333/collections/chatbot_documents
```

### Check Logs
```python
import logging
logging.basicConfig(level=logging.INFO)
```

Look for:
- "Processing document: {filename}"
- "Generated {n} embeddings"
- "Stored {n} embeddings for chatbot {id}"

## Cost Estimation

### OpenAI Embedding Costs

**text-embedding-3-small**: $0.02 per 1M tokens

Example calculation:
- 100-page document ≈ 50,000 words
- 50,000 words / 1000 per chunk = 50 chunks
- 50 chunks × 750 tokens avg = 37,500 tokens
- Cost: $0.0008 per document

### Qdrant Costs

- **Self-hosted**: Free (requires server resources)
- **Cloud**: Starting at $25/month

## Performance

### Processing Speed

Typical performance:
- Text extraction: ~1 second per page
- Chunking: ~0.1 seconds per document
- Embedding generation: ~0.5 seconds per batch (100 chunks)
- Qdrant storage: ~0.2 seconds per batch

**Total**: ~2-5 seconds for a 100-page document

### Optimization Tips

1. **Batch processing**: Process multiple documents concurrently
2. **Async operations**: Use async for API calls
3. **Caching**: Cache frequently accessed embeddings
4. **Index optimization**: Configure Qdrant HNSW parameters

## Troubleshooting

### Issue: "Qdrant connection failed"
**Solution**: Ensure Qdrant is running on correct host/port

### Issue: "OpenAI API error"
**Solution**: Check API key and rate limits

### Issue: "No text extracted"
**Solution**: Verify document format and file integrity

### Issue: "Embeddings not found"
**Solution**: Check chatbot_id filter and collection name

## Advanced Features

### 1. Custom Metadata
Add custom fields to embedding metadata:
```python
metadata = {
    'doc_id': doc_id,
    'chatbot_id': chatbot_id,
    'custom_field': 'custom_value',
    'tags': ['important', 'technical']
}
```

### 2. Hybrid Search
Combine vector search with keyword filters:
```python
results = qdrant_service.client.search(
    collection_name=collection_name,
    query_vector=embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="chatbot_id", match=MatchValue(value=123)),
            FieldCondition(key="tags", match=MatchAny(any=["important"]))
        ]
    )
)
```

### 3. Re-indexing
Update embeddings for existing documents:
```python
# Delete old embeddings
qdrant_service.delete_chatbot_embeddings(chatbot_id)

# Re-process documents
# (Automatically happens on new upload)
```

## Security Considerations

1. **API Keys**: Store securely in environment variables
2. **Access Control**: Implement authentication for API endpoints
3. **Data Privacy**: Consider encrypting stored documents
4. **Rate Limiting**: Protect against abuse

## Next Steps

1. ✅ Basic pipeline implemented
2. 🔄 Add retrieval endpoint for chatbot queries
3. 🔄 Implement caching for frequent queries
4. 🔄 Add analytics for embedding usage
5. 🔄 Support for more document formats (PPT, HTML, etc.)
6. 🔄 Implement incremental updates
7. 🔄 Add embedding versioning

## Support

For issues or questions:
- Check logs in console output
- Verify all environment variables are set
- Ensure Qdrant and OpenAI services are accessible
- Review API documentation at `/docs`
