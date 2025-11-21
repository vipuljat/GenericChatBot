# Quick Start Guide - Embeddings Pipeline

## Prerequisites

1. **PostgreSQL** running on localhost:5432
2. **OpenAI API Key**
3. **Python 3.8+** with pip

## Setup Steps

### 1. Qdrant Setup (No Docker Required!)

**Good News!** Qdrant now works with **local file-based storage** - no Docker needed!

The application will automatically create a `qdrant_storage` folder in your project directory and use it for vector storage.

**Configuration:**
- Local storage is automatically used when no Qdrant server is detected
- Data stored in `./qdrant_storage/` folder
- No additional installation required!

**Alternative: If you have Docker (Optional)**
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

**Alternative: Standalone Qdrant Binary (Optional)**
Download from: https://github.com/qdrant/qdrant/releases
```powershell
# Windows
.\qdrant.exe
```

The system automatically detects which mode to use:
1. ✅ **Local file storage** (no server) - Default, works out of the box
2. Qdrant server (if running on localhost:6333)
3. Remote/Cloud Qdrant (if API key provided)

### 2. Configure Environment

Update `.env` file:
```env
# Add your OpenAI API key
OPENAI_API_KEY=sk-your-key-here

# Qdrant is already configured for localhost
QDRANT_URL=localhost
QDRANT_PORT=6333
```

### 3. Install Dependencies

```bash
cd ChatBot
pip install -r requirements.txt
```

### 4. Start Backend

```bash
uvicorn main:app --reload
```

The server will start at `http://localhost:8000`

Visit `http://localhost:8000/docs` for API documentation

## Testing the Pipeline

### 1. Test with cURL

```bash
curl -X POST "http://localhost:8000/chatbot/chatbots" \
  -H "Content-Type: multipart/form-data" \
  -F "chatbot_name=Test Bot" \
  -F "description=A test chatbot" \
  -F "instructions=Answer questions from the document" \
  -F "is_quiz_mode=false" \
  -F "is_active=true" \
  -F "file=@/path/to/your/document.pdf"
```

### 2. Test with Python

```python
import requests

url = "http://localhost:8000/chatbot/chatbots"

files = {
    'file': open('document.pdf', 'rb')
}

data = {
    'chatbot_name': 'Test Bot',
    'description': 'A test chatbot',
    'instructions': 'Answer questions from the document',
    'is_quiz_mode': False,
    'is_active': True
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### 3. What to Expect

**Console Output:**
```
INFO - Creating chatbot: Test Bot
INFO - Created chatbot with ID: 1
INFO - Processing main document: document.pdf
INFO - Processing document: document.pdf
INFO - Extracted 5000 characters from PDF
INFO - Created 5 chunks from text
INFO - Generated 5 embeddings
INFO - Storing embeddings in Qdrant
INFO - Stored 5 embeddings for chatbot 1
INFO - Chatbot 1 created successfully with embeddings
```

**API Response:**
```json
{
    "message": "Chatbot created successfully",
    "chatbot_id": 1
}
```

## Verify Storage

### Check PostgreSQL

```sql
-- Connect to database
psql -U postgres -d GenericChatbotDB

-- Check chatbot created
SELECT * FROM chatbots_details;

-- Check document saved
SELECT * FROM chatbot_documents;
```

### Check Qdrant

**For local file storage (default):**
```powershell
# Check storage folder exists
Test-Path .\qdrant_storage

# View storage contents
Get-ChildItem .\qdrant_storage
```

**If using Qdrant server:**
```bash
# View collection
curl http://localhost:6333/collections/chatbot_documents

# Count points
curl http://localhost:6333/collections/chatbot_documents/points/count
```

## Troubleshooting

### Issue: "Qdrant connection error"

**Solution:** The system automatically uses local file storage (no server needed).

**Check storage folder:**
```powershell
# Should see qdrant_storage folder created
ls qdrant_storage
```

**If you want to use Qdrant server instead:**
1. Download from: https://github.com/qdrant/qdrant/releases
2. Run: `.\qdrant.exe` (Windows) or `./qdrant` (Linux/Mac)
3. Server will start on `localhost:6333`

### Issue: "OpenAI API error"

**Check API key:**
```bash
echo $OPENAI_API_KEY
```

**Update .env:**
```env
OPENAI_API_KEY=sk-your-actual-key-here
```

### Issue: "Database connection error"

**Check PostgreSQL:**
```bash
pg_isready -h localhost -p 5432
```

**Update .env with correct credentials:**
```env
DATABASE_URL=postgresql://user:password@localhost:5432/GenericChatbotDB
```

### Issue: "No text extracted from document"

**Supported formats:**
- PDF (.pdf)
- Word (.docx)
- Text (.txt)

**Check file integrity:**
- Ensure file is not corrupted
- Try opening in native application
- Check file permissions

## Next Steps

1. **Create a retrieval endpoint** for searching chatbot documents
2. **Integrate with LLM** to generate responses using retrieved context
3. **Add more document formats** (PowerPoint, HTML, Markdown)
4. **Implement caching** for frequently accessed embeddings
5. **Add monitoring** for embedding generation costs

## API Endpoints

### Create Chatbot with Documents
```
POST /chatbot/chatbots
```

Parameters:
- `chatbot_name` (string, required)
- `description` (string, required)
- `instructions` (string, required)
- `conversation_starters` (array[string], optional)
- `is_quiz_mode` (boolean, default: false)
- `is_active` (boolean, default: true)
- `recommended_model` (string, optional)
- `file` (file, required) - Main document
- `quiz_file` (file, optional) - Quiz document

### Render Quiz Questions
```
POST /chatbot/quiz/render
```

Parameters:
- `file` (file, required) - Quiz document

## Cost Estimation

**OpenAI Embeddings (text-embedding-3-small):**
- Cost: $0.02 per 1M tokens
- Average document (100 pages): ~$0.001

**Example:**
- 10 documents/day × 30 days = 300 documents/month
- Cost: ~$0.30/month for embeddings

**Qdrant:**
- Self-hosted: Free (requires server)
- Cloud: Starting at $25/month

## Support

- **API Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Logs**: Check terminal output for detailed logs
- **Documentation**: See `EMBEDDINGS_README.md` for detailed info

## Example Workflow

1. **Upload Document** → Creates chatbot + generates embeddings
2. **User asks question** → Generate query embedding
3. **Search Qdrant** → Find relevant document chunks
4. **Send to LLM** → Generate response with context
5. **Return answer** → User gets accurate, document-based response

That's it! Your embedding pipeline is ready to use! 🚀
