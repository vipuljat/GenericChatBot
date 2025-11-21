import os
import logging
from typing import List, Dict, Tuple
from docx import Document
import PyPDF2
from fastapi import UploadFile
import config
from agent.agent import render_questions_from_file
import shutil

logger = logging.getLogger(__name__)

class DocumentProcessingService:
    """Service for processing and chunking documents"""
    
    def __init__(self):
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self.upload_dir = config.UPLOAD_DIR
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from PDF or DOCX file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension in ['.docx', '.doc']:
                return self._extract_from_docx(file_path)
            elif file_extension == '.txt':
                return self._extract_from_txt(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            raise
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            logger.info(f"Extracted {len(text)} characters from DOCX")
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX: {str(e)}")
            raise
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(f"Extracted {len(text)} characters from TXT")
            return text
        except Exception as e:
            logger.error(f"Error reading TXT: {str(e)}")
            raise
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to include with each chunk
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        if not text:
            return []
        
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                'chunk_index': len(chunks),
                'chunk_size': len(chunk_words),
                'text': chunk_text
            })
            
            chunks.append(chunk_metadata)
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def save_uploaded_file(self, upload_file: UploadFile) -> str:
        """
        Save uploaded file to disk
        
        Args:
            upload_file: FastAPI UploadFile object
            
        Returns:
            Path to saved file
        """
        try:
            filename = upload_file.filename
            file_path = os.path.join(self.upload_dir, filename)
            
            with open(file_path, "wb") as buffer:
                buffer.write(upload_file.file.read())
            
            logger.info(f"Saved file to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise
    
    def process_document(
        self,
        file_path: str,
        doc_id: int,
        chatbot_id: int,
        doc_name: str
    ) -> Tuple[List[str], List[Dict]]:
        """
        Process document: extract text, chunk it, and prepare metadata
        
        Args:
            file_path: Path to document
            doc_id: Document ID from database
            chatbot_id: Chatbot ID
            doc_name: Original document name
            
        Returns:
            Tuple of (chunk_texts, chunk_metadata_list)
        """
        try:
            # Extract text
            text = self.extract_text_from_file(file_path)
            
            if not text or len(text.strip()) == 0:
                logger.warning(f"No text extracted from {file_path}")
                return [], []
            
            # Prepare base metadata
            base_metadata = {
                'doc_id': doc_id,
                'chatbot_id': chatbot_id,
                'doc_name': doc_name,
                'file_path': file_path
            }
            
            # Chunk text
            chunks = self.chunk_text(text, base_metadata)
            
            # Separate text and metadata
            chunk_texts = [chunk['text'] for chunk in chunks]
            chunk_metadata = chunks
            
            logger.info(f"Processed document {doc_name}: {len(chunk_texts)} chunks")
            return chunk_texts, chunk_metadata
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

# Singleton instance
_document_service = None

def get_document_service() -> DocumentProcessingService:
    """Get or create document processing service instance"""
    global _document_service
    if _document_service is None:
        _document_service = DocumentProcessingService()
    return _document_service

async def render_quiz_questions(file: UploadFile) -> List[Dict]:
    """
    Render quiz questions from uploaded file
    
    Args:
        file: Uploaded quiz document
        
    Returns:
        List of rendered questions
    """
    try:
        doc_service = get_document_service()
        file_path = doc_service.save_uploaded_file(file)
        
        # Call the agent to render questions
        rendered_questions = render_questions_from_file(file_path)
        return rendered_questions
    except Exception as e:
        logger.error(f"Error rendering quiz questions: {str(e)}")
        raise
