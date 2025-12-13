"""Document text extraction service supporting multiple formats."""

from typing import Optional, Tuple
import io
from pathlib import Path

from utils.logging import log


def extract_text_from_pdf(file_content: bytes) -> Optional[str]:
    """
    Extract text from PDF bytes using PyPDF2.
    
    Args:
        file_content: PDF file content as bytes
        
    Returns:
        Extracted text or None if extraction fails
    """
    try:
        from PyPDF2 import PdfReader
        
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)
        
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                log.warning(f"Error extracting text from page {page_num + 1}: {e}")
                continue
        
        if not text_parts:
            log.warning("No text extracted from PDF")
            return None
        
        full_text = "\n\n".join(text_parts)
        log.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages (PDF)")
        
        return full_text
        
    except ImportError:
        log.error("PyPDF2 not installed. Install with: pip install PyPDF2")
        raise
    except Exception as e:
        log.error(f"Error extracting text from PDF: {e}", exc_info=True)
        return None


def extract_text_from_docx(file_content: bytes) -> Optional[str]:
    """
    Extract text from DOCX bytes using python-docx.
    
    Args:
        file_content: DOCX file content as bytes
        
    Returns:
        Extracted text or None if extraction fails
    """
    try:
        from docx import Document
        
        docx_file = io.BytesIO(file_content)
        doc = Document(docx_file)
        
        text_parts = []
        
        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        
        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if row_text.strip():
                    text_parts.append(row_text)
        
        if not text_parts:
            log.warning("No text extracted from DOCX")
            return None
        
        full_text = "\n\n".join(text_parts)
        log.info(f"Extracted {len(full_text)} characters from DOCX")
        
        return full_text
        
    except ImportError:
        log.error("python-docx not installed. Install with: pip install python-docx")
        raise
    except Exception as e:
        log.error(f"Error extracting text from DOCX: {e}", exc_info=True)
        return None


def extract_text_from_doc(file_content: bytes) -> Optional[str]:
    """
    Extract text from DOC (legacy Word) bytes using antiword or textract.
    
    Args:
        file_content: DOC file content as bytes
        
    Returns:
        Extracted text or None if extraction fails
    """
    try:
        import textract
        
        # textract can handle DOC files
        text = textract.process(file_content, extension='doc').decode('utf-8')
        
        if not text or not text.strip():
            log.warning("No text extracted from DOC")
            return None
        
        log.info(f"Extracted {len(text)} characters from DOC")
        return text
        
    except ImportError:
        log.error("textract not installed. Install with: pip install textract")
        log.info("Note: textract requires system dependencies (antiword, etc.)")
        return None
    except Exception as e:
        log.error(f"Error extracting text from DOC: {e}", exc_info=True)
        return None


def extract_text_from_txt(file_content: bytes) -> Optional[str]:
    """
    Extract text from TXT bytes.
    
    Args:
        file_content: TXT file content as bytes
        
    Returns:
        Extracted text or None if extraction fails
    """
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                text = file_content.decode(encoding)
                log.info(f"Extracted {len(text)} characters from TXT (encoding: {encoding})")
                return text
            except UnicodeDecodeError:
                continue
        
        log.error("Failed to decode TXT file with any encoding")
        return None
        
    except Exception as e:
        log.error(f"Error extracting text from TXT: {e}", exc_info=True)
        return None


def extract_text_from_rtf(file_content: bytes) -> Optional[str]:
    """
    Extract text from RTF bytes using striprtf.
    
    Args:
        file_content: RTF file content as bytes
        
    Returns:
        Extracted text or None if extraction fails
    """
    try:
        from striprtf.striprtf import rtf_to_text
        
        rtf_string = file_content.decode('utf-8', errors='ignore')
        text = rtf_to_text(rtf_string)
        
        if not text or not text.strip():
            log.warning("No text extracted from RTF")
            return None
        
        log.info(f"Extracted {len(text)} characters from RTF")
        return text
        
    except ImportError:
        log.error("striprtf not installed. Install with: pip install striprtf")
        return None
    except Exception as e:
        log.error(f"Error extracting text from RTF: {e}", exc_info=True)
        return None


def get_file_extension(filename: str) -> str:
    """Get lowercase file extension without dot."""
    return Path(filename).suffix.lower().lstrip('.')


def detect_document_type(file_content: bytes, filename: str) -> Tuple[str, str]:
    """
    Detect document type from filename and content.
    
    Args:
        file_content: File content as bytes
        filename: Original filename
        
    Returns:
        Tuple of (file_type, mime_type)
    """
    extension = get_file_extension(filename)
    
    # Map extensions to types
    type_map = {
        'pdf': ('pdf', 'application/pdf'),
        'docx': ('docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
        'doc': ('doc', 'application/msword'),
        'txt': ('txt', 'text/plain'),
        'rtf': ('rtf', 'application/rtf'),
    }
    
    if extension in type_map:
        return type_map[extension]
    
    # Try to detect from content magic bytes
    if file_content.startswith(b'%PDF'):
        return ('pdf', 'application/pdf')
    elif file_content.startswith(b'PK\x03\x04'):  # ZIP-based (DOCX, etc.)
        return ('docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
    elif file_content.startswith(b'\xd0\xcf\x11\xe0'):  # OLE2 (DOC)
        return ('doc', 'application/msword')
    elif file_content.startswith(b'{\\rtf'):
        return ('rtf', 'application/rtf')
    
    # Default to text
    return ('txt', 'text/plain')


def extract_text_from_document(file_content: bytes, filename: str) -> Optional[str]:
    """
    Extract text from any supported document format.
    
    Supported formats: PDF, DOCX, DOC, TXT, RTF
    
    Args:
        file_content: Document file content as bytes
        filename: Original filename for type detection
        
    Returns:
        Extracted text or None if extraction fails
    """
    try:
        # Detect document type
        doc_type, mime_type = detect_document_type(file_content, filename)
        
        log.info(f"Detected document type: {doc_type} for file: {filename}")
        
        # Route to appropriate extractor
        extractors = {
            'pdf': extract_text_from_pdf,
            'docx': extract_text_from_docx,
            'doc': extract_text_from_doc,
            'txt': extract_text_from_txt,
            'rtf': extract_text_from_rtf,
        }
        
        extractor = extractors.get(doc_type)
        
        if not extractor:
            log.error(f"No extractor available for type: {doc_type}")
            return None
        
        text = extractor(file_content)
        
        if text:
            log.info(f"Successfully extracted text from {filename} ({doc_type})")
        else:
            log.warning(f"No text extracted from {filename} ({doc_type})")
        
        return text
        
    except Exception as e:
        log.error(f"Error extracting text from document {filename}: {e}", exc_info=True)
        return None


def get_supported_extensions() -> list[str]:
    """Get list of supported file extensions."""
    return ['pdf', 'docx', 'doc', 'txt', 'rtf']


def is_supported_document(filename: str) -> bool:
    """Check if file extension is supported."""
    extension = get_file_extension(filename)
    return extension in get_supported_extensions()