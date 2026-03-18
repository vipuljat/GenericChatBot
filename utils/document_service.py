"""Document text extraction service supporting multiple formats."""

from typing import Optional, Tuple
import io
from pathlib import Path

from utils.logging import log


def _extract_pdf_pdfplumber(file_content: bytes) -> Optional[str]:
    """
    Extract text using pdfplumber — better layout, column order, and table preservation
    compared to PyPDF2.  Tables are extracted separately and appended as structured text
    so their rows are not merged into a single line.
    """
    import pdfplumber

    text_parts = []
    with pdfplumber.open(io.BytesIO(file_content)) as pdf:
        num_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages):
            try:
                # Extract prose/layout text
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

                # Extract tables as structured text (rows separated by newlines)
                table_blocks = []
                for table in (page.extract_tables() or []):
                    rows = []
                    for row in table:
                        if row:
                            cells = [str(c).strip() for c in row if c and str(c).strip()]
                            if cells:
                                rows.append(" | ".join(cells))
                    if rows:
                        table_blocks.append("\n".join(rows))

                # Append table text that isn't already captured in prose text
                combined = page_text.strip()
                for tblock in table_blocks:
                    # Only add if first 40 chars of table block not already in prose
                    if tblock[:40] not in combined:
                        combined = combined + "\n\n" + tblock if combined else tblock

                if combined:
                    text_parts.append(combined)

            except Exception as e:
                log.warning(f"pdfplumber error on page {page_num + 1}: {e}")

    if not text_parts:
        log.warning("pdfplumber: no text extracted")
        return None

    full_text = "\n\n".join(text_parts)
    log.info(f"Extracted {len(full_text)} chars from {num_pages} pages (pdfplumber)")
    return full_text


def _extract_pdf_pypdf2(file_content: bytes) -> Optional[str]:
    """PyPDF2-based extraction (fallback when pdfplumber is unavailable)."""
    from PyPDF2 import PdfReader

    reader = PdfReader(io.BytesIO(file_content))
    text_parts = []
    for page_num, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text)
        except Exception as e:
            log.warning(f"PyPDF2 error on page {page_num + 1}: {e}")

    if not text_parts:
        log.warning("PyPDF2: no text extracted")
        return None

    full_text = "\n\n".join(text_parts)
    log.info(f"Extracted {len(full_text)} chars from {len(reader.pages)} pages (PyPDF2 fallback)")
    return full_text


def extract_text_from_pdf(file_content: bytes) -> Optional[str]:
    """
    Extract text from PDF bytes.
    Primary: pdfplumber (better table/layout handling).
    Fallback: PyPDF2.
    """
    try:
        return _extract_pdf_pdfplumber(file_content)
    except ImportError:
        log.warning("pdfplumber not installed, falling back to PyPDF2")
    except Exception as e:
        log.warning(f"pdfplumber failed ({e}), falling back to PyPDF2")

    try:
        return _extract_pdf_pypdf2(file_content)
    except Exception as e:
        log.error(f"PDF extraction failed: {e}", exc_info=True)
        return None


def extract_text_from_docx(file_content: bytes) -> Optional[str]:
    """
    Extract text from DOCX bytes using python-docx.
    Paragraphs and tables are yielded in document order (not paragraphs-then-tables)
    so the extracted text matches the original document layout.
    """
    try:
        from docx import Document
        from docx.oxml.ns import qn

        doc = Document(io.BytesIO(file_content))
        text_parts = []

        for child in doc.element.body.iterchildren():
            tag = child.tag

            if tag == qn('w:p'):
                # Paragraph
                para_text = child.text_content().strip() if hasattr(child, 'text_content') else ""
                # Fallback: join all text runs manually
                if not para_text:
                    para_text = "".join(
                        node.text or ""
                        for node in child.iter()
                        if node.tag == qn('w:t')
                    ).strip()
                if para_text:
                    text_parts.append(para_text)

            elif tag == qn('w:tbl'):
                # Table — emit each row as a pipe-delimited line, blank line between rows
                row_texts = []
                for tr in child.iter(qn('w:tr')):
                    cells = []
                    for tc in tr.iter(qn('w:tc')):
                        cell_text = "".join(
                            node.text or ""
                            for node in tc.iter()
                            if node.tag == qn('w:t')
                        ).strip()
                        if cell_text:
                            cells.append(cell_text)
                    if cells:
                        row_texts.append(" | ".join(cells))
                if row_texts:
                    text_parts.append("\n".join(row_texts))

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