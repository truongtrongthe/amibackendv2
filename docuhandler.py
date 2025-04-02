
import pdfplumber
from docx import Document
import uuid
from io import BytesIO
from utilities import logger
from database import save_training_with_chunk
import asyncio 
from werkzeug.datastructures import FileStorage  # Type hint for Flask file

async def process_document(file: FileStorage, user_id: str, mode: str = "default") -> bool:
    doc_id = str(uuid.uuid4())
    chunks = []

    # Determine file type
    file_extension = file.filename.split('.')[-1].lower()
    file_content = file.read()

    try:
        if file_extension == 'pdf':
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        words = text.split()
                        for j in range(0, len(words), 300):
                            chunk = " ".join(words[j:j + 300])
                            chunk_id = f"chunk_{i}_{j // 300}"
                            if not chunk_id:  # Double-check chunk_id
                                logger.error(f"Generated invalid chunk_id for page {i}, chunk {j}")
                                return False
                            chunks.append((chunk_id, chunk))

        elif file_extension == 'docx':
            doc = Document(BytesIO(file_content))
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            text = " ".join(full_text)
            if text:
                words = text.split()
                for j in range(0, len(words), 300):
                    chunk = " ".join(words[j:j + 300])
                    chunk_id = f"chunk_0_{j // 300}"
                    if not chunk_id:  # Double-check chunk_id
                        logger.error(f"Generated invalid chunk_id for chunk {j}")
                        return False
                    chunks.append((chunk_id, chunk))

        else:
            logger.error(f"Unsupported file type: {file_extension}")
            return False

        if not chunks:
            logger.warning(f"No text extracted from {file.filename}")
            return False

        # Save raw chunks and extract knowledge
        tasks = []
        for chunk_id, chunk_text in chunks:
            tasks.append(save_training_with_chunk(chunk_text, user_id, mode, doc_id, chunk_id, is_raw=True))
            tasks.append(save_training_with_chunk(chunk_text, user_id, mode, doc_id, chunk_id, is_raw=False))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) or not result:
                logger.error(f"Processing failed for one or more chunks: {result}")
                return False

        logger.info(f"Processed document {doc_id} with {len(chunks)} chunks from {file.filename}")
        return True

    except Exception as e:
        logger.error(f"Failed to process document: {e}")
        return False
    finally:
        file.close()