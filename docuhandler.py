# docuhandler.py
import pdfplumber
from docx import Document
import uuid
from io import BytesIO
from utilities import logger
from database import save_training_with_chunk
import asyncio
from werkzeug.datastructures import FileStorage
from langchain_openai import ChatOpenAI

# Initialize LLM
LLM = ChatOpenAI(model="gpt-4o-mini", streaming=False)

async def summarize_with_llm(text: str, max_length: int = 150) -> str:
    """Summarize text using the LLM in the original language, aiming for a concise output."""
    try:
        prompt = (
            f"Summarize the following text in a concise manner, aiming for around {max_length} words. "
            "Focus on the main ideas and key points, and keep the summary in the same language as the input text:\n\n" + text
        )
        response = await LLM.ainvoke(prompt)
        summary = response.content.strip()
        logger.info(f"Generated summary (approx {len(summary.split())} words): {summary}")
        return summary
    except Exception as e:
        logger.error(f"LLM summarization failed: {e}")
        return "Summary unavailable due to processing error."

async def summarize_document(file: FileStorage, user_id: str, mode: str = "default") -> tuple[bool, str]:
    """
    Process a document: extract full text, summarize with LLM in original language, and chunk by word count (500 words).
    Returns (success: bool, summary: str). Does not save summary to database.
    """
    full_text = ""

    file_extension = file.filename.split('.')[-1].lower()
    try:
        logger.debug(f"Reading file content for {file.filename}")
        file_content = file.read()
        if not file_content:
            logger.warning(f"Empty file content for {file.filename}")
            return False, "Empty file content."

        # Step 1: Extract full text
        if file_extension == 'pdf':
            logger.debug("Opening PDF file")
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                full_text = " ".join(page.extract_text() or "" for page in pdf.pages if page.extract_text())
                if not full_text.strip():
                    logger.warning(f"No text extracted from {file.filename}")
                    return False, "No text extracted."
        elif file_extension == 'docx':
            logger.debug("Opening DOCX file")
            doc = Document(BytesIO(file_content))
            full_text = " ".join(para.text.strip() for para in doc.paragraphs if para.text.strip())
            if not full_text.strip():
                logger.warning(f"No text extracted from {file.filename}")
                return False, "No text extracted."
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            return False, f"Unsupported file type: {file_extension}"

        logger.debug(f"Extracted full_text (first 100 chars): {full_text[:100]}...")


        # Step 3: Summarize full text with LLM in original language (not saved)
        logger.debug("Summarizing text with LLM")
        summary = await summarize_with_llm(full_text)
        logger.info(f"Full text length: {len(full_text.split())} words")

        return summary

    except Exception as e:
        logger.error(f"Failed to process document: {type(e).__name__}: {str(e)} - File: {file.filename}")
        return False, "Processing error occurred."
    finally:
        file.close()

async def process_document(file: FileStorage, user_id: str, mode: str = "default", bank: str = "") -> bool:
        doc_id = str(uuid.uuid4())
        chunks = []

        logger.info(f"bank name at process-document={bank}")
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
                tasks.append(save_training_with_chunk(
                            input=chunk_text,
                            user_id=user_id,
                            mode=mode,
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            bank_name=bank,  # Explicitly pass bank to bank_name
                            is_raw=True
                        ))
                tasks.append(save_training_with_chunk(
                            input=chunk_text,
                            user_id=user_id,
                            mode=mode,
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            bank_name=bank,  # Explicitly pass bank to bank_name
                            is_raw=False
                        ))

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