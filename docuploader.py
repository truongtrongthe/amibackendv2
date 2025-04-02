import streamlit as st
import asyncio
from database import save_training_with_chunk,query_knowledge
import pdfplumber
import uuid
import logging
from io import BytesIO  # Explicitly import BytesIO
import docx  # Ensure docx is imported correctly

from langchain_openai import ChatOpenAI
from utilities import logger
from langchain_core.messages import HumanMessage

LLM = ChatOpenAI(model="gpt-4o-mini", streaming=True)
# Set up logging

async def process_document(file, user_id: str, mode: str = "default"):
    doc_id = str(uuid.uuid4())
    chunks = []
    
    # Determine file type and extract text
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        # Handle PDF files
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    words = text.split()
                    for j in range(0, len(words), 300):
                        chunk = " ".join(words[j:j+300])
                        chunks.append((f"chunk_{i}_{j//300}", chunk))
    
    elif file_extension == 'docx':
        # Handle Word documents
        try:
            # Read the file bytes and pass to docx.Document
            file_bytes = BytesIO(file.read())  # Convert UploadedFile to BytesIO
            doc = docx.Document(file_bytes)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            text = " ".join(full_text)
            if text:
                words = text.split()
                for j in range(0, len(words), 300):
                    chunk = " ".join(words[j:j+300])
                    chunks.append((f"chunk_0_{j//300}", chunk))
        except Exception as e:
            logger.error(f"Failed to process Word document: {e}")
            return False
    
    else:
        logger.error(f"Unsupported file type: {file_extension}")
        return False

    if not chunks:
        logger.warning(f"No text extracted from {file.name}")
        return False
    
    # Save raw chunks and extract knowledge
    tasks = []
    for chunk_id, chunk_text in chunks:
        tasks.append(save_training_with_chunk(chunk_text, user_id, mode=mode, doc_id=doc_id, chunk_id=chunk_id, is_raw=True))
        tasks.append(save_training_with_chunk(chunk_text, user_id, mode=mode, doc_id=doc_id, chunk_id=chunk_id))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception) or not result:
            logger.error(f"Processing failed for one or more chunks: {result}")
            return False
    
    logger.info(f"Processed document {doc_id} with {len(chunks)} chunks from {file.name}")
    return True

async def generate_answer(user_id: str, query: str, top_k: int = 5) -> str:
    # Retrieve relevant chunks
    knowledge = await query_knowledge(user_id, query, top_k)
    
    if not knowledge:
        return "No relevant information found."
    
    # Combine retrieved chunks into a context
    context = "\n\n".join([entry["raw"] for entry in knowledge])
    prompt = f"Based on the following information:\n{context}\n\nAnswer this question: {query}"
    
    # Call an LLM (assuming inferLLM is your LLM interface)
    try:
        response = await asyncio.to_thread(LLM.invoke, prompt)
        answer = response.content.strip()
        logger.info(f"Generated answer for '{query}': {answer}")
        return answer
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return "Failed to generate an answer."

def main():
    st.title("Document Processor & Knowledge Query")
    
    # User ID input
    user_id = st.text_input("Enter User ID", "user123")
    
    # File uploader section

    uploaded_file = st.file_uploader("Choose a PDF or Word file", type=["pdf", "docx"])
    if st.button("Process Document") and uploaded_file is not None:
        with st.spinner("Processing document..."):
            result = asyncio.run(process_document(uploaded_file, user_id))
            if result:
                st.success("Document processed successfully!")
            else:
                st.error("Failed to process document.")
    
    # Query section
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question")
    if st.button("Search Knowledge") and query:
        with st.spinner("Searching and generating answer..."):
            answer = asyncio.run(generate_answer(user_id, query))
            st.write("**Answer:**")
            st.write(answer)
    
    # Optional: Display file details
    if uploaded_file is not None:
        file_details = {"Filename": uploaded_file.name, "Filesize": f"{uploaded_file.size / 1024:.2f} KB"}
        st.write("File Details:")
        st.json(file_details)

if __name__ == "__main__":
    main()