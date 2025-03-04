from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain.docstore.document import Document
from datetime import datetime
global vector_store

llm = ChatOpenAI(model="gpt-4o", streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
faiss_index_path = "faiss_index"

def initialize_vector_store(user_id: str):
    if os.path.exists(faiss_index_path) and os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
        try:
            return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Failed to load FAISS index: {e}. Rebuilding...")
    print("FAISS index not found or invalid. Creating new index...")
    initial_timestamp = datetime.now().isoformat()
    vector_store = FAISS.from_documents(
        [Document(page_content="", metadata={"timestamp": initial_timestamp, "source": "init", "user_id": user_id})],
        embeddings
    )
    vector_store.save_local(faiss_index_path)
    return vector_store
